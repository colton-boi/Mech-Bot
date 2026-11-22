//! [`MovePlanner`]: picks this decision's `move_dir` and `boost` by simulating
//! 33 short candidate moves (16 headings and a stop, each walked and, when
//! allowed, boosted) for 0.6 s with the server's kinematics and scoring each
//! trajectory on dodging, range, cover, cubes, walls, and the zone.
//!
//! Everything the planner asks of the map and of memory goes through two
//! private traits, [`MoveEnv`] and [`Recall`], so the core is testable with a
//! mock arena. [`MovePlanner::plan`] is a thin adapter over
//! [`ApexNav`] and [`BotMemory`]. A decision allocates nothing.

use super::genome::{ApexGenome, LaserJukeMode, StrafeStyle};
use super::memory::BotMemory;
use super::nav::ApexNav;
use super::types::{clamp, dist_xy, wrap_pi, ItemKind, Mode, MoveDecision, NavGoal, Rng64, ThreatInfo};
use proto::{Obs, ProjectileKind, WeaponKind, WeaponObs};
use std::f64::consts::PI;
use world::{Bounds, NO_HINT};

/// Simulation step, seconds: one decision (`tick_hz` 20, `decide_every` 2).
const DT: f64 = 0.1;
/// Steps simulated per candidate (0.6 s).
const STEPS: usize = 6;
/// Headings evaluated, every 22.5°.
const HEADINGS: usize = 16;
/// Candidates: every heading walked and boosted, plus one stop.
const CANDIDATES: usize = 2 * HEADINGS + 1;
/// Bullets arriving later than this are ignored, seconds.
const BULLET_HORIZON: f64 = 0.8;
/// Miss distance beyond the body radius that still counts as a hit, meters.
const BULLET_MARGIN: f64 = 1.2;
/// Mortar shell gravity, m/s².
const MORTAR_G: f64 = 22.0;
/// Shells landing later than this are ignored, seconds.
const MORTAR_HORIZON: f64 = 3.5;
/// `config.weapons.mortar.splash_radius` default, used when we hold no mortar.
const MORTAR_SPLASH_R: f64 = 7.0;
/// Extra distance kept from a predicted mortar blast, meters.
const MORTAR_MARGIN: f64 = 2.5;
/// `config.cubes.pickup_radius` default, meters.
const PICKUP_RADIUS: f64 = 3.5;
/// Extra distance kept from cubes when at the cube cap, meters.
const CUBE_MARGIN: f64 = 1.5;
/// Clearance below this is penalized, meters.
const LOW_CLEARANCE: f64 = 3.2;
/// An enemy laser aimed within this of us is a threat, radians.
const LASER_AIM_TOL: f64 = 0.18;
/// Quiet walk speed, below `sound.step_speed` (5), m/s.
const QUIET_SPEED: f64 = 4.6;
/// How far ahead a strafe leg checks for a wall before flipping, meters.
const STRAFE_LOOK_M: f64 = 2.5;
/// Fixed capacities, so a decision allocates nothing.
const MAX_BULLETS: usize = 24;
const MAX_SHELLS: usize = 6;
const MAX_CHARGERS: usize = 2;
const MAX_WATCHERS: usize = 2;
const MAX_CUBES: usize = 16;
const MAX_ARC_THREATS: usize = 4;
const MAX_SEP: usize = 4;
/// FFA: threats closer than this are kept apart from and hidden from, meters.
const SEP_RANGE: f64 = 145.0;
/// Radius of the ring sampled to find a peekable LOS edge, meters.
const PEEK_RING_M: f64 = 6.0;
/// Velocity-continuity bonus is off this long after a strafe flip, seconds.
const FLIP_SETTLE_S: f64 = 0.3;
/// The final shrink: a next radius this small pins us to its center, meters.
const FINAL_RADIUS: f64 = 12.0;

// ---------------------------------------------------------------------------
// The planner's view of the map and of memory.

/// The map queries the planner needs.
trait MoveEnv {
    fn walkable(&self, x: f64, y: f64) -> bool;
    /// Distance to the nearest obstacle, meters.
    fn clearance(&self, x: f64, y: f64) -> f64;
    /// How exposed a spot is; larger is more exposed.
    fn exposure(&self, x: f64, y: f64) -> f64;
    /// A wall or ledge just ahead of `pos` along `dir`.
    fn step_blocked(&self, pos: [f64; 3], dir: [f64; 2]) -> bool;
    /// Line of sight, with a per-line hint as in [`world::World::los_hinted`].
    fn los(&self, a: [f64; 3], b: [f64; 3], hint: &mut u32) -> bool;
    fn bounds(&self) -> Bounds;
}

/// The memory queries the planner needs.
trait Recall {
    fn last_damage_t(&self) -> f64;
    fn is_ffa(&self) -> bool;
    /// Pushes remembered cube positions.
    fn cubes(&self, out: &mut Points<MAX_CUBES>);
}

struct NavEnv<'a>(&'a ApexNav);

impl MoveEnv for NavEnv<'_> {
    fn walkable(&self, x: f64, y: f64) -> bool {
        self.0.is_walkable_xy(x, y)
    }
    fn clearance(&self, x: f64, y: f64) -> f64 {
        self.0.clearance_at(x, y)
    }
    fn exposure(&self, x: f64, y: f64) -> f64 {
        self.0.exposure_at(x, y)
    }
    fn step_blocked(&self, pos: [f64; 3], dir: [f64; 2]) -> bool {
        self.0.step_blocked_ahead(pos, dir, STRAFE_LOOK_M)
    }
    fn los(&self, a: [f64; 3], b: [f64; 3], hint: &mut u32) -> bool {
        self.0.world().los_hinted(a, b, hint)
    }
    fn bounds(&self) -> Bounds {
        self.0.world().bounds
    }
}

struct MemRecall<'a>(&'a BotMemory);

impl Recall for MemRecall<'_> {
    fn last_damage_t(&self) -> f64 {
        self.0.last_damage_t()
    }
    fn is_ffa(&self) -> bool {
        self.0.is_ffa()
    }
    fn cubes(&self, out: &mut Points<MAX_CUBES>) {
        for it in self.0.items() {
            if it.kind == ItemKind::Cube {
                out.push([it.pos[0], it.pos[1]]);
            }
        }
    }
}

/// A fixed-capacity list of 2-D points; pushes past capacity are dropped.
struct Points<const N: usize> {
    pts: [[f64; 2]; N],
    len: usize,
}

impl<const N: usize> Points<N> {
    fn new() -> Self {
        Self { pts: [[0.0; 2]; N], len: 0 }
    }
    fn push(&mut self, p: [f64; 2]) {
        if self.len < N {
            self.pts[self.len] = p;
            self.len += 1;
        }
    }
    fn as_slice(&self) -> &[[f64; 2]] {
        &self.pts[..self.len]
    }
}

// ---------------------------------------------------------------------------
// Small helpers.

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn norm(v: [f64; 2]) -> [f64; 2] {
    let l = v[0].hypot(v[1]);
    if l > 1e-9 {
        [v[0] / l, v[1] / l]
    } else {
        [0.0, 0.0]
    }
}

fn xy(p: [f64; 3]) -> [f64; 2] {
    [p[0], p[1]]
}

fn own_holds(obs: &Obs, kind: WeaponKind) -> bool {
    obs.me.weapons.iter().flatten().any(|w| w.kind() == kind)
}

fn threat_holds(t: &ThreatInfo, kind: WeaponKind) -> bool {
    t.weapons.contains(&Some(kind))
}

/// Our laser is charging (movement is slowed by `charge_move_mult`).
fn own_laser_charging(obs: &Obs) -> bool {
    obs.me
        .weapons
        .iter()
        .flatten()
        .any(|w| matches!(w, WeaponObs::Laser { charge, .. } if *charge > 0.0))
}

/// The enemy's laser is charging and pointed within [`LASER_AIM_TOL`] of `at`.
fn laser_aimed_at(t: &ThreatInfo, at: [f64; 2]) -> bool {
    if !t.charging || t.is_ghost || !t.visible_now {
        return false;
    }
    let to = sub(at, xy(t.pos));
    wrap_pi(to[1].atan2(to[0]) - t.aim[0]).abs() <= LASER_AIM_TOL
}

/// A LOS edge to `body` lies within reach of `pos`: sampling `pos` and a ring
/// of [`PEEK_RING_M`] around it finds both a spot that sees `body` and one
/// that does not. In an open field there is nothing to peek from.
fn peek_edge_nearby<E: MoveEnv>(env: &E, pos: [f64; 3], eye_h: f64, body: [f64; 3], hint: &mut u32) -> bool {
    let (mut seen, mut hidden) = (false, false);
    for k in 0..=8 {
        let p = if k == 0 {
            xy(pos)
        } else {
            let a = (k - 1) as f64 * (PI / 4.0);
            [pos[0] + PEEK_RING_M * a.cos(), pos[1] + PEEK_RING_M * a.sin()]
        };
        if k > 0 && !env.walkable(p[0], p[1]) {
            continue;
        }
        if env.los([p[0], p[1], pos[2] + eye_h], body, hint) {
            seen = true;
        } else {
            hidden = true;
        }
        if seen && hidden {
            return true;
        }
    }
    false
}

/// Strafe leg bounds in seconds and the chance to flip at a leg's end.
fn cadence(style: StrafeStyle) -> (f64, f64, f64) {
    match style {
        StrafeStyle::FastJink => (0.35, 0.90, 0.85),
        StrafeStyle::Mid12s => (1.0, 2.4, 0.70),
        StrafeStyle::Steady35s => (3.0, 5.0, 0.70),
    }
}

/// The fighting range we want against `target`.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RangeBand {
    /// Preferred distance, meters.
    pref: f64,
    /// Pushing in on a laser that just fired (`cdwin_push`).
    cdwin: bool,
    /// The band is our lightning's (we close to brawl).
    lightning: bool,
}

/// Matchup-aware: our weapons against the target's, finishing a weak target
/// up close in Practice, and standing off when a crowd is near.
fn preferred_range(obs: &Obs, mem_ffa: bool, genome: &ApexGenome, target: &ThreatInfo, threats: &[ThreatInfo]) -> RangeBand {
    let ffa_posture = mem_ffa || (genome.quiet_walk && obs.alive > 3);
    let have_lightning = own_holds(obs, WeaponKind::Lightning);
    let enemy_lightning = threat_holds(target, WeaponKind::Lightning);
    let healthy = obs.me.max_hp <= 0.0 || obs.me.hp >= 0.5 * obs.me.max_hp;
    let others_within = |r: f64| {
        threats
            .iter()
            .filter(|th| th.id != target.id && !th.is_ghost && obs.t - th.last_seen_t < 3.0 && th.dist < r)
            .count()
    };
    let mut band = RangeBand {
        pref: genome.gun_range,
        cdwin: false,
        lightning: false,
    };
    let safe_brawl =
        !threat_holds(target, WeaponKind::Laser) || target.dist < 62.0 || target.hp < 360.0 || (!target.charging && obs.t - target.last_beam_t <= 1.5);
    if have_lightning && obs.me.energy > 30.0 && (!enemy_lightning || target.hp < 320.0) && safe_brawl {
        band.pref = 28.0;
        band.lightning = true;
    } else if own_holds(obs, WeaponKind::Laser) {
        band.pref = genome.laser_range;
        if enemy_lightning && !have_lightning {
            band.pref = band.pref.max(72.0);
        }
    } else {
        if enemy_lightning {
            band.pref = band.pref.max(72.0);
        }
        // Gun against a laser in the open loses; keep out of its sweet spot.
        if !ffa_posture && threat_holds(target, WeaponKind::Laser) {
            band.pref = band.pref.max(95.0);
        }
        if !ffa_posture && threat_holds(target, WeaponKind::Mortar) && healthy {
            band.pref = band.pref.min(45.0);
        }
    }
    // Practice: finish a weak, isolated target up close.
    let target_weak = target.max_hp > 0.0 && target.hp < 0.35 * target.max_hp;
    let me_ok = obs.me.max_hp <= 0.0 || obs.me.hp > 0.45 * obs.me.max_hp;
    if !ffa_posture && target_weak && me_ok && target.dist < 90.0 && others_within(140.0) == 0 {
        band.pref = 22.0;
    }
    // FFA standoff: shift into 95..130 m when enemy holds Laser/Mortar or another threat is within 160 m.
    if ffa_posture
        && !(band.lightning && target.hp < 320.0)
        && (threat_holds(target, WeaponKind::Laser) || threat_holds(target, WeaponKind::Mortar) || others_within(160.0) >= 1)
    {
        band.pref = band.pref.clamp(95.0, 130.0);
        band.lightning = false;
    }
    // A crowd nearby: stand off rather than brawl in the middle of it.
    if others_within(150.0) >= 2 && !(band.lightning && target.hp < 320.0) {
        band.pref = band.pref.max(115.0);
        if ffa_posture {
            band.pref = band.pref.min(130.0);
        }
    }
    if genome.cdwin_push && threat_holds(target, WeaponKind::Laser) && !target.charging && obs.t - target.last_beam_t <= 1.5 && target.dist < 150.0 {
        band.pref = band.pref.min(35.0);
        band.cdwin = true;
    }
    band
}

// ---------------------------------------------------------------------------
// Kinematics.

/// The movement numbers for this decision.
#[derive(Clone, Copy, Debug)]
struct Kin {
    pos: [f64; 3],
    vel: [f64; 2],
    walk: f64,
    boost: f64,
    accel: f64,
    /// Seconds of boost the energy pays for.
    boost_s: f64,
    radius: f64,
}

#[derive(Clone, Copy, Debug)]
struct Traj {
    dir: [f64; 2],
    boost: bool,
    /// Positions at `k * DT`.
    pts: [[f64; 2]; STEPS + 1],
    /// Velocity at the end.
    end_vel: [f64; 2],
    /// Step at which a wall stopped it.
    blocked_at: Option<usize>,
    min_clear: f64,
}

impl Traj {
    fn end(&self) -> [f64; 2] {
        self.pts[STEPS]
    }
    fn avg_vel(&self) -> [f64; 2] {
        let d = sub(self.end(), self.pts[0]);
        let t = DT * STEPS as f64;
        [d[0] / t, d[1] / t]
    }
    /// Position at time `t` (clamped to the simulated window).
    fn at(&self, t: f64) -> [f64; 2] {
        let f = clamp(t / DT, 0.0, STEPS as f64);
        let k = (f.floor() as usize).min(STEPS - 1);
        let s = f - k as f64;
        let (a, b) = (self.pts[k], self.pts[k + 1]);
        [a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s]
    }
}

/// Simulates holding `dir` (unit or zero) for [`STEPS`] decisions: velocity
/// approaches `dir * speed` by at most `accel * DT` per step, as in the sim.
/// A step onto an unwalkable or out-of-bounds spot stops the mech there.
fn simulate<E: MoveEnv>(env: &E, kin: &Kin, dir: [f64; 2], boost: bool) -> Traj {
    let bounds = env.bounds();
    let moving = dir[0] != 0.0 || dir[1] != 0.0;
    let mut p = xy(kin.pos);
    let mut v = kin.vel;
    let mut traj = Traj {
        dir,
        boost,
        pts: [p; STEPS + 1],
        end_vel: v,
        blocked_at: None,
        min_clear: f64::INFINITY,
    };
    for k in 0..STEPS {
        let t = k as f64 * DT;
        let speed = if boost && moving && t < kin.boost_s { kin.boost } else { kin.walk };
        let want = [dir[0] * speed, dir[1] * speed];
        let dv = sub(want, v);
        let dv_len = dv[0].hypot(dv[1]);
        let max = kin.accel * DT;
        let f = if dv_len > max { max / dv_len } else { 1.0 };
        let nv = [v[0] + dv[0] * f, v[1] + dv[1] * f];
        let np = [p[0] + 0.5 * (v[0] + nv[0]) * DT, p[1] + 0.5 * (v[1] + nv[1]) * DT];
        let moved = (np[0] - p[0]).abs() + (np[1] - p[1]).abs() > 1e-9;
        // From an unwalkable spot (inside the wall margin) only the map edge
        // blocks, so the mech can walk back out.
        let from_ok = env.walkable(p[0], p[1]);
        if moved && (!bounds.contains(np[0], np[1], kin.radius) || (from_ok && !env.walkable(np[0], np[1]))) {
            traj.blocked_at = Some(k);
            for q in &mut traj.pts[k + 1..] {
                *q = p;
            }
            v = [0.0, 0.0];
            break;
        }
        traj.min_clear = traj.min_clear.min(env.clearance(np[0], np[1]));
        p = np;
        v = nv;
        traj.pts[k + 1] = p;
    }
    traj.end_vel = v;
    if traj.min_clear == f64::INFINITY {
        traj.min_clear = env.clearance(p[0], p[1]);
    }
    traj
}

// ---------------------------------------------------------------------------
// Threat predictions, built once per decision.

#[derive(Clone, Copy, Debug, Default)]
struct Bullet {
    p: [f64; 3],
    v: [f64; 3],
}

#[derive(Clone, Copy, Debug, Default)]
struct Shell {
    at: [f64; 2],
    t: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Charger {
    eye: [f64; 3],
    pos: [f64; 2],
    /// Unit beam direction.
    dir: [f64; 2],
}

/// The swept-circle closest approach of a bullet to a trajectory over
/// [`BULLET_HORIZON`]: `(miss, time)`. The last simulated velocity is held
/// past the simulated window. Bullets that pass at the wrong height are
/// skipped by the caller.
fn bullet_miss(b: &Bullet, traj: &Traj) -> (f64, f64) {
    let bv = [b.v[0], b.v[1]];
    let mut best = (f64::INFINITY, 0.0);
    let mut t0 = 0.0;
    let mut seg = |p0: [f64; 2], cv: [f64; 2], t0: f64, len: f64| {
        let bp = [b.p[0] + bv[0] * t0, b.p[1] + bv[1] * t0];
        let r0 = sub(bp, p0);
        let rv = sub(bv, cv);
        let rv2 = dot(rv, rv);
        let s = if rv2 > 1e-12 { clamp(-dot(r0, rv) / rv2, 0.0, len) } else { 0.0 };
        let m = (r0[0] + rv[0] * s).hypot(r0[1] + rv[1] * s);
        if m < best.0 {
            best = (m, t0 + s);
        }
    };
    for k in 0..STEPS {
        let d = sub(traj.pts[k + 1], traj.pts[k]);
        seg(traj.pts[k], [d[0] / DT, d[1] / DT], t0, DT);
        t0 += DT;
    }
    seg(traj.end(), traj.end_vel, t0, BULLET_HORIZON - t0);
    best
}

/// When a mortar shell at `p` moving at `v` falls to height `ground`, within
/// [`MORTAR_HORIZON`]: `(impact xy, seconds)`.
fn mortar_impact(p: [f64; 3], v: [f64; 3], ground: f64) -> Option<([f64; 2], f64)> {
    let disc = v[2] * v[2] + 2.0 * MORTAR_G * (p[2] - ground);
    if disc < 0.0 {
        return None;
    }
    let t = (v[2] + disc.sqrt()) / MORTAR_G;
    (t > 0.0 && t <= MORTAR_HORIZON).then(|| ([p[0] + v[0] * t, p[1] + v[1] * t], t))
}

// ---------------------------------------------------------------------------
// The planner.

/// Strafe state carried between decisions, and the candidate move planner.
#[derive(Clone, Debug)]
pub struct MovePlanner {
    /// +1 strafes counter-clockwise around the target, -1 clockwise.
    strafe_sign: f64,
    /// When the current strafe leg ends.
    leg_end_t: f64,
    /// When `hit_flip` last flipped the strafe.
    last_hit_flip_t: f64,
    /// When the strafe last flipped for any reason.
    last_flip_t: f64,
}

/// Everything the scorer needs, built once per decision.
struct Ctx<'a> {
    obs: &'a Obs,
    genome: &'a ApexGenome,
    mode: Mode,
    goal: Option<NavGoal>,
    target: Option<&'a ThreatInfo>,
    has_los_to_target: bool,
    laser_ready_or_charging: bool,
    band: Option<RangeBand>,
    radius: f64,
    height: f64,
    eye: f64,
    feet: f64,
    walk: f64,
    boost_speed: f64,
    front_half: f64,
    ffa: bool,
    bullets: [Bullet; MAX_BULLETS],
    n_bullets: usize,
    shells: [Shell; MAX_SHELLS],
    n_shells: usize,
    splash_r: f64,
    chargers: [Charger; MAX_CHARGERS],
    n_chargers: usize,
    watchers: [[f64; 3]; MAX_WATCHERS],
    n_watchers: usize,
    arc_threats: [[f64; 2]; MAX_ARC_THREATS],
    n_arc: usize,
    cubes: Points<MAX_CUBES>,
    avoid_cubes: bool,
    /// Reachable loose cube within 35 m (no closer to target than us) to drift toward in Fight.
    fight_cube: Option<[f64; 2]>,
    /// Line-of-sight hints, one per line kind, kept across candidates.
    charger_hints: [u32; MAX_CHARGERS],
    watcher_hints: [u32; MAX_WATCHERS],
    peek_hint: u32,
    /// Our velocity's unit vector, or zero when nearly still.
    cur_vel_unit: [f64; 2],
    /// The strafe has not just flipped, so velocity continuity applies.
    flip_settled: bool,
    /// A mortar shell lands near us.
    mortar_danger: bool,
    /// A LOS edge to the target is within reach and peeking makes sense.
    can_peek: bool,
    /// FFA: nearest threats within [`SEP_RANGE`], at eye height.
    sep_threats: [[f64; 3]; MAX_SEP],
    n_sep: usize,
    sep_hints: [u32; MAX_SEP],
    /// The final shrink: pinned to the next zone center.
    final_hold: bool,
}

impl MovePlanner {
    pub fn new(rng: &mut Rng64) -> Self {
        Self {
            strafe_sign: if rng.chance(0.5) { 1.0 } else { -1.0 },
            leg_end_t: f64::NEG_INFINITY,
            last_hit_flip_t: f64::NEG_INFINITY,
            last_flip_t: f64::NEG_INFINITY,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        &mut self,
        obs: &Obs,
        nav: &ApexNav,
        mem: &BotMemory,
        genome: &ApexGenome,
        mode: Mode,
        goal: Option<NavGoal>,
        target: Option<&ThreatInfo>,
        threats: &[ThreatInfo],
        has_los_to_target: bool,
        laser_ready_or_charging: bool,
        chasing_laser: bool,
        rng: &mut Rng64,
    ) -> MoveDecision {
        self.plan_core(
            obs,
            &NavEnv(nav),
            &MemRecall(mem),
            genome,
            mode,
            goal,
            target,
            threats,
            has_los_to_target,
            laser_ready_or_charging,
            chasing_laser,
            rng,
        )
    }

    /// Advances the strafe leg: a new leg when the old one ends (flipping
    /// with the style's chance), and an immediate flip on a fresh hit.
    fn update_strafe(&mut self, t: f64, genome: &ApexGenome, last_damage_t: f64, rng: &mut Rng64) {
        let (lo, hi, p_flip) = cadence(genome.strafe_style);
        if genome.hit_flip && t - last_damage_t <= 0.15 && t - self.last_hit_flip_t >= 0.5 {
            self.strafe_sign = -self.strafe_sign;
            self.last_hit_flip_t = t;
            self.last_flip_t = t;
            self.leg_end_t = t + rng.range(lo, hi);
            return;
        }
        if t >= self.leg_end_t {
            if self.leg_end_t.is_finite() && rng.chance(p_flip) {
                self.strafe_sign = -self.strafe_sign;
                self.last_flip_t = t;
            }
            self.leg_end_t = t + rng.range(lo, hi);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn plan_core<E: MoveEnv, M: Recall>(
        &mut self,
        obs: &Obs,
        env: &E,
        mem: &M,
        genome: &ApexGenome,
        mode: Mode,
        goal: Option<NavGoal>,
        target: Option<&ThreatInfo>,
        threats: &[ThreatInfo],
        has_los_to_target: bool,
        laser_ready_or_charging: bool,
        chasing_laser: bool,
        rng: &mut Rng64,
    ) -> MoveDecision {
        let me = &obs.me;
        let st = &me.stats;
        let t = obs.t;
        let pos2 = xy(me.pos);
        let last_dmg = mem.last_damage_t();

        // Strafe cadence, hit flip, and the early flip at a wall.
        self.update_strafe(t, genome, last_dmg, rng);
        let target = target.filter(|_| mode == Mode::Fight);
        if let Some(tg) = target {
            let to = norm(sub(xy(tg.pos), pos2));
            let side = [-to[1] * self.strafe_sign, to[0] * self.strafe_sign];
            if side != [0.0, 0.0] && env.step_blocked(me.pos, side) && !env.step_blocked(me.pos, [-side[0], -side[1]]) {
                self.strafe_sign = -self.strafe_sign;
                self.last_flip_t = t;
                let (lo, hi, _) = cadence(genome.strafe_style);
                self.leg_end_t = t + rng.range(lo, hi);
            }
        }

        let ffa = mem.is_ffa();
        let band = target.map(|tg| preferred_range(obs, ffa, genome, tg, threats));
        let speed = me.vel[0].hypot(me.vel[1]);
        let cur_vel_unit = if speed > 1.0 { [me.vel[0] / speed, me.vel[1] / speed] } else { [0.0, 0.0] };
        let mut peek_hint = NO_HINT;
        let can_peek = genome.laser_peek
            && own_holds(obs, WeaponKind::Laser)
            && target.is_some_and(|tg| {
                dist_xy(pos2, xy(tg.pos)) > 65.0
                    && peek_edge_nearby(env, me.pos, st.eye_height, [tg.pos[0], tg.pos[1], tg.pos[2] + 0.5 * st.height], &mut peek_hint)
            });
        let final_hold = obs.zone.next_radius > 0.0 && obs.zone.next_radius <= FINAL_RADIUS;
        let fight_cube = if mode == Mode::Fight && me.cubes < genome.cube_cap {
            target.and_then(|tg| {
                obs.cubes
                    .iter()
                    .filter(|c| (c.pos[2] - me.pos[2]).abs() <= 3.2 && env.walkable(c.pos[0], c.pos[1]))
                    .filter_map(|c| {
                        let d_me = dist_xy(pos2, xy(c.pos));
                        let d_tg = dist_xy(xy(tg.pos), xy(c.pos));
                        (d_me <= 35.0 && d_me <= d_tg).then_some((d_me, xy(c.pos)))
                    })
                    .min_by(|a, b| a.0.total_cmp(&b.0))
                    .map(|(_, p)| p)
            })
        } else {
            None
        };
        let mut ctx = Ctx {
            obs,
            genome,
            mode,
            goal,
            target,
            has_los_to_target,
            laser_ready_or_charging,
            band,
            radius: st.radius,
            height: st.height,
            eye: st.eye_height,
            feet: me.pos[2],
            walk: st.walk_speed.max(0.1),
            boost_speed: st.boost_speed.max(0.1),
            front_half: (st.front_arc_deg.to_radians() * 0.5).clamp(0.1, PI),
            ffa,
            bullets: [Bullet::default(); MAX_BULLETS],
            n_bullets: 0,
            shells: [Shell::default(); MAX_SHELLS],
            n_shells: 0,
            splash_r: MORTAR_SPLASH_R,
            chargers: [Charger::default(); MAX_CHARGERS],
            n_chargers: 0,
            watchers: [[0.0; 3]; MAX_WATCHERS],
            n_watchers: 0,
            arc_threats: [[0.0; 2]; MAX_ARC_THREATS],
            n_arc: 0,
            cubes: Points::new(),
            avoid_cubes: me.cubes >= genome.cube_cap,
            fight_cube,
            charger_hints: [NO_HINT; MAX_CHARGERS],
            watcher_hints: [NO_HINT; MAX_WATCHERS],
            peek_hint,
            cur_vel_unit,
            flip_settled: t - self.last_flip_t > FLIP_SETTLE_S,
            mortar_danger: false,
            can_peek,
            sep_threats: [[0.0; 3]; MAX_SEP],
            n_sep: 0,
            sep_hints: [NO_HINT; MAX_SEP],
            final_hold,
        };
        for w in me.weapons.iter().flatten() {
            if let WeaponObs::Mortar { stats, .. } = w {
                ctx.splash_r = stats.splash_radius;
            }
        }

        // Projectiles: bullets that could reach us, and mortar landings.
        let hit_r = st.radius + BULLET_MARGIN;
        for p in &obs.projectiles {
            match p.kind {
                ProjectileKind::Gun | ProjectileKind::Npc => {
                    let r0 = sub(xy(p.pos), pos2);
                    let v = [p.vel[0], p.vel[1]];
                    let v2 = dot(v, v);
                    if v2 < 1e-6 {
                        continue;
                    }
                    let tc = -dot(r0, v) / v2;
                    if tc <= 0.0 || tc > BULLET_HORIZON {
                        continue;
                    }
                    // Reachable miss: our 0.8 s of travel plus the hit radius.
                    let miss = (r0[0] + v[0] * tc).hypot(r0[1] + v[1] * tc);
                    if miss > hit_r + st.boost_speed * BULLET_HORIZON {
                        continue;
                    }
                    let z = p.pos[2] + p.vel[2] * tc;
                    if z < me.pos[2] - 1.0 || z > me.pos[2] + st.height + 1.0 {
                        continue;
                    }
                    if ctx.n_bullets < MAX_BULLETS {
                        ctx.bullets[ctx.n_bullets] = Bullet { p: p.pos, v: p.vel };
                        ctx.n_bullets += 1;
                    }
                }
                ProjectileKind::Mortar => {
                    if let Some((at, ti)) = mortar_impact(p.pos, p.vel, me.pos[2]) {
                        if dist_xy(at, pos2) < ctx.splash_r + st.radius + MORTAR_MARGIN + st.boost_speed * 1.5 && ctx.n_shells < MAX_SHELLS {
                            ctx.shells[ctx.n_shells] = Shell { at, t: ti };
                            ctx.n_shells += 1;
                        }
                    }
                }
            }
        }
        let mortar_danger = ctx.shells[..ctx.n_shells]
            .iter()
            .any(|s| dist_xy(s.at, pos2) < ctx.splash_r + st.radius + MORTAR_MARGIN);
        ctx.mortar_danger = mortar_danger;

        // Lasers charging at us.
        for th in threats {
            if ctx.n_chargers < MAX_CHARGERS && threat_holds(th, WeaponKind::Laser) && laser_aimed_at(th, pos2) {
                let a = th.aim[0];
                ctx.chargers[ctx.n_chargers] = Charger {
                    eye: [th.pos[0], th.pos[1], th.pos[2] + st.eye_height],
                    pos: xy(th.pos),
                    dir: [a.cos(), a.sin()],
                };
                ctx.n_chargers += 1;
            }
        }

        // Extra watchers (closest first) and threats to keep in the front arc.
        let target_id = target.map(|tg| tg.id);
        let mut others: [(f64, usize); 8] = [(f64::INFINITY, usize::MAX); 8];
        for (i, th) in threats.iter().enumerate() {
            if Some(th.id) == target_id || th.is_ghost || t - th.last_seen_t > 3.0 || th.dist > 180.0 {
                continue;
            }
            // Insertion into the 8 closest.
            let n = others.len();
            let mut j = n;
            while j > 0 && others[j - 1].0 > th.dist {
                j -= 1;
            }
            if j < n {
                others.copy_within(j..n - 1, j + 1);
                others[j] = (th.dist, i);
            }
        }
        for &(d, i) in others.iter().take_while(|o| o.1 != usize::MAX) {
            let th = &threats[i];
            if ctx.n_watchers < MAX_WATCHERS {
                ctx.watchers[ctx.n_watchers] = [th.pos[0], th.pos[1], th.pos[2] + st.eye_height];
                ctx.n_watchers += 1;
            }
            if d < 150.0 && ctx.n_arc < MAX_ARC_THREATS {
                ctx.arc_threats[ctx.n_arc] = xy(th.pos);
                ctx.n_arc += 1;
            }
            if ffa && d < SEP_RANGE && ctx.n_sep < MAX_SEP {
                ctx.sep_threats[ctx.n_sep] = [th.pos[0], th.pos[1], th.pos[2] + st.eye_height];
                ctx.n_sep += 1;
            }
        }

        // Home hold: in reach of home, inside safe zone, nothing incoming, and unseen by anyone near.
        let z = &obs.zone;
        let in_cur_zone = z.radius <= 0.0 || dist_xy(pos2, z.center) <= z.radius - 3.0;
        let in_next_zone = z.next_radius <= 0.0 || obs.t < z.shrink_start - 20.0 || dist_xy(pos2, z.next_center) <= z.next_radius - 3.0;
        if mode == Mode::Home
            && !final_hold
            && ctx.n_bullets == 0
            && !mortar_danger
            && ctx.n_chargers == 0
            && in_cur_zone
            && in_next_zone
            && goal.is_some_and(|g| dist_xy(pos2, xy(g.pos)) <= g.radius)
        {
            let seen = threats.iter().any(|th| !th.is_ghost && th.visible_now && th.los && th.dist < 130.0);
            if !seen {
                return MoveDecision {
                    move_dir: [0.0, 0.0],
                    boost: false,
                    jump: false,
                };
            }
        }

        // Cubes to steer around when full.
        if ctx.avoid_cubes {
            let mut all = Points::<MAX_CUBES>::new();
            for c in &obs.cubes {
                all.push(xy(c.pos));
            }
            mem.cubes(&mut all);
            for &c in all.as_slice() {
                if dist_xy(c, pos2) < 25.0 {
                    ctx.cubes.push(c);
                }
            }
        }

        // Boost and quiet walk.
        let charging = own_laser_charging(obs);
        let have_lightning = own_holds(obs, WeaponKind::Lightning);
        let has_laser = own_holds(obs, WeaponKind::Laser);
        let threat_near = threats.iter().any(|th| th.visible_now && th.los && th.dist < 85.0);
        let dodging = ctx.n_bullets > 0 || mortar_danger || ctx.n_chargers > 0;
        let home_evade_los =
            mode == Mode::Home && ffa && me.energy > 35.0 && threats.iter().any(|th| !th.is_ghost && th.visible_now && th.los && th.dist < 85.0);
        let chasing_laser_now = mode == Mode::Loot
            && (chasing_laser
                || obs
                    .pickups
                    .iter()
                    .any(|p| p.kind == WeaponKind::Laser && goal.is_some_and(|g| dist_xy(xy(p.pos), xy(g.pos)) < 2.0)));
        let energy_ok_20 = if me.max_energy > 0.0 {
            me.energy > 0.20 * me.max_energy
        } else {
            me.energy > 0.20
        };
        let early_uncontested_laser =
            !has_laser && t < 25.0 && energy_ok_20 && chasing_laser_now && !threats.iter().any(|th| !th.is_ghost && th.visible_now && th.los && th.dist < 60.0);
        let reserve = if have_lightning {
            35.0
        } else if matches!(mode, Mode::ZoneMust | Mode::Retreat) {
            12.0
        } else if early_uncontested_laser {
            0.20 * me.max_energy.max(100.0)
        } else {
            22.0
        };
        let farm_orbit_quiet = mode == Mode::Farm && goal.is_some_and(|g| dist_xy(pos2, xy(g.pos)) <= g.radius);
        let quiet = genome.quiet_walk
            && !dodging
            && !home_evade_los
            && (matches!(mode, Mode::Home | Mode::ZoneShould | Mode::Explore) || (mode == Mode::Loot && !early_uncontested_laser) || farm_orbit_quiet)
            && !threat_near
            && t - last_dmg > 3.0;
        let cdwin = band.is_some_and(|b| b.cdwin);
        let closing = band.is_some_and(|b| b.lightning) && target.is_some_and(|tg| tg.dist > 50.0);
        let boost_reason = matches!(mode, Mode::ZoneMust | Mode::Retreat | Mode::Drop)
            || mortar_danger
            || ctx.n_chargers > 0
            || t - last_dmg <= 1.5
            || cdwin
            || closing
            || home_evade_los
            || early_uncontested_laser;
        let boost_ok = !charging && !quiet && me.grounded && me.energy > reserve && boost_reason;

        let charge_mult = if charging { st.charge_move_mult } else { 1.0 };
        let quiet_scale = if quiet { (QUIET_SPEED / ctx.walk).min(1.0) } else { 1.0 };
        let kin = Kin {
            pos: me.pos,
            vel: [me.vel[0], me.vel[1]],
            walk: st.walk_speed * charge_mult * quiet_scale,
            boost: st.boost_speed * charge_mult,
            accel: if me.grounded { st.ground_accel } else { st.air_accel },
            boost_s: if st.boost_drain > 0.0 { me.energy / st.boost_drain } else { f64::INFINITY },
            radius: st.radius,
        };

        // Evaluate every candidate.
        let mut best: Option<(f64, Traj)> = None;
        let mut best_moving: Option<(f64, Traj)> = None;
        // Stop first, so ties (nothing to do) keep the mech still.
        for c in 0..CANDIDATES {
            let (dir, boost) = if c == 0 {
                ([0.0, 0.0], false)
            } else {
                let i = c - 1;
                let h = (i % HEADINGS) as f64 * (2.0 * PI / HEADINGS as f64);
                ([h.cos(), h.sin()], i >= HEADINGS)
            };
            if boost && !boost_ok {
                continue;
            }
            let traj = simulate(env, &kin, dir, boost);
            let mut cost = self.score(&mut ctx, env, &traj);
            if cost.is_nan() {
                cost = f64::INFINITY;
            }
            if best.as_ref().is_none_or(|b| cost < b.0) {
                best = Some((cost, traj));
            }
            if dir != [0.0, 0.0] && best_moving.as_ref().is_none_or(|b| cost < b.0) {
                best_moving = Some((cost, traj));
            }
        }
        let Some((_, mut pick)) = best else {
            return MoveDecision::default();
        };
        if matches!(mode, Mode::Fight | Mode::Farm) && pick.dir == [0.0, 0.0] {
            if let Some((_, m)) = best_moving {
                pick = m;
            }
        }
        MoveDecision {
            move_dir: [pick.dir[0] * quiet_scale, pick.dir[1] * quiet_scale],
            boost: pick.boost,
            jump: false,
        }
    }

    /// Cost of a trajectory; lower is better.
    fn score<E: MoveEnv>(&self, ctx: &mut Ctx, env: &E, traj: &Traj) -> f64 {
        let obs = ctx.obs;
        let g = ctx.genome;
        let start = traj.pts[0];
        let end = traj.end();
        let avg_v = traj.avg_vel();
        let moved = dist_xy(start, end);
        let mut cost = 0.0;

        if traj.boost {
            cost += 3.0;
        }
        if let Some(k) = traj.blocked_at {
            cost += 25.0 + 30.0 * (1.0 - k as f64 / STEPS as f64);
        }

        // (a) Bullets: swept circle in the candidate's frame.
        let hit_r = ctx.radius + BULLET_MARGIN;
        for b in &ctx.bullets[..ctx.n_bullets] {
            let (miss, tm) = bullet_miss(b, traj);
            if miss < hit_r {
                let z = b.p[2] + b.v[2] * tm;
                if z >= ctx.feet - 1.0 && z <= ctx.feet + ctx.height + 1.0 {
                    let urgency = 1.0 - 0.5 * (tm / BULLET_HORIZON);
                    cost += g.dodge_weight * 8.0 * (1.0 - miss / hit_r) * urgency + g.dodge_weight * 2.0;
                }
            }
        }

        // (b) Mortar blasts at the predicted landing.
        let blast_r = ctx.splash_r + ctx.radius + MORTAR_MARGIN;
        for s in &ctx.shells[..ctx.n_shells] {
            let d = dist_xy(traj.at(s.t), s.at);
            if d < blast_r {
                cost += 150.0 + 300.0 * (1.0 - d / blast_r);
            }
        }

        // (c) Lasers charging at us.
        let end_body = [end[0], end[1], ctx.feet + 0.5 * ctx.height];
        for i in 0..ctx.n_chargers {
            let ch = ctx.chargers[i];
            match g.laser_juke_mode {
                LaserJukeMode::None => {}
                LaserJukeMode::StrafeBoost => {
                    let n = [-ch.dir[1], ch.dir[0]];
                    let across = dot(avg_v, n).abs() / ctx.boost_speed;
                    cost -= 60.0 * across;
                    if traj.boost {
                        cost -= 15.0;
                    }
                    let off = dot(sub(end, ch.pos), n).abs();
                    if off < ctx.radius + 1.5 {
                        cost += 30.0;
                    }
                }
                LaserJukeMode::BreakLosCover => {
                    if env.los(ch.eye, end_body, &mut ctx.charger_hints[i]) {
                        cost += 20.0;
                    } else {
                        cost -= 80.0;
                    }
                }
            }
        }

        // (d) Goal and range band.
        match (ctx.mode, ctx.target, ctx.band) {
            (Mode::Fight, Some(tg), Some(band)) => {
                let tp = xy(tg.pos);
                let to = norm(sub(tp, start));
                let tangent = [-to[1] * self.strafe_sign, to[0] * self.strafe_sign];
                let d0 = dist_xy(start, tp);
                let err = d0 - band.pref;
                let mut radial = clamp(err / 15.0, -1.0, 1.0);
                let peek_hold = ctx.can_peek && !ctx.laser_ready_or_charging;
                let hunting = !ctx.has_los_to_target && !peek_hold;
                // No sight line: follow the A* waypoint around the cover rather than into it.
                let detour = ctx.goal.filter(|_| hunting).map(|nav| xy(nav.pos)).and_then(|wp| {
                    let to_wp = norm(sub(wp, start));
                    (to_wp != [0.0, 0.0] && dist_xy(start, wp) > 1.5).then_some(to_wp)
                });
                if let Some(to_wp) = detour {
                    cost -= 28.0 * dot(avg_v, to_wp) / ctx.walk;
                    if traj.blocked_at.is_none() {
                        cost -= 4.0 * dot(avg_v, tangent) / ctx.walk;
                    }
                } else {
                    if hunting {
                        radial = radial.max(0.5);
                    }
                    let desired = norm([
                        tangent[0] * (1.0 - 0.7 * radial.abs()) + to[0] * radial,
                        tangent[1] * (1.0 - 0.7 * radial.abs()) + to[1] * radial,
                    ]);
                    let w = if band.cdwin || (band.lightning && err > 0.0) { 32.0 } else { 25.0 };
                    cost -= w * dot(avg_v, desired) / ctx.walk;
                    cost += 0.5 * (dist_xy(end, tp) - band.pref).abs();
                }
                if moved < 1.5 {
                    cost += 40.0;
                }
                // Slight bias toward a reachable loose cube within 35 m that is no closer to the target.
                if let Some(cube_xy) = ctx.fight_cube {
                    let d0_c = dist_xy(start, cube_xy);
                    let de_c = dist_xy(end, cube_xy);
                    if d0_c > de_c {
                        cost -= (1.2 * (d0_c - de_c)).min(10.0);
                    }
                }
                // Commit to the current leg instead of wobbling between
                // neighboring headings, unless dodging or just flipped.
                if ctx.flip_settled && ctx.n_bullets == 0 && ctx.n_chargers == 0 && !ctx.mortar_danger && traj.dir != [0.0, 0.0] {
                    cost -= 4.0 * dot(traj.dir, ctx.cur_vel_unit);
                }
            }
            (mode, _, _) => {
                // The final shrink overrides wandering toward any other goal.
                if let Some(goal) = ctx.goal.filter(|_| !ctx.final_hold) {
                    let gp = xy(goal.pos);
                    let d0 = dist_xy(start, gp);
                    let de = dist_xy(end, gp);
                    if goal.exact {
                        cost += 2.0 * de;
                    } else if d0 > goal.radius {
                        cost -= 3.0 * (d0 - de.max(goal.radius));
                    } else if mode == Mode::Farm {
                        let to = norm(sub(gp, start));
                        let tangent = [-to[1] * self.strafe_sign, to[0] * self.strafe_sign];
                        cost -= 12.0 * dot(avg_v, tangent) / ctx.walk;
                        cost += 0.5 * (de - goal.radius).abs();
                        if moved < 1.5 {
                            cost += 40.0;
                        }
                    }
                } else if mode == Mode::Fight && moved < 1.5 {
                    cost += 40.0;
                }
                if matches!(mode, Mode::Retreat | Mode::Home) && g.exposure_weight > 0.0 {
                    cost += g.exposure_weight * 0.15 * env.exposure(end[0], end[1]);
                }
                // FFA: keep apart from nearby threats and out of their sight.
                if ctx.ffa && matches!(mode, Mode::Home | Mode::ZoneShould | Mode::Explore) {
                    for i in 0..ctx.n_sep {
                        let th = ctx.sep_threats[i];
                        cost += 0.8 * (140.0 - dist_xy(end, xy(th))).max(0.0);
                        if !env.los(th, end_body, &mut ctx.sep_hints[i]) {
                            cost -= 25.0;
                        }
                    }
                }
            }
        }

        // (e) Laser peek at the LOS boundary, only where an edge is in reach.
        if let (Mode::Fight, Some(tg), true) = (ctx.mode, ctx.target, ctx.can_peek) {
            let eye = [end[0], end[1], ctx.feet + ctx.eye];
            let body = [tg.pos[0], tg.pos[1], tg.pos[2] + 0.5 * ctx.height];
            let los = env.los(eye, body, &mut ctx.peek_hint);
            if ctx.laser_ready_or_charging && ctx.n_chargers == 0 {
                cost += if los { -20.0 } else { 20.0 };
            } else if los {
                cost += 20.0;
            } else {
                cost -= 10.0;
            }
        }

        // (f) Extra watchers and the front arc.
        let watcher_pen = if ctx.ffa { 18.0 } else { 12.0 };
        for i in 0..ctx.n_watchers {
            if env.los(ctx.watchers[i], end_body, &mut ctx.watcher_hints[i]) {
                cost += watcher_pen;
            }
        }
        if ctx.n_arc > 0 {
            let facing = match ctx.target {
                Some(tg) => {
                    let d = sub(xy(tg.pos), end);
                    d[1].atan2(d[0])
                }
                None => obs.me.aim[0],
            };
            for &tp in &ctx.arc_threats[..ctx.n_arc] {
                let d = sub(tp, end);
                let off = wrap_pi(d[1].atan2(d[0]) - facing).abs();
                if off > ctx.front_half {
                    cost += 10.0 * (off - ctx.front_half) / PI + 4.0;
                }
            }
        }

        // (g) Cubes we cannot use.
        if ctx.avoid_cubes {
            let keep = PICKUP_RADIUS + CUBE_MARGIN;
            for &c in ctx.cubes.as_slice() {
                let d = traj.pts[1..].iter().map(|&p| dist_xy(p, c)).fold(f64::INFINITY, f64::min);
                if d < keep {
                    cost += 50.0 + 10.0 * (keep - d);
                }
            }
        }

        // (h) Walls and the zone.
        let settling_home = ctx.mode == Mode::Home && ctx.goal.is_some_and(|goal| dist_xy(end, xy(goal.pos)) <= goal.radius + 2.0);
        if traj.min_clear < LOW_CLEARANCE && !settling_home {
            cost += 6.0 * (LOW_CLEARANCE - traj.min_clear.max(0.0));
        }
        let z = &obs.zone;
        if z.radius > 0.0 {
            let ex = dist_xy(end, z.center) - (z.radius - 3.0);
            if ex > 0.0 {
                cost += 20.0 + 6.0 * ex;
            }
        }
        if z.next_radius > 0.0 && obs.t >= z.shrink_start - 20.0 {
            let ex = dist_xy(end, z.next_center) - (z.next_radius - 3.0);
            if ex > 0.0 {
                let w = if ctx.mode == Mode::Fight { 0.5 } else { 1.0 };
                cost += w * (5.0 + 2.0 * ex);
            }
        }
        // The final shrink leaves no room: hold its exact center.
        if ctx.final_hold {
            let w = if ctx.mode == Mode::Fight { 1.5 } else { 4.0 };
            cost += w * dist_xy(end, z.next_center);
        }
        cost
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::{ActorObs, LaserStats, MechStats, ProjectileObs};

    /// An open square arena with axis-aligned wall boxes `[min_x, min_y, max_x, max_y]`.
    struct Mock {
        half: f64,
        walls: Vec<[f64; 4]>,
    }

    impl Mock {
        fn open() -> Self {
            Self {
                half: 200.0,
                walls: Vec::new(),
            }
        }
        fn rect_dist(r: &[f64; 4], x: f64, y: f64) -> f64 {
            let dx = (r[0] - x).max(x - r[2]).max(0.0);
            let dy = (r[1] - y).max(y - r[3]).max(0.0);
            dx.hypot(dy)
        }
        /// Liang–Barsky: the 2-D segment `a`–`b` crosses the box.
        fn seg_hits(r: &[f64; 4], a: [f64; 3], b: [f64; 3]) -> bool {
            let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
            let (mut t0, mut t1) = (0.0f64, 1.0f64);
            for (p, q) in [(-dx, a[0] - r[0]), (dx, r[2] - a[0]), (-dy, a[1] - r[1]), (dy, r[3] - a[1])] {
                if p.abs() < 1e-12 {
                    if q < 0.0 {
                        return false;
                    }
                } else {
                    let t = q / p;
                    if p < 0.0 {
                        t0 = t0.max(t);
                    } else {
                        t1 = t1.min(t);
                    }
                }
            }
            t0 <= t1
        }
    }

    impl MoveEnv for Mock {
        fn walkable(&self, x: f64, y: f64) -> bool {
            self.walls.iter().all(|r| Self::rect_dist(r, x, y) > 2.5)
        }
        fn clearance(&self, x: f64, y: f64) -> f64 {
            self.walls.iter().map(|r| Self::rect_dist(r, x, y)).fold(50.0, f64::min)
        }
        fn exposure(&self, _: f64, _: f64) -> f64 {
            0.0
        }
        fn step_blocked(&self, pos: [f64; 3], dir: [f64; 2]) -> bool {
            !self.walkable(pos[0] + dir[0] * 3.0, pos[1] + dir[1] * 3.0)
        }
        fn los(&self, a: [f64; 3], b: [f64; 3], _: &mut u32) -> bool {
            !self.walls.iter().any(|r| Self::seg_hits(r, a, b))
        }
        fn bounds(&self) -> Bounds {
            Bounds {
                min_x: -self.half,
                max_x: self.half,
                min_y: -self.half,
                max_y: self.half,
            }
        }
    }

    #[derive(Default)]
    struct Mem {
        last_dmg: f64,
        cubes: Vec<[f64; 2]>,
        ffa: bool,
    }

    impl Recall for Mem {
        fn last_damage_t(&self) -> f64 {
            self.last_dmg
        }
        fn is_ffa(&self) -> bool {
            self.ffa
        }
        fn cubes(&self, out: &mut Points<MAX_CUBES>) {
            for &c in &self.cubes {
                out.push(c);
            }
        }
    }

    fn mem() -> Mem {
        Mem {
            last_dmg: f64::NEG_INFINITY,
            cubes: Vec::new(),
            ffa: false,
        }
    }

    fn obs() -> Obs {
        let mut o = Obs {
            t: 100.0,
            ..Default::default()
        };
        o.me.stats = MechStats::default();
        o.me.hp = 1400.0;
        o.me.max_hp = 1400.0;
        o.me.energy = 100.0;
        o.me.max_energy = 100.0;
        o.me.grounded = true;
        o
    }

    fn threat(id: u32, pos: [f64; 3], weapons: [Option<WeaponKind>; 2]) -> ThreatInfo {
        ThreatInfo {
            id,
            pos,
            vel: [0.0; 3],
            aim: [0.0, 0.0],
            hp: 1400.0,
            max_hp: 1400.0,
            cubes: 0,
            weapons,
            charging: false,
            dist: pos[0].hypot(pos[1]),
            los: true,
            visible_now: true,
            last_seen_t: 100.0,
            last_hit_us_t: f64::NEG_INFINITY,
            last_beam_t: f64::NEG_INFINITY,
            is_ghost: false,
            is_aware: true,
            threat_dps: 100.0,
            target_score: 0.0,
        }
    }

    fn give(o: &mut Obs, slot: usize, kind: WeaponKind) {
        o.me.weapons[slot] = Some(match kind {
            WeaponKind::Laser => WeaponObs::Laser {
                cooldown: 0.0,
                charge: 0.0,
                stats: LaserStats::default(),
            },
            WeaponKind::Lightning => WeaponObs::Lightning {
                cooldown: 0.0,
                stats: Default::default(),
            },
            WeaponKind::Mortar => WeaponObs::Mortar {
                cooldown: 0.0,
                stats: Default::default(),
            },
            WeaponKind::Gun => WeaponObs::Gun {
                cooldown: 0.0,
                stats: Default::default(),
                ammo: 12.0,
                reload: 0.0,
            },
        });
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        p: &mut MovePlanner,
        o: &Obs,
        env: &Mock,
        m: &Mem,
        g: &ApexGenome,
        mode: Mode,
        goal: Option<NavGoal>,
        target: Option<&ThreatInfo>,
        threats: &[ThreatInfo],
    ) -> MoveDecision {
        let mut rng = Rng64::new(7, 0);
        p.plan_core(o, env, m, g, mode, goal, target, threats, true, true, false, &mut rng)
    }

    fn kin(o: &Obs) -> Kin {
        let st = &o.me.stats;
        Kin {
            pos: o.me.pos,
            vel: [o.me.vel[0], o.me.vel[1]],
            walk: st.walk_speed,
            boost: st.boost_speed,
            accel: st.ground_accel,
            boost_s: 10.0,
            radius: st.radius,
        }
    }

    #[test]
    fn strafe_cadence_bands() {
        for (style, lo, hi) in [
            (StrafeStyle::FastJink, 0.35, 0.90),
            (StrafeStyle::Mid12s, 1.0, 2.4),
            (StrafeStyle::Steady35s, 3.0, 5.0),
        ] {
            let g = ApexGenome {
                strafe_style: style,
                hit_flip: false,
                ..ApexGenome::default()
            };
            let mut rng = Rng64::new(1, 1);
            let mut p = MovePlanner::new(&mut rng);
            let mut flips = 0;
            let mut legs = 0;
            let mut t = 0.0;
            for _ in 0..4000 {
                let (sign, end) = (p.strafe_sign, p.leg_end_t);
                p.update_strafe(t, &g, f64::NEG_INFINITY, &mut rng);
                if p.leg_end_t != end {
                    let leg = p.leg_end_t - t;
                    assert!(leg >= lo && leg < hi, "{style:?} leg {leg}");
                    if end.is_finite() {
                        legs += 1;
                        flips += usize::from(p.strafe_sign != sign);
                    }
                }
                t += DT;
            }
            let (_, _, pf) = cadence(style);
            let rate = flips as f64 / legs as f64;
            assert!((rate - pf).abs() < 0.12, "{style:?} flip rate {rate}");
        }
    }

    #[test]
    fn hit_flip_flips_once_per_half_second() {
        let g = ApexGenome {
            hit_flip: true,
            strafe_style: StrafeStyle::Steady35s,
            ..ApexGenome::default()
        };
        let mut rng = Rng64::new(3, 0);
        let mut p = MovePlanner::new(&mut rng);
        p.update_strafe(10.0, &g, f64::NEG_INFINITY, &mut rng);
        let s0 = p.strafe_sign;
        p.update_strafe(10.1, &g, 10.0, &mut rng);
        assert_eq!(p.strafe_sign, -s0);
        assert!(p.leg_end_t >= 10.1 + 3.0);
        // A second hit inside the 0.5 s lockout does not flip back.
        p.update_strafe(10.3, &g, 10.25, &mut rng);
        assert_eq!(p.strafe_sign, -s0);
        // A stale hit (> 0.15 s ago) does not flip.
        p.update_strafe(11.0, &g, 10.25, &mut rng);
        assert_eq!(p.strafe_sign, -s0);
        p.update_strafe(11.0, &g, 10.95, &mut rng);
        assert_eq!(p.strafe_sign, s0);
        let off = ApexGenome { hit_flip: false, ..g };
        p.update_strafe(12.0, &off, 11.95, &mut rng);
        assert_eq!(p.strafe_sign, s0);
    }

    #[test]
    fn range_band_rules() {
        let g = ApexGenome {
            laser_range: 120.0,
            gun_range: 90.0,
            cdwin_push: true,
            ..ApexGenome::default()
        };
        let mut o = obs();
        give(&mut o, 0, WeaponKind::Gun);
        let plain = threat(1, [100.0, 0.0, 0.0], [Some(WeaponKind::Gun), None]);
        assert_eq!(preferred_range(&o, false, &g, &plain, &[]).pref, 90.0);
        let lightning = threat(1, [100.0, 0.0, 0.0], [Some(WeaponKind::Lightning), None]);
        let g_close = ApexGenome { gun_range: 50.0, ..g };
        assert_eq!(preferred_range(&o, false, &g_close, &lightning, &[]).pref, 72.0);
        let mortar = threat(1, [100.0, 0.0, 0.0], [Some(WeaponKind::Mortar), None]);
        assert_eq!(preferred_range(&o, false, &g, &mortar, &[]).pref, 45.0);
        o.me.hp = 300.0;
        assert_eq!(preferred_range(&o, false, &g, &mortar, &[]).pref, 90.0);
        o.me.hp = 1400.0;

        give(&mut o, 1, WeaponKind::Laser);
        assert_eq!(preferred_range(&o, false, &g, &plain, &[]).pref, 120.0);
        let g_short = ApexGenome { laser_range: 60.0, ..g };
        assert_eq!(preferred_range(&o, false, &g_short, &lightning, &[]).pref, 72.0);

        give(&mut o, 1, WeaponKind::Lightning);
        let b = preferred_range(&o, false, &g, &plain, &[]);
        assert!(b.lightning && b.pref == 28.0);
        assert!(!preferred_range(&o, false, &g, &lightning, &[]).lightning);
        let weak = ThreatInfo { hp: 300.0, ..lightning };
        assert_eq!(preferred_range(&o, false, &g, &weak, &[]).pref, 28.0);
        o.me.energy = 20.0;
        assert_eq!(preferred_range(&o, false, &g, &plain, &[]).pref, 90.0);

        // Cooldown window on a laser that just fired.
        let mut laser = threat(1, [100.0, 0.0, 0.0], [Some(WeaponKind::Laser), None]);
        laser.last_beam_t = o.t - 1.0;
        let b = preferred_range(&o, false, &g, &laser, &[]);
        assert!(b.cdwin && b.pref == 35.0);
        laser.charging = true;
        assert!(!preferred_range(&o, false, &g, &laser, &[]).cdwin);
        laser.charging = false;
        laser.last_beam_t = o.t - 2.0;
        assert!(!preferred_range(&o, false, &g, &laser, &[]).cdwin);
        laser.last_beam_t = o.t - 1.0;
        laser.dist = 160.0;
        assert!(!preferred_range(&o, false, &g, &laser, &[]).cdwin);
    }

    #[test]
    fn kinematics_match_the_sim() {
        let o = obs();
        let env = Mock::open();
        let k = kin(&o);
        let t = simulate(&env, &k, [1.0, 0.0], false);
        // 58.5 m/s² for 0.2 s reaches 11.7 m/s, then cruise.
        let expect = 0.5 * 58.5 * 0.2 * 0.2 + 11.7 * 0.4;
        assert!((t.end()[0] - expect).abs() < 1e-9, "{}", t.end()[0]);
        assert!((t.end_vel[0] - 11.7).abs() < 1e-9);
        let b = simulate(&env, &k, [1.0, 0.0], true);
        assert!((b.end_vel[0] - 19.5).abs() < 1e-9);
        let slow = Kin { walk: 11.7 * 0.5, ..k };
        let s = simulate(&env, &slow, [1.0, 0.0], false);
        assert!((s.end_vel[0] - 5.85).abs() < 1e-9);
        // A wall stops it.
        let walled = Mock {
            half: 200.0,
            walls: vec![[5.0, -10.0, 8.0, 10.0]],
        };
        let w = simulate(&walled, &k, [1.0, 0.0], false);
        assert!(w.blocked_at.is_some());
        assert!(w.end()[0] < 2.5);
    }

    #[test]
    fn swept_circle_bullet_miss() {
        let o = obs();
        let env = Mock::open();
        let k = kin(&o);
        let b = Bullet {
            p: [180.0, 0.0, 4.0],
            v: [-252.0, 0.0, 0.0],
        };
        let stop = simulate(&env, &k, [0.0, 0.0], false);
        let side = simulate(&env, &k, [0.0, 1.0], false);
        let (m_stop, t_stop) = bullet_miss(&b, &stop);
        let (m_side, _) = bullet_miss(&b, &side);
        assert!(m_stop < 1e-6 && (t_stop - 180.0 / 252.0).abs() < 1e-6);
        assert!(m_side > o.me.stats.radius + BULLET_MARGIN, "{m_side}");
    }

    #[test]
    fn dodges_incoming_bullet() {
        let mut o = obs();
        o.projectiles.push(ProjectileObs {
            id: 1,
            kind: ProjectileKind::Gun,
            pos: [180.0, 0.0, 4.0],
            vel: [-252.0, 0.0, 0.0],
        });
        let g = ApexGenome {
            dodge_weight: 10.0,
            ..ApexGenome::default()
        };
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Home, None, None, &[]);
        assert!(d.move_dir[1].abs() > 0.5, "{:?}", d.move_dir);
    }

    #[test]
    fn mortar_impact_prediction_and_dodge() {
        let (at, t) = mortar_impact([10.0, 0.0, 11.0], [-5.0, 0.0, 0.0], 0.0).unwrap();
        assert!((t - 1.0).abs() < 1e-9 && (at[0] - 5.0).abs() < 1e-9);
        assert!(mortar_impact([0.0, 0.0, 500.0], [0.0, 0.0, 0.0], 0.0).is_none());

        let mut o = obs();
        o.projectiles.push(ProjectileObs {
            id: 2,
            kind: ProjectileKind::Mortar,
            pos: [0.0, 0.0, 11.0],
            vel: [0.0, 0.0, 0.0],
        });
        let g = ApexGenome::default();
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Home, None, None, &[]);
        assert!(d.boost, "boost to dodge a shell");
        assert!(d.move_dir[0].hypot(d.move_dir[1]) > 0.99);
    }

    #[test]
    fn fight_never_stands_still() {
        let mut o = obs();
        give(&mut o, 0, WeaponKind::Gun);
        let g = ApexGenome {
            gun_range: 100.0,
            ..ApexGenome::default()
        };
        let tg = threat(1, [100.0, 0.0, 0.0], [Some(WeaponKind::Gun), None]);
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Fight, None, Some(&tg), &[tg]);
        assert!(d.move_dir[0].hypot(d.move_dir[1]) > 0.99);
        // At range, mostly strafing across the line to the target.
        assert!(d.move_dir[1].abs() > 0.6, "{:?}", d.move_dir);
        // Boxed in on all sides: still moves.
        let boxed = Mock {
            half: 200.0,
            walls: vec![[-6.0, 3.0, 6.0, 6.0], [-6.0, -6.0, 6.0, -3.0], [3.0, -6.0, 6.0, 6.0], [-6.0, -6.0, -3.0, 6.0]],
        };
        let d = run(&mut p, &o, &boxed, &mem(), &g, Mode::Fight, None, Some(&tg), &[tg]);
        assert!(d.move_dir != [0.0, 0.0]);
    }

    #[test]
    fn closes_to_range_when_far() {
        let mut o = obs();
        give(&mut o, 0, WeaponKind::Gun);
        let g = ApexGenome {
            gun_range: 50.0,
            ..ApexGenome::default()
        };
        let tg = threat(1, [150.0, 0.0, 0.0], [Some(WeaponKind::Gun), None]);
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Fight, None, Some(&tg), &[tg]);
        assert!(d.move_dir[0] > 0.5, "{:?}", d.move_dir);
    }

    #[test]
    fn fight_without_los_follows_the_nav_waypoint() {
        let mut o = obs();
        give(&mut o, 0, WeaponKind::Gun);
        let g = ApexGenome {
            gun_range: 30.0,
            laser_peek: false,
            ..ApexGenome::default()
        };
        // A wall straddles the line to the target; A* routes over its top end.
        let env = Mock {
            half: 200.0,
            walls: vec![[15.0, -25.0, 20.0, 25.0]],
        };
        let tg = threat(1, [80.0, 0.0, 0.0], [Some(WeaponKind::Gun), None]);
        assert!(!env.los([0.0, 0.0, 8.0], [80.0, 0.0, 4.5], &mut NO_HINT.clone()));
        let wp = NavGoal {
            pos: [17.0, 32.0, 0.0],
            radius: 2.0,
            exact: false,
        };
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = p.plan_core(&o, &env, &mem(), &g, Mode::Fight, Some(wp), Some(&tg), &[tg], false, true, false, &mut rng);
        let dir = norm(d.move_dir);
        let to_wp = norm([17.0, 32.0]);
        assert!(dot(dir, to_wp) > 0.8, "{:?}", d.move_dir);
        // Without the waypoint it would head at the wall instead.
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = p.plan_core(&o, &env, &mem(), &g, Mode::Fight, None, Some(&tg), &[tg], false, true, false, &mut rng);
        assert!(dot(norm(d.move_dir), to_wp) < 0.8, "{:?}", d.move_dir);
    }

    #[test]
    fn steers_around_cubes_at_cap() {
        let mut o = obs();
        o.me.cubes = 20;
        let g = ApexGenome {
            quiet_walk: false,
            ..ApexGenome::default()
        };
        let goal = NavGoal {
            pos: [40.0, 0.0, 0.0],
            radius: 2.0,
            exact: false,
        };
        let m = Mem {
            last_dmg: f64::NEG_INFINITY,
            cubes: vec![[6.0, 0.0]],
            ffa: false,
        };
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &Mock::open(), &m, &g, Mode::Explore, Some(goal), None, &[]);
        let traj = simulate(&Mock::open(), &kin(&o), norm(d.move_dir), false);
        let closest = traj.pts.iter().map(|&q| dist_xy(q, [6.0, 0.0])).fold(f64::INFINITY, f64::min);
        assert!(closest >= PICKUP_RADIUS + CUBE_MARGIN, "{closest} {:?}", d.move_dir);
        assert!(d.move_dir[0] > 0.0);
        // Under the cap it walks straight through.
        o.me.cubes = 0;
        let d = run(&mut p, &o, &Mock::open(), &m, &g, Mode::Explore, Some(goal), None, &[]);
        assert!(d.move_dir[0] > 0.99, "{:?}", d.move_dir);
    }

    fn charger_at_us() -> ThreatInfo {
        let mut th = threat(1, [60.0, 0.0, 0.0], [Some(WeaponKind::Laser), None]);
        th.charging = true;
        th.aim = [PI, 0.0];
        th
    }

    #[test]
    fn break_los_cover_hides() {
        let mut o = obs();
        give(&mut o, 0, WeaponKind::Gun);
        let g = ApexGenome {
            laser_juke_mode: LaserJukeMode::BreakLosCover,
            laser_peek: false,
            ..ApexGenome::default()
        };
        let env = Mock {
            half: 200.0,
            walls: vec![[5.0, 3.0, 8.0, 12.0]],
        };
        let th = charger_at_us();
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &env, &mem(), &g, Mode::Fight, None, Some(&th), &[th]);
        let mut k = kin(&o);
        k.boost_s = 10.0;
        let traj = simulate(&env, &k, norm(d.move_dir), d.boost);
        let e = traj.end();
        assert!(!env.los([60.0, 0.0, 8.0], [e[0], e[1], 4.5], &mut NO_HINT.clone()), "{:?} -> {e:?}", d.move_dir);
    }

    #[test]
    fn strafe_boost_crosses_the_beam() {
        let mut o = obs();
        give(&mut o, 0, WeaponKind::Gun);
        let g = ApexGenome {
            laser_juke_mode: LaserJukeMode::StrafeBoost,
            ..ApexGenome::default()
        };
        let th = charger_at_us();
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Fight, None, Some(&th), &[th]);
        assert!(d.boost);
        assert!(d.move_dir[1].abs() > 0.9, "{:?}", d.move_dir);
        // No boost while our own laser charges.
        o.me.weapons[0] = Some(WeaponObs::Laser {
            cooldown: 0.0,
            charge: 0.3,
            stats: LaserStats::default(),
        });
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Fight, None, Some(&th), &[th]);
        assert!(!d.boost);
    }

    #[test]
    fn boost_gating() {
        let g = ApexGenome::default();
        let goal = NavGoal {
            pos: [100.0, 0.0, 0.0],
            radius: 2.0,
            exact: false,
        };
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let mut o = obs();
        // ZoneMust boosts when energy is above the 12 reserve.
        o.me.energy = 15.0;
        assert!(run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::ZoneMust, Some(goal), None, &[]).boost);
        // Loot without a reason does not.
        o.me.energy = 100.0;
        assert!(!run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Loot, Some(goal), None, &[]).boost);
        // Airborne never boosts (that would be the jet).
        o.me.grounded = false;
        assert!(!run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::ZoneMust, Some(goal), None, &[]).boost);
        o.me.grounded = true;
        // With lightning the reserve is 35.
        give(&mut o, 0, WeaponKind::Lightning);
        o.me.energy = 30.0;
        assert!(!run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::ZoneMust, Some(goal), None, &[]).boost);
    }

    #[test]
    fn quiet_walk_scales_speed() {
        let o = obs();
        let goal = NavGoal {
            pos: [100.0, 0.0, 0.0],
            radius: 2.0,
            exact: false,
        };
        let g = ApexGenome {
            quiet_walk: true,
            ..ApexGenome::default()
        };
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Explore, Some(goal), None, &[]);
        let mag = d.move_dir[0].hypot(d.move_dir[1]);
        assert!((mag - 4.6 / 11.7).abs() < 1e-9, "{mag}");
        // Not with a threat near, nor in other modes.
        let near = threat(1, [60.0, 60.0, 0.0], [None, None]);
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Explore, Some(goal), None, &[near]);
        assert!((d.move_dir[0].hypot(d.move_dir[1]) - 1.0).abs() < 1e-9);
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Farm, Some(goal), None, &[]);
        assert!((d.move_dir[0].hypot(d.move_dir[1]) - 1.0).abs() < 1e-9);
        // Home and ZoneShould stay quiet.
        for mode in [Mode::Home, Mode::ZoneShould] {
            let d = run(&mut p, &o, &Mock::open(), &mem(), &g, mode, Some(goal), None, &[]);
            assert!((d.move_dir[0].hypot(d.move_dir[1]) - 4.6 / 11.7).abs() < 1e-9, "{mode:?}");
        }
        // Early Explore (before 90 s) also walks quietly.
        let mut early = o.clone();
        early.t = 60.0;
        let d = run(&mut p, &early, &Mock::open(), &mem(), &g, Mode::Explore, Some(goal), None, &[]);
        assert!((d.move_dir[0].hypot(d.move_dir[1]) - 4.6 / 11.7).abs() < 1e-9, "{:?}", d.move_dir);
    }

    #[test]
    fn quiet_walk_never_throttles_early_unarmed_laser_loot() {
        let goal = NavGoal {
            pos: [100.0, 0.0, 0.0],
            radius: 2.0,
            exact: false,
        };
        let g = ApexGenome {
            quiet_walk: true,
            ..ApexGenome::default()
        };
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        // Without a laser, chasing a Laser at t < 25.0 with energy > 20% and no close LOS threat runs fast / boosts.
        let mut early_laser = obs();
        early_laser.t = 5.0;
        early_laser.pickups.push(proto::PickupObs {
            id: 10,
            kind: WeaponKind::Laser,
            pos: [100.0, 0.0, 0.0],
        });
        let d_fast = run(&mut p, &early_laser, &Mock::open(), &mem(), &g, Mode::Loot, Some(goal), None, &[]);
        assert!(d_fast.move_dir[0].hypot(d_fast.move_dir[1]) > 0.9, "unarmed early laser {:?}", d_fast.move_dir);
        assert!(d_fast.boost, "early uncontested laser allows boost");
        let mut t20_laser = early_laser.clone();
        t20_laser.t = 20.0;
        let d_t20 = run(&mut p, &t20_laser, &Mock::open(), &mem(), &g, Mode::Loot, Some(goal), None, &[]);
        assert!(d_t20.move_dir[0].hypot(d_t20.move_dir[1]) > 0.9, "t=20s early laser {:?}", d_t20.move_dir);
        assert!(d_t20.boost);
        // Contested within 60 m or past t = 25.0 stays quiet without boost.
        let close_seer = threat(1, [50.0, 0.0, 0.0], [None, None]);
        let d_contested = run(&mut p, &early_laser, &Mock::open(), &mem(), &g, Mode::Loot, Some(goal), None, &[close_seer]);
        assert!(!d_contested.boost, "contested <60m does not boost");
        // Otherwise Mode::Loot uses quiet walk (cube, late laser, or armed).
        for t in [5.0, 30.0] {
            let mut o = obs();
            o.t = t;
            let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Loot, Some(goal), None, &[]);
            assert!((d.move_dir[0].hypot(d.move_dir[1]) - 4.6 / 11.7).abs() < 1e-9, "cube t={t} {:?}", d.move_dir);
        }
        let mut late_laser = early_laser;
        late_laser.t = 26.0;
        let d_late = run(&mut p, &late_laser, &Mock::open(), &mem(), &g, Mode::Loot, Some(goal), None, &[]);
        assert!(
            (d_late.move_dir[0].hypot(d_late.move_dir[1]) - 4.6 / 11.7).abs() < 1e-9,
            "late laser {:?}",
            d_late.move_dir
        );
    }

    #[test]
    fn farm_orbits_inside_standoff_never_stands_still() {
        let o = obs();
        let goal = NavGoal {
            pos: [20.0, 0.0, 0.0],
            radius: 26.0,
            exact: false,
        };
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let g_quiet = ApexGenome {
            quiet_walk: true,
            ..ApexGenome::default()
        };
        let d_q = run(&mut p, &o, &Mock::open(), &mem(), &g_quiet, Mode::Farm, Some(goal), None, &[]);
        let mag_q = d_q.move_dir[0].hypot(d_q.move_dir[1]);
        assert!((mag_q - 4.6 / 11.7).abs() < 1e-9 && d_q.move_dir[1].abs() > 0.1, "{:?}", d_q.move_dir);

        let g_loud = ApexGenome {
            quiet_walk: false,
            ..ApexGenome::default()
        };
        let d_l = run(&mut p, &o, &Mock::open(), &mem(), &g_loud, Mode::Farm, Some(goal), None, &[]);
        let mag_l = d_l.move_dir[0].hypot(d_l.move_dir[1]);
        assert!(mag_l > 0.9 && d_l.move_dir[1].abs() > 0.5, "{:?}", d_l.move_dir);
    }

    #[test]
    fn idle_stays_still() {
        let o = obs();
        let g = ApexGenome::default();
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Home, None, None, &[]);
        assert_eq!(d.move_dir, [0.0, 0.0]);
        let goal = NavGoal {
            pos: [1.0, 0.0, 0.0],
            radius: 3.0,
            exact: false,
        };
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Loot, Some(goal), None, &[]);
        assert_eq!(d.move_dir, [0.0, 0.0]);
    }

    #[test]
    fn walks_out_of_the_wall_margin() {
        let o = obs();
        let g = ApexGenome {
            quiet_walk: false,
            ..ApexGenome::default()
        };
        // The wall's 2.5 m margin covers the start.
        let env = Mock {
            half: 200.0,
            walls: vec![[1.0, -10.0, 4.0, 10.0]],
        };
        assert!(!env.walkable(0.0, 0.0));
        let goal = NavGoal {
            pos: [-40.0, 0.0, 0.0],
            radius: 2.0,
            exact: false,
        };
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &env, &mem(), &g, Mode::ZoneMust, Some(goal), None, &[]);
        assert!(d.move_dir[0] < -0.9, "{:?}", d.move_dir);
    }

    #[test]
    fn heads_into_the_zone() {
        let mut o = obs();
        o.zone.center = [100.0, 0.0];
        o.zone.radius = 50.0;
        o.zone.next_center = [100.0, 0.0];
        o.zone.next_radius = 30.0;
        let g = ApexGenome::default();
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Home, None, None, &[]);
        assert!(norm(d.move_dir)[0] > 0.9, "{:?}", d.move_dir);
    }

    #[test]
    fn deterministic_and_fast() {
        let mut o = obs();
        give(&mut o, 0, WeaponKind::Gun);
        give(&mut o, 1, WeaponKind::Laser);
        for i in 0..10 {
            o.projectiles.push(ProjectileObs {
                id: i,
                kind: if i % 3 == 0 { ProjectileKind::Mortar } else { ProjectileKind::Gun },
                pos: [100.0 + i as f64, 10.0 * i as f64, 6.0],
                vel: [-200.0, 0.0, 2.0],
            });
        }
        o.mechs.push(ActorObs {
            id: 1,
            kind: proto::ActorKind::Mech,
            pos: [80.0, 0.0, 0.0],
            vel: [0.0; 3],
            aim: [PI, 0.0],
            hp: None,
            max_hp: None,
            cubes: None,
            weapons: None,
            charging: Some(true),
        });
        let g = ApexGenome {
            laser_peek: true,
            laser_juke_mode: LaserJukeMode::BreakLosCover,
            hit_flip: true,
            ..ApexGenome::default()
        };
        let env = Mock {
            half: 200.0,
            walls: vec![[10.0, 5.0, 14.0, 20.0], [-20.0, -14.0, -5.0, -10.0]],
        };
        let mut tg = charger_at_us();
        tg.pos = [80.0, 0.0, 0.0];
        let others = [tg, threat(2, [0.0, 90.0, 0.0], [None, None]), threat(3, [-50.0, -50.0, 0.0], [None, None])];
        let m = Mem {
            last_dmg: 99.95,
            cubes: vec![[3.0, 3.0]],
            ffa: false,
        };
        let mut r1 = Rng64::new(5, 2);
        let mut r2 = Rng64::new(5, 2);
        let mut p1 = MovePlanner::new(&mut r1);
        let mut p2 = MovePlanner::new(&mut r2);
        let n = 200;
        let start = std::time::Instant::now();
        for i in 0..n {
            o.t = 100.0 + i as f64 * DT;
            let a = p1.plan_core(&o, &env, &m, &g, Mode::Fight, None, Some(&tg), &others, true, true, false, &mut r1);
            let b = p2.plan_core(&o, &env, &m, &g, Mode::Fight, None, Some(&tg), &others, true, true, false, &mut r2);
            assert_eq!(a, b);
        }
        let per = start.elapsed().as_secs_f64() / (2 * n) as f64;
        // Loose bound so a busy machine does not flake; release measures well under 0.4 ms.
        let limit = if cfg!(debug_assertions) { 2e-2 } else { 2e-3 };
        assert!(per < limit, "{:.3} ms per plan", per * 1e3);
    }

    #[test]
    fn open_field_fight_ignores_laser_peek() {
        let mut o = obs();
        give(&mut o, 0, WeaponKind::Laser);
        let tg = threat(1, [100.0, 0.0, 0.0], [Some(WeaponKind::Gun), None]);
        let body = [100.0, 0.0, 0.5 * o.me.stats.height];
        // Nothing to peek from in the open; a wall edge in reach is.
        assert!(!peek_edge_nearby(&Mock::open(), o.me.pos, o.me.stats.eye_height, body, &mut NO_HINT.clone()));
        let walled = Mock {
            half: 200.0,
            walls: vec![[8.0, -1.0, 10.0, 30.0]],
        };
        assert!(peek_edge_nearby(&walled, o.me.pos, o.me.stats.eye_height, body, &mut NO_HINT.clone()));
        // Laser cooling down, target in sight, open field: same move as without
        // laser_peek, and a full-speed one.
        for los in [true, false] {
            let plan = |peek: bool| {
                let g = ApexGenome {
                    laser_peek: peek,
                    laser_range: 100.0,
                    ..ApexGenome::default()
                };
                let mut rng = Rng64::new(1, 0);
                let mut p = MovePlanner::new(&mut rng);
                p.plan_core(&o, &Mock::open(), &mem(), &g, Mode::Fight, None, Some(&tg), &[tg], los, false, false, &mut rng)
            };
            let (with, without) = (plan(true), plan(false));
            assert_eq!(with, without, "los={los}");
            assert!(with.move_dir[0].hypot(with.move_dir[1]) > 0.99, "{:?}", with.move_dir);
        }
    }

    #[test]
    fn practice_finish_close_and_crowd_standoff() {
        let g = ApexGenome {
            gun_range: 50.0,
            laser_range: 120.0,
            cdwin_push: false,
            ..ApexGenome::default()
        };
        let mut o = obs();
        give(&mut o, 0, WeaponKind::Gun);
        let weak = ThreatInfo {
            hp: 400.0,
            ..threat(1, [80.0, 0.0, 0.0], [Some(WeaponKind::Gun), None])
        };
        assert_eq!(preferred_range(&o, false, &g, &weak, &[weak]).pref, 22.0);
        // Not in FFA, not when we are hurt, not when far, not with a watcher near.
        assert_eq!(preferred_range(&o, true, &g, &weak, &[weak]).pref, 50.0);
        o.me.hp = 500.0;
        assert_eq!(preferred_range(&o, false, &g, &weak, &[weak]).pref, 50.0);
        o.me.hp = 1400.0;
        let far = ThreatInfo { dist: 95.0, ..weak };
        assert_eq!(preferred_range(&o, false, &g, &far, &[far]).pref, 50.0);
        let watcher = threat(2, [0.0, 120.0, 0.0], [None, None]);
        assert_eq!(preferred_range(&o, false, &g, &weak, &[weak, watcher]).pref, 50.0);
        // A ghost or stale threat is no watcher.
        let ghost = ThreatInfo { is_ghost: true, ..watcher };
        let stale = ThreatInfo {
            last_seen_t: o.t - 5.0,
            ..watcher
        };
        assert_eq!(preferred_range(&o, false, &g, &weak, &[weak, ghost, stale]).pref, 22.0);

        // Gun against a laser in Practice keeps out of the open-range duel; in FFA shifts to 95..130 m standoff.
        let laser = threat(1, [80.0, 0.0, 0.0], [Some(WeaponKind::Laser), None]);
        assert_eq!(preferred_range(&o, false, &g, &laser, &[laser]).pref, 95.0);
        assert_eq!(preferred_range(&o, true, &g, &laser, &[laser]).pref, 95.0);

        // Threats within 150-160 m in FFA: stand off in 95..130 m.
        let plain = threat(1, [80.0, 0.0, 0.0], [Some(WeaponKind::Gun), None]);
        let crowd = [plain, threat(2, [0.0, 120.0, 0.0], [None, None]), threat(3, [-130.0, 0.0, 0.0], [None, None])];
        assert_eq!(preferred_range(&o, true, &g, &plain, &crowd).pref, 115.0);
        assert_eq!(preferred_range(&o, true, &g, &plain, &crowd[..2]).pref, 95.0);
        give(&mut o, 1, WeaponKind::Laser);
        assert_eq!(preferred_range(&o, true, &g, &plain, &crowd).pref, 120.0);
        // Lightning on a nearly dead target keeps brawling through the crowd.
        give(&mut o, 1, WeaponKind::Lightning);
        let dying = ThreatInfo { hp: 300.0, ..plain };
        let crowd = [dying, crowd[1], crowd[2]];
        let b = preferred_range(&o, true, &g, &dying, &crowd);
        assert!(b.lightning && b.pref == 28.0, "{b:?}");
        assert_eq!(preferred_range(&o, true, &g, &plain, &[plain, crowd[1], crowd[2]]).pref, 115.0);
    }

    #[test]
    fn ffa_keeps_apart_and_home_holds_still() {
        let g = ApexGenome {
            quiet_walk: false,
            ..ApexGenome::default()
        };
        let ffa = Mem { ffa: true, ..mem() };
        let near = threat(2, [40.0, 0.0, 0.0], [Some(WeaponKind::Gun), None]);
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        // FFA: back away from a close threat; outside FFA stay put.
        let d = run(&mut p, &obs(), &Mock::open(), &ffa, &g, Mode::Explore, None, None, &[near]);
        assert!(d.move_dir[0] < -0.5, "{:?}", d.move_dir);
        let d = run(&mut p, &obs(), &Mock::open(), &mem(), &g, Mode::Explore, None, None, &[near]);
        assert_eq!(d.move_dir, [0.0, 0.0]);
        // FFA: break its line of sight behind a wall when one is in reach.
        let cover = Mock {
            half: 200.0,
            walls: vec![[6.0, 4.0, 9.0, 30.0]],
        };
        let d = run(&mut p, &obs(), &cover, &ffa, &g, Mode::Explore, None, None, &[near]);
        let traj = simulate(&cover, &kin(&obs()), norm(d.move_dir), false);
        let e = traj.end();
        assert!(
            !cover.los([40.0, 0.0, 8.0], [e[0], e[1], 4.5], &mut NO_HINT.clone()),
            "{:?} -> {e:?}",
            d.move_dir
        );

        // Home in reach, beside a wall: the hold bonus outweighs the small
        // clearance gain of edging away; Explore edges away.
        let wall = Mock {
            half: 200.0,
            walls: vec![[3.0, -10.0, 5.0, 10.0]],
        };
        let home = NavGoal {
            pos: [0.0, 0.0, 0.0],
            radius: 8.0,
            exact: false,
        };
        let d = run(&mut p, &obs(), &wall, &mem(), &g, Mode::Explore, Some(home), None, &[]);
        assert!(d.move_dir != [0.0, 0.0]);
        let d = run(&mut p, &obs(), &wall, &mem(), &g, Mode::Home, Some(home), None, &[]);
        assert_eq!(d.move_dir, [0.0, 0.0]);
        // A visible threat with LOS within 130 m cancels the hold; a stale track < 130 m still freezes [0, 0].
        let mut stale = threat(2, [-100.0, 0.0, 0.0], [None, None]);
        stale.visible_now = false;
        stale.last_seen_t = obs().t - 10.0;
        let d = run(&mut p, &obs(), &wall, &ffa, &g, Mode::Home, Some(home), None, &[stale]);
        assert_eq!(d.move_dir, [0.0, 0.0], "home_hold freezes even with stale tracks < 130 m");
        assert!(!d.boost && !d.jump);
        let watcher = threat(2, [-100.0, 0.0, 0.0], [None, None]);
        let d = run(&mut p, &obs(), &wall, &ffa, &g, Mode::Home, Some(home), None, &[watcher]);
        assert!(d.move_dir != [0.0, 0.0]);
    }

    #[test]
    fn final_shrink_holds_the_center() {
        let g = ApexGenome {
            quiet_walk: false,
            ..ApexGenome::default()
        };
        let mut o = obs();
        o.me.pos = [15.0, 0.0, 0.0];
        o.zone.next_center = [20.0, 0.0];
        // Home wanders off toward a goal behind us.
        let goal = NavGoal {
            pos: [-30.0, 0.0, 0.0],
            radius: 2.0,
            exact: false,
        };
        let mut rng = Rng64::new(1, 0);
        let mut p = MovePlanner::new(&mut rng);
        o.zone.next_radius = 13.0;
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Home, Some(goal), None, &[]);
        assert!(d.move_dir[0] < -0.5, "{:?}", d.move_dir);
        // The final shrink pins it to the center instead.
        o.zone.next_radius = 10.0;
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Home, Some(goal), None, &[]);
        assert!(d.move_dir[0] > 0.5, "{:?}", d.move_dir);
        // On the center it stays.
        o.me.pos = [20.0, 0.0, 0.0];
        let d = run(&mut p, &o, &Mock::open(), &mem(), &g, Mode::Home, Some(goal), None, &[]);
        assert_eq!(d.move_dir, [0.0, 0.0]);
    }
}
