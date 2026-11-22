//! Port of the Python submission, `mech-bot/bot.py` ("cbot"). Every strategy of
//! it is a policy here: `aggro` (the live one), `survive`, `hide`, and `roof`.
//!
//! Three layers per decision, as in the original:
//! - Memory: threats, alerts from sounds, the last enemy sighting, hold spot, roof target.
//! - Movement: a state machine; each strategy is an ordered priority list of states.
//! - Gunnery: torso aim and triggers, independent of the legs, gated by the strategy's fire mode.
//!
//! plus reflexes (shell and laser dodge, unstick), pickups, and upgrades.
//!
//! Parity notes. The map model is `nav.rs`, a port of `world.py`. The strafe
//! timing draws from CPython's Mersenne Twister seeded with `hello.seed`, as
//! the original does. Python sizes each flow-field slice by wall clock; here
//! it is `flow_pops` queue pops per decision, so a seed reproduces a match.
//! The default (65536) searches far more than Python manages in its 10 ms
//! (about 2000 pops on one CPU), so fields for new goals are ready at once
//! where Python steers by sampled clear headings for seconds; the parity data
//! is captured at the same budget.
//! Python rounds the action's numbers (move to 1e-3, aim to 1e-4); this does not.
//!
//! Beyond parity, each behind a `Params` switch that is off by default (tests in
//! `tests/switches.rs`): recover and its line-of-sight variant, a track table of seen
//! mechs, mode detection (unexplained deaths, squad signatures A and B from unprovoked
//! attackers), the free-for-all early caution and final, concealed hold spots, exposure break, silent walk,
//! the centre brake, gun fire gates, loadout fight range, and target-score weights.
//! Not built: STRATEGY.md 2.1's beam and HP-loss rows and the weak mutual-ignore row,
//! practice detection, `fight_ratio`, `crate_farm_range`, `dodge_weight`, `reload_cover`.

use crate::nav::{Flow, MechDims, NavMap, NBR, NONE_TOP};
use crate::pyrand::PyRandom;
use crate::{Params, ParamsF64, Policy};
use proto::{
    Action, ActorKind, ActorObs, Config, DropObs, GameEvent, Hello, Obs, ProjectileKind, SelfObs,
    SoundDist, SoundKind, WeaponKind, WeaponObs,
};
use std::f64::consts::PI;
use std::sync::Arc;

/// Python's weapon preference, best first.
const WEAPON_PREF: [WeaponKind; 3] = [WeaponKind::Lightning, WeaponKind::Laser, WeaponKind::Mortar];
/// Python's upgrade preference, best first.
const UPGRADE_PREF: [&str; 22] = [
    "plating",
    "armor",
    "nanites",
    "gyros",
    "gun_rate",
    "gun_damage",
    "lightning_efficiency",
    "laser_damage",
    "laser_charge",
    "mortar_reload",
    "lightning_range",
    "lightning_chain",
    "sensors",
    "servos",
    "gun_velocity",
    "gun_mag",
    "reactor",
    "capacitor",
    "jets",
    "mortar_splash",
    "mortar_velocity",
    "laser_pierce",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum St {
    Escape,
    Dodge,
    Retreat,
    Break,
    Fight,
    Rotate,
    Drop,
    Loot,
    Farm,
    Hunt,
    Climb,
    Perch,
    Hold,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FireMode {
    Any,
    Engaged,
    Attacked,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    Survive,
    Hide,
    Roof,
    Aggro,
}

impl Strategy {
    pub fn parse(name: &str) -> Option<Strategy> {
        match name {
            "survive" => Some(Strategy::Survive),
            "hide" => Some(Strategy::Hide),
            "roof" => Some(Strategy::Roof),
            "aggro" => Some(Strategy::Aggro),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Strategy::Survive => "survive",
            Strategy::Hide => "hide",
            Strategy::Roof => "roof",
            Strategy::Aggro => "aggro",
        }
    }

    fn states(self) -> &'static [St] {
        use St::*;
        match self {
            Strategy::Survive => &[
                Escape, Dodge, Retreat, Break, Fight, Rotate, Drop, Loot, Farm, Hold,
            ],
            Strategy::Hide => &[Escape, Dodge, Rotate, Hold],
            Strategy::Roof => &[Escape, Dodge, Climb, Perch, Rotate, Hold],
            Strategy::Aggro => &[Escape, Dodge, Break, Fight, Loot, Farm, Hunt, Rotate, Hold],
        }
    }

    fn fire(self) -> FireMode {
        match self {
            Strategy::Survive => FireMode::Engaged,
            Strategy::Hide | Strategy::Roof => FireMode::Attacked,
            Strategy::Aggro => FireMode::Any,
        }
    }

    fn corner_spots(self) -> bool {
        self == Strategy::Hide
    }
}

pub fn wrap(a: f64) -> f64 {
    (a + PI).rem_euclid(2.0 * PI) - PI
}

fn unit(dx: f64, dy: f64) -> (f64, f64) {
    let l = dx.hypot(dy);
    if l > 1e-9 {
        (dx / l, dy / l)
    } else {
        (0.0, 0.0)
    }
}

fn finite(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn heading_or(mv: (f64, f64), fallback: f64) -> f64 {
    if mv != (0.0, 0.0) {
        mv.1.atan2(mv.0)
    } else {
        fallback
    }
}

/// Python's `round(v)`: nearest integer, ties to even.
fn pround(v: f64) -> i64 {
    v.round_ties_even() as i64
}

/// Per-decision facts derived from one observation.
struct Ctx<'a> {
    obs: &'a Obs,
    nav: &'a NavMap,
    me: &'a SelfObs,
    t: f64,
    x: f64,
    y: f64,
    z: f64,
    vx: f64,
    vy: f64,
    hp_frac: f64,
    muzzle: [f64; 3],
    /// The next circle shrunk by the margin: `(cx, cy, r)`.
    region: (f64, f64, f64),
    enemies: Vec<(f64, &'a ActorObs)>,
    npcs: Vec<(f64, &'a ActorObs)>,
    ground: f64,
}

impl<'a> Ctx<'a> {
    fn new(obs: &'a Obs, nav: &'a NavMap, p: &ParamsF64) -> Ctx<'a> {
        let me = &obs.me;
        let [x, y, z] = me.pos;
        let zone = &obs.zone;
        let margin = p.zone_margin.min(p.zone_margin_frac * zone.next_radius);
        let mut enemies = Vec::new();
        let mut npcs = Vec::new();
        for m in &obs.mechs {
            let d = (m.pos[0] - x).hypot(m.pos[1] - y);
            if m.kind == ActorKind::Mech {
                enemies.push((d, m))
            } else {
                npcs.push((d, m))
            }
        }
        enemies.sort_by(|a, b| a.0.total_cmp(&b.0));
        npcs.sort_by(|a, b| a.0.total_cmp(&b.0));
        Ctx {
            obs,
            nav,
            me,
            t: obs.t,
            x,
            y,
            z,
            vx: me.vel[0],
            vy: me.vel[1],
            hp_frac: me.hp / me.max_hp.max(1.0),
            muzzle: [x, y, z + me.stats.muzzle_height],
            region: (
                zone.next_center[0],
                zone.next_center[1],
                (zone.next_radius - margin).max(0.0),
            ),
            enemies,
            npcs,
            ground: nav.ground(x, y),
        }
    }

    fn dist_region_at(&self, x: f64, y: f64) -> f64 {
        let (cx, cy, r) = self.region;
        (x - cx).hypot(y - cy) - r
    }

    fn dist_region(&self) -> f64 {
        self.dist_region_at(self.x, self.y)
    }

    fn all_actors(&self) -> impl Iterator<Item = &(f64, &'a ActorObs)> {
        self.enemies.iter().chain(self.npcs.iter())
    }

    fn region_penalty(&self, dx: f64, dy: f64) -> f64 {
        let ahead = 12.0;
        let (cx, cy, r) = self.region;
        let out = (self.x + dx * ahead - cx).hypot(self.y + dy * ahead - cy) - r.max(10.0);
        let zone = &self.obs.zone;
        let out_now = (self.x + dx * ahead - zone.center[0])
            .hypot(self.y + dy * ahead - zone.center[1])
            - zone.radius;
        0.03 * out.max(0.0) + 0.2 * (out_now + 15.0).max(0.0)
    }
}

struct RoofSel {
    i: usize,
    ap: (f64, f64, f64, f64),
    key: (f64, f64, f64),
}

type Target<'a> = (f64, &'a ActorObs);

/// Mech ids are 0-15 (protocol: NPCs 100+, crates 1000+).
fn is_mech_id(id: u32) -> bool {
    id < 100
}

/// What the bot remembers about one mech: the input to mode signatures, concealment,
/// exposure, and silent walk.
struct Track {
    id: u32,
    pos: [f64; 3],
    yaw: f64,
    seen_t: f64,
    /// Last and first time it damaged us.
    hurt_t: f64,
    first_hurt_t: Option<f64>,
    /// We hit it before it first damaged us, so its attacks say nothing about the mode.
    provoked: bool,
    /// Signature B: (since, last decision it held) and whether it has counted.
    ignore: Option<(f64, f64)>,
    counted_b: bool,
}

impl Track {
    fn new(id: u32) -> Track {
        Track {
            id,
            pos: [0.0; 3],
            yaw: 0.0,
            seen_t: -1e9,
            hurt_t: -1e9,
            first_hurt_t: None,
            provoked: false,
            ignore: None,
            counted_b: false,
        }
    }

    fn eye(&self) -> [f64; 3] {
        [self.pos[0], self.pos[1], self.pos[2] + 4.5]
    }

    fn dist(&self, x: f64, y: f64) -> f64 {
        (self.pos[0] - x).hypot(self.pos[1] - y)
    }

    /// Its torso yaw points at (x, y) within `tol`.
    fn aims_at(&self, x: f64, y: f64, tol: f64) -> bool {
        wrap(self.yaw - (y - self.pos[1]).atan2(x - self.pos[0])).abs() < tol
    }
}

/// Signature A: a pair of unprovoked attackers that spare each other.
struct Pair {
    ids: (u32, u32),
    since: f64,
    last: f64,
    counted: bool,
}

/// Seconds a squad signature must hold before it counts. GUESS: STRATEGY.md 2.1 gives
/// 3 s for signature B; A uses the same.
const SIG_HOLD_S: f64 = 3.0;
/// A held condition survives gaps up to this between decisions.
const SIG_GAP_S: f64 = 0.3;
/// Concealment weight while recovering with `recover_los_break`. GUESS: worth 100 m of walking.
const RECOVER_W_CONCEAL: f64 = 100.0;
/// Concealment weight of the exposure-break spot, at least. GUESS.
const BREAK_W_CONCEAL: f64 = 100.0;
/// Spots ray-tested for concealment per pick, best first by the other terms.
const CONCEAL_TESTS: usize = 24;

pub struct Cbot {
    pub params: Params,
    p: ParamsF64,
    pub strategy: Strategy,
    /// Attach a debug label to each action.
    pub debug: bool,
    cfg: Config,
    slot: u32,
    rng: PyRandom,
    nav: Option<Arc<NavMap>>,
    flow: Flow,
    state: St,
    /// (bearing, t, attacker id)
    threat: Option<(f64, f64, u32)>,
    /// (bearing, t) from heard sounds
    alert: Option<(f64, f64)>,
    last_enemy_t: f64,
    last_damage_t: f64,
    /// (x, y, watch yaw)
    spot: Option<(f64, f64, f64)>,
    spot_key: Option<(i64, i64, i64)>,
    roof: Option<RoofSel>,
    roof_fail: Vec<f64>,
    climb_marked: bool,
    strafe_sign: i32,
    strafe_until: f64,
    slow_count: u32,
    stuck_until: f64,
    stuck_turn: i32,
    shell_threat: Option<(f64, f64, f64)>,
    laser_threat: Option<f64>,
    jet_hint: bool,
    recovering: bool,
    /// Free-for-all log-odds (STRATEGY.md 2.1) and its inputs.
    mode_l: f64,
    mode_drift: f64,
    prev_alive: Option<u32>,
    prev_t: f64,
    kills: u32,
    /// Aggro plays survive's rules this decision (`ffa_passive_on`).
    passive: bool,
    /// Known mechs, sorted by id.
    tracks: Vec<Track>,
    pairs: Vec<Pair>,
    /// Log-odds taken off by signatures A and B (each capped at 4), and whether any counted.
    sig_a: f64,
    sig_b: f64,
    sig_seen: bool,
    practice: bool,
    /// The free-for-all final rules apply this decision (`ffa_final_on`).
    final_: bool,
    /// The free-for-all early caution applies this decision (`ffa_early_on`).
    early: bool,
    /// Next time a held spot may be re-tested for concealment.
    spot_check_t: f64,
    /// Exposure-break spot and when it may be re-picked.
    break_spot: Option<(f64, f64)>,
    break_pick_t: f64,
    /// The gun was on target at this id last decision (`slew_gate`); the last target shot at.
    last_on: Option<u32>,
    last_target: Option<u32>,
}

impl Cbot {
    pub fn new(strategy: Strategy, params: Params) -> Cbot {
        Cbot {
            p: params.to_f64(),
            params,
            strategy,
            debug: false,
            cfg: Config::default(),
            slot: 0,
            rng: PyRandom::new(0),
            nav: None,
            flow: Flow::default(),
            state: St::Hold,
            threat: None,
            alert: None,
            last_enemy_t: -1e9,
            last_damage_t: -1e9,
            spot: None,
            spot_key: None,
            roof: None,
            roof_fail: Vec::new(),
            climb_marked: false,
            strafe_sign: 1,
            strafe_until: 0.0,
            slow_count: 0,
            stuck_until: -1.0,
            stuck_turn: 1,
            shell_threat: None,
            laser_threat: None,
            jet_hint: false,
            recovering: false,
            mode_l: 0.5,
            mode_drift: 0.0,
            prev_alive: None,
            prev_t: 0.0,
            kills: 0,
            passive: false,
            tracks: Vec::new(),
            pairs: Vec::new(),
            sig_a: 0.0,
            sig_b: 0.0,
            sig_seen: false,
            practice: false,
            final_: false,
            early: false,
            spot_check_t: 0.0,
            break_spot: None,
            break_pick_t: 0.0,
            last_on: None,
            last_target: None,
        }
    }

    /// The current state's name, for tests and debug labels.
    pub fn state_name(&self) -> &'static str {
        match self.state {
            St::Escape => "ESCAPE",
            St::Dodge => "DODGE",
            St::Retreat => "RETREAT",
            St::Break => "BREAK",
            St::Fight => "FIGHT",
            St::Rotate => "ROTATE",
            St::Drop => "DROP",
            St::Loot => "LOOT",
            St::Farm => "FARM",
            St::Hunt => "HUNT",
            St::Climb => "CLIMB",
            St::Perch => "PERCH",
            St::Hold => "HOLD",
        }
    }

    fn load_nav(&mut self, dir: &str) -> Arc<NavMap> {
        let m = &self.cfg.mech;
        let dims = MechDims {
            step_up: m.step_up,
            height: m.height,
            max_slope_deg: m.max_slope_deg,
        };
        let nav = NavMap::shared(dir, dims);
        self.roof_fail = vec![0.0; nav.roofs.len()];
        self.nav = Some(nav.clone());
        nav
    }

    // ================= memory =================
    fn remember(&mut self, c: &Ctx) {
        for ev in &c.obs.events {
            if let GameEvent::Damaged {
                bearing,
                by: Some(by),
                ..
            } = ev
            {
                self.last_damage_t = c.t;
                let mut b = *bearing;
                for (_, m) in c.all_actors() {
                    if m.id == *by {
                        b = Some((m.pos[1] - c.y).atan2(m.pos[0] - c.x));
                    }
                }
                if let Some(b) = b {
                    self.threat = Some((b, c.t, *by));
                }
            }
        }
        for s in &c.obs.sounds {
            let loud = matches!(
                s.kind,
                SoundKind::Gun
                    | SoundKind::Laser
                    | SoundKind::LaserCharge
                    | SoundKind::Lightning
                    | SoundKind::Mortar
                    | SoundKind::Step
                    | SoundKind::Boost
                    | SoundKind::Jet
            );
            if loud && matches!(s.dist, SoundDist::Near | SoundDist::Mid) {
                self.alert = Some((s.bearing, c.t));
            }
        }
        if !c.enemies.is_empty() {
            self.last_enemy_t = c.t;
        }
        for (_, m) in &c.enemies {
            let tr = self.track(m.id);
            tr.pos = m.pos;
            tr.yaw = m.aim[0];
            tr.seen_t = c.t;
        }
        for ev in &c.obs.events {
            match *ev {
                GameEvent::Hit { target, .. } if is_mech_id(target) => {
                    let tr = self.track(target);
                    if tr.first_hurt_t.is_none() {
                        tr.provoked = true;
                    }
                }
                GameEvent::Damaged { by: Some(by), .. } if is_mech_id(by) => {
                    let tr = self.track(by);
                    tr.hurt_t = c.t;
                    tr.first_hurt_t.get_or_insert(c.t);
                }
                _ => {}
            }
        }
    }

    fn track(&mut self, id: u32) -> &mut Track {
        let i = match self.tracks.binary_search_by_key(&id, |t| t.id) {
            Ok(i) => i,
            Err(i) => {
                self.tracks.insert(i, Track::new(id));
                i
            }
        };
        &mut self.tracks[i]
    }

    /// Eye points of the remembered mechs nearest to us, at most 4.
    fn watchers(&self, c: &Ctx) -> Vec<[f64; 3]> {
        let mut near: Vec<(f64, [f64; 3])> = self
            .tracks
            .iter()
            .filter(|t| c.t - t.seen_t < self.p.track_memory_s)
            .map(|t| (t.dist(c.x, c.y), t.eye()))
            .collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        near.into_iter().take(4).map(|(_, e)| e).collect()
    }

    fn threat_recent(&self, c: &Ctx, s: f64) -> bool {
        self.threat.is_some_and(|th| c.t - th.1 < s)
    }

    /// The visible actor that hurt us most recently.
    fn attacker<'a>(&self, c: &Ctx<'a>) -> Option<Target<'a>> {
        if !self.threat_recent(c, self.p.threat_memory_s) {
            return None;
        }
        let by = self.threat?.2;
        c.all_actors().find(|(_, m)| m.id == by).copied()
    }

    // ================= beyond parity =================
    /// STRATEGY.md 2.1, the strong signal and the weak drift. GUESS: positive evidence
    /// only (an unexplained death), since squad signatures are not tracked yet.
    fn read_mode(&mut self, c: &Ctx) {
        let ours = c
            .obs
            .events
            .iter()
            .filter(|e| matches!(e, GameEvent::Kill { target } if *target < 100))
            .count() as u32;
        self.kills += ours;
        let alive = c.obs.alive;
        if let Some(prev) = self.prev_alive {
            if alive < prev && c.t < 390.0 {
                self.mode_l += 4.0 * f64::from((prev - alive).saturating_sub(ours));
            }
        }
        let field = self.cfg.match_.mechs as u32;
        if c.t > 60.0 && alive + self.kills == field && self.mode_drift < 1.5 {
            let step = (0.03 * (c.t - self.prev_t).max(0.0)).min(1.5 - self.mode_drift);
            self.mode_drift += step;
            self.mode_l -= step;
        }
        if alive > 0 {
            self.prev_alive = Some(alive);
        }
        self.prev_t = c.t;
        self.read_signatures(c);
        let p = &self.p;
        let ffa = self.mode_l >= p.ffa_threshold;
        self.passive = p.ffa_passive_on > 0.5 && ffa && c.t >= p.ffa_passive_from_t;
        self.practice = self.mode_l < p.practice_threshold && self.sig_seen;
        self.final_ =
            p.ffa_final_on > 0.5 && !self.practice && c.obs.zone.next_radius <= p.ffa_final_r;
        let armed = c.me.weapons[1].is_some() || f64::from(c.me.cubes) >= p.early_cubes;
        self.early = p.ffa_early_on > 0.5
            && !self.practice
            && !armed
            && c.t < p.early_until_t
            && !self.final_;
    }

    /// Squad signatures A and B (STRATEGY.md 2.1), from unprovoked attackers only: a mech
    /// we shot first and that shoots back proves nothing, since free-for-all bots retaliate.
    /// GUESS: a signature counts only while each mech in it has damaged us within the last
    /// SIG_HOLD_S; one stray hit 10 s ago called practice in a passive sim lobby.
    fn read_signatures(&mut self, c: &Ctx) {
        let t = c.t;
        let tol = self.p.aim_at_me_rad;
        let att: Vec<usize> = (0..self.tracks.len())
            .filter(|&i| {
                let tr = &self.tracks[i];
                tr.seen_t == t && !tr.provoked && t - tr.hurt_t < 10.0
            })
            .collect();
        // A: two of them within 100 m, in sight of each other, neither aiming at the other.
        for (k, &i) in att.iter().enumerate() {
            for &j in &att[k + 1..] {
                let (a, b) = (&self.tracks[i], &self.tracks[j]);
                let ok = a.dist(b.pos[0], b.pos[1]) < 100.0
                    && !a.aims_at(b.pos[0], b.pos[1], tol)
                    && !b.aims_at(a.pos[0], a.pos[1], tol)
                    && c.nav.los(a.eye(), b.eye());
                if !ok {
                    continue;
                }
                let ids = (a.id, b.id);
                let pair = match self.pairs.iter().position(|p| p.ids == ids) {
                    Some(n) => &mut self.pairs[n],
                    None => {
                        self.pairs.push(Pair {
                            ids,
                            since: t,
                            last: t,
                            counted: false,
                        });
                        self.pairs.last_mut().expect("just pushed")
                    }
                };
                if t - pair.last > SIG_GAP_S {
                    pair.since = t;
                }
                pair.last = t;
                let firing = t - a.hurt_t <= SIG_HOLD_S && t - b.hurt_t <= SIG_HOLD_S;
                if !pair.counted && firing && t - pair.since >= SIG_HOLD_S {
                    pair.counted = true;
                    self.sig_seen = true;
                    if self.sig_a > -4.0 {
                        self.sig_a -= 2.0;
                        self.mode_l -= 2.0;
                    }
                }
            }
        }
        // B: one of them ignores a nearer mech it can see.
        for &i in &att {
            if self.tracks[i].counted_b {
                continue;
            }
            let a = &self.tracks[i];
            let to_me = a.dist(c.x, c.y);
            let ignores = self.tracks.iter().any(|m| {
                m.id != a.id
                    && m.seen_t == t
                    && a.dist(m.pos[0], m.pos[1]) < to_me
                    && !a.aims_at(m.pos[0], m.pos[1], tol)
                    && c.nav.los(a.eye(), m.eye())
            });
            let a = &mut self.tracks[i];
            if !ignores {
                a.ignore = None;
                continue;
            }
            let since = match a.ignore {
                Some((since, last)) if t - last <= SIG_GAP_S => since,
                _ => t,
            };
            a.ignore = Some((since, t));
            if t - since >= SIG_HOLD_S && t - a.hurt_t <= SIG_HOLD_S {
                a.counted_b = true;
                self.sig_seen = true;
                if self.sig_b > -4.0 {
                    self.sig_b -= 2.0;
                    self.mode_l -= 2.0;
                }
            }
        }
    }

    /// TACTICS.md probe 1, aggro only.
    fn recover(&mut self, c: &Ctx) {
        if self.p.recover_on <= 0.5 || self.strategy != Strategy::Aggro {
            self.recovering = false;
            return;
        }
        let p = &self.p;
        let pressed = self.threat_recent(c, p.recover_quiet_s)
            || c.enemies.iter().any(|(d, _)| *d < p.recover_clear_dist);
        if pressed || c.hp_frac >= p.recover_exit_frac {
            self.recovering = false;
        } else if c.hp_frac < p.recover_frac && !self.recovering {
            self.recovering = true;
            self.spot = None;
        }
    }

    /// The free-for-all log-odds, for tests and tuning reports.
    pub fn mode_log_odds(&self) -> f64 {
        self.mode_l
    }

    pub fn is_recovering(&self) -> bool {
        self.recovering
    }

    /// The hold spot `(x, y)` last chosen.
    pub fn hold_spot(&self) -> Option<(f64, f64)> {
        self.spot.map(|(x, y, _)| (x, y))
    }

    /// The mode reads practice: low log-odds and a squad signature seen.
    pub fn is_practice(&self) -> bool {
        self.practice
    }

    /// The free-for-all final rules applied at the last decision.
    pub fn in_final(&self) -> bool {
        self.final_
    }

    /// The free-for-all early caution applied at the last decision.
    pub fn in_early_caution(&self) -> bool {
        self.early
    }

    /// TACTICS #12: walk under the step-sound speed in calm states with a mech remembered nearby.
    fn silent(&self, c: &Ctx) -> bool {
        let p = &self.p;
        p.silent_speed_frac > 0.0
            && matches!(
                self.state,
                St::Rotate | St::Hold | St::Loot | St::Drop | St::Perch
            )
            && !self.zone_urgent(c)
            && self
                .tracks
                .iter()
                .any(|t| c.t - t.seen_t < p.track_memory_s && t.dist(c.x, c.y) < p.silent_range)
    }

    // ================= reflexes =================
    fn reflex_scan(&mut self, c: &Ctx) {
        self.shell_threat = None;
        let mortar = &self.cfg.weapons.mortar;
        let g = mortar.gravity;
        let sr = mortar.splash_radius + c.me.stats.radius + self.p.shell_margin;
        for pr in &c.obs.projectiles {
            if pr.kind != ProjectileKind::Mortar {
                continue;
            }
            if let Some(land) = shell_landing(c.nav, pr.pos, pr.vel, g) {
                if (land.0 - c.x).hypot(land.1 - c.y) < sr {
                    self.shell_threat = Some(land);
                }
            }
        }
        self.laser_threat = None;
        for (d, m) in &c.enemies {
            if m.charging == Some(true) && *d < self.p.laser_threat_range {
                let to_me = (c.y - m.pos[1]).atan2(c.x - m.pos[0]);
                if wrap(m.aim[0] - to_me).abs() < self.p.laser_dodge_rad + 4f64.atan2(d.max(1.0)) {
                    self.laser_threat = Some(to_me);
                }
            }
        }
    }

    fn unstick(
        &mut self,
        c: &Ctx,
        mv: (f64, f64),
        boost: bool,
        jump: bool,
    ) -> ((f64, f64), bool, bool) {
        let p = &self.p;
        let want = mv.0.hypot(mv.1) > 0.5;
        let speed = c.vx.hypot(c.vy);
        if want && speed < p.stuck_speed && c.me.grounded && c.t >= self.stuck_until {
            self.slow_count += 1;
        } else if speed >= p.stuck_speed {
            self.slow_count = 0;
        }
        if f64::from(self.slow_count) >= p.stuck_decisions {
            self.slow_count = 0;
            self.stuck_until = c.t + p.stuck_jet_s;
            self.stuck_turn = -self.stuck_turn;
        }
        if c.t < self.stuck_until && want {
            let a = mv.1.atan2(mv.0) + f64::from(self.stuck_turn) * 0.6;
            return ((a.cos(), a.sin()), true, true);
        }
        (mv, boost, jump)
    }

    // ================= navigation =================
    /// Unit move toward (x, y): the flow field when it is ready, sampled clear headings otherwise.
    fn go(&mut self, c: &Ctx, x: f64, y: f64, r: f64) -> (f64, f64) {
        let nav = c.nav;
        if c.z - c.ground > 3.0 {
            return unit(x - c.x, y - c.y);
        }
        self.flow.request(nav, x, y, r);
        if self.flow.field_key == Some(Flow::key(nav, x, y, r)) {
            match self.flow.dir(nav, c.x, c.y) {
                Some((0.0, 0.0)) => {
                    return if (x - c.x).hypot(y - c.y) > r {
                        unit(x - c.x, y - c.y)
                    } else {
                        (0.0, 0.0)
                    };
                }
                Some(d) => return d,
                None => {
                    // the goal is in another walkable component: head straight at it and jet over
                    self.jet_hint = true;
                    return unit(x - c.x, y - c.y);
                }
            }
        }
        let (gx, gy) = unit(x - c.x, y - c.y);
        best_heading(c, |dx, dy| dx * gx + dy * gy, 10.0)
    }

    fn zone_urgent(&self, c: &Ctx) -> bool {
        if !c.me.in_zone {
            return true;
        }
        let d = c.dist_region();
        if d <= 0.0 {
            return false;
        }
        let z = &c.obs.zone;
        let time_left = if c.t >= z.shrink_start {
            z.shrink_end - c.t
        } else {
            z.shrink_start - c.t + (z.shrink_end - z.shrink_start)
        };
        d / c.me.stats.walk_speed + self.p.zone_slack_s > time_left
    }

    // ================= hold spots =================
    fn choose_spot(&mut self, c: &Ctx) {
        let p = &self.p;
        let w = c.nav;
        let (cx, cy, r) = c.region;
        let (ax, ay) = if c.dist_region() <= 0.0 {
            (c.x, c.y)
        } else {
            let l = (c.x - cx).hypot(c.y - cy).max(1.0);
            (cx + (c.x - cx) * r / l, cy + (c.y - cy) * r / l)
        };
        let sr = p.spot_search_r;
        let stride = if sr > 80.0 { 2 } else { 1 };
        let threats: Vec<[f64; 3]> = c.enemies.iter().take(4).map(|(_, m)| m.pos).collect();
        let (mut best, mut bs) = (None, 1e18);
        let w_conceal = self.conceal_w();
        let mut cands = Vec::new();
        let i0 = ((ax - w.rx0) / w.cs) as i64;
        let j0 = ((ay - w.ry0) / w.cs) as i64;
        let span = (sr / w.cs) as i64;
        let corner = self.strategy.corner_spots();
        let my_comp = w.component(c.x, c.y);
        let (nw, nh) = (w.nw as i64, w.nh as i64);
        let mut j = (j0 - span).max(1);
        while j < (j0 + span + 1).min(nh - 1) {
            let mut i = (i0 - span).max(1);
            while i < (i0 + span + 1).min(nw - 1) {
                let cc = (j * nw + i) as usize;
                i += stride;
                if w.blocked[cc] != 0 || (my_comp != 0 && w.comp[cc] != my_comp) {
                    continue;
                }
                let (x, y) = w.ncenter(cc);
                let dc = (x - cx).hypot(y - cy);
                if dc > r.max(8.0) {
                    continue;
                }
                let cov = f64::from(w.cover[cc]);
                let w_center = if self.early { 0.0 } else { p.spot_w_center };
                let mut s = (x - c.x).hypot(y - c.y)
                    + p.spot_w_cover * w.cs * (cov - 1.0).powi(2) * 0.5
                    + w_center * dc;
                if corner {
                    let n = w.nw;
                    let walls: u32 = [
                        cc + 1,
                        cc - 1,
                        cc + n,
                        cc - n,
                        cc + n + 1,
                        cc + n - 1,
                        cc - n + 1,
                        cc - n - 1,
                    ]
                    .iter()
                    .map(|&o| u32::from(w.blocked[o]))
                    .sum();
                    s -= 15.0 * f64::from(walls.min(4));
                }
                for t in &threats {
                    let dd = (x - t[0]).hypot(y - t[1]);
                    if dd < 100.0 {
                        s += p.spot_w_enemy * (1.0 - dd / 100.0);
                    }
                }
                if w_conceal > 0.0 {
                    cands.push((s, x, y));
                }
                if s < bs {
                    best = Some((x, y));
                    bs = s;
                }
            }
            j += stride;
        }
        if w_conceal > 0.0 {
            best = concealed_best(c, cands, &self.watchers(c), w_conceal).or(best);
        }
        let (x, y) = best.unwrap_or((cx, cy));
        let cc = w.ncell(x, y);
        let (mut ox, mut oy) = (0i64, 0i64);
        for (di, dj, _) in NBR {
            let k = cc + dj * w.nw as i64 + di;
            if (0..w.blocked.len() as i64).contains(&k) && w.blocked[k as usize] != 0 {
                ox -= di;
                oy -= dj;
            }
        }
        let watch = if ox != 0 || oy != 0 {
            (oy as f64).atan2(ox as f64)
        } else {
            (cy - y).atan2(cx - x)
        };
        self.spot = Some((x, y, watch));
        self.spot_key = Some((pround(cx), pround(cy), pround(r)));
    }

    /// Concealment weight of hold spots this decision; 0 keeps Python's spots.
    fn conceal_w(&self) -> f64 {
        let p = &self.p;
        let mut w = p.spot_w_conceal;
        if self.final_ {
            w = w.max(p.final_w_conceal);
        }
        if self.recovering && p.recover_los_break > 0.5 {
            w = w.max(RECOVER_W_CONCEAL);
        }
        w
    }

    fn braking(&self, c: &Ctx) -> bool {
        self.p.centre_brake > 0.5 && c.obs.zone.next_radius <= self.p.brake_r
    }

    fn ensure_spot(&mut self, c: &Ctx) -> (f64, f64, f64) {
        if self.braking(c) {
            // the exact next centre (or the free cell nearest it), facing the nearest remembered mech
            let [mut x, mut y] = c.obs.zone.next_center;
            if let Some(cell) = c.nav.nearest_free(x, y, 3).filter(|_| !c.nav.free(x, y)) {
                (x, y) = c.nav.ncenter(cell);
            }
            let watch = self
                .watchers(c)
                .first()
                .map_or(c.me.aim[0], |e| (e[1] - y).atan2(e[0] - x));
            self.spot = Some((x, y, watch));
            self.spot_key = None;
            return (x, y, watch);
        }
        let (cx, cy, r) = c.region;
        let mut stale = match self.spot {
            None => true,
            Some((x, y, _)) => {
                self.spot_key != Some((pround(cx), pround(cy), pround(r)))
                    || (x - cx).hypot(y - cy) > r.max(8.0)
            }
        };
        if !stale && self.conceal_w() > 0.0 && c.t >= self.spot_check_t {
            // at most once a second: leave a spot a remembered mech can see
            self.spot_check_t = c.t + 1.0;
            let (x, y, _) = self.spot.expect("not stale");
            let at = [x, y, c.nav.ground(x, y) + 4.5];
            stale = self.watchers(c).iter().any(|e| c.nav.los(*e, at));
        }
        if stale {
            self.choose_spot(c);
        }
        self.spot.expect("choose_spot sets a spot")
    }

    fn watch_yaw(&self, c: &Ctx, base: f64) -> f64 {
        base + self.p.scan_amp * (c.t * self.p.scan_rate + f64::from(self.slot)).sin()
    }

    // ================= roof =================
    fn choose_roof(&mut self, c: &Ctx) {
        let w = c.nav;
        let (cx, cy, r) = c.region;
        loop {
            let p = &self.p;
            let (mut best, mut bs) = (None, 1e18);
            for (i, rf) in w.roofs.iter().enumerate() {
                if self.roof_fail[i] >= p.roof_max_tries || rf.area < p.roof_min_area {
                    continue;
                }
                let (x, y) = rf.c;
                let h = rf.top - w.ground(x, y);
                if !(p.roof_min_h <= h && h <= p.roof_max_h) {
                    continue;
                }
                if (x - cx).hypot(y - cy) > r - 10.0 {
                    continue;
                }
                let d = (x - c.x).hypot(y - c.y);
                if d > p.roof_max_dist {
                    continue;
                }
                if d < bs {
                    best = Some(i);
                    bs = d;
                }
            }
            let Some(best) = best else {
                self.roof = None;
                return;
            };
            match w.approach_point(&w.roofs[best], c.x, c.y) {
                Some(ap) => {
                    self.roof = Some(RoofSel {
                        i: best,
                        ap,
                        key: c.region,
                    });
                    return;
                }
                None => self.roof_fail[best] = p.roof_max_tries,
            }
        }
    }

    fn on_roof(&self, c: &Ctx) -> bool {
        let Some(roof) = &self.roof else { return false };
        let top = c.nav.roofs[roof.i].top;
        c.me.grounded && (c.z - top).abs() < 1.5 && (c.nav.roof_top(c.x, c.y) - top).abs() < 1.5
    }

    fn roof_valid(&mut self, c: &Ctx) -> bool {
        let moved = self
            .roof
            .as_ref()
            .is_none_or(|rf| (rf.key.0, rf.key.1) != (c.region.0, c.region.1));
        if moved {
            let on_roof = self.on_roof(c);
            if let Some(roof) = self.roof.as_mut().filter(|_| on_roof) {
                let (cx, cy, r) = c.region;
                let (x, y) = c.nav.roofs[roof.i].c;
                if (x - cx).hypot(y - cy) <= r {
                    roof.key = c.region;
                    return true;
                }
            }
            self.choose_roof(c);
        }
        self.roof.is_some()
    }

    // ================= state predicates =================
    fn wants(&mut self, s: St, c: &Ctx) -> bool {
        match s {
            St::Escape => self.zone_urgent(c),
            St::Dodge => self.shell_threat.is_some() || self.laser_threat.is_some(),
            St::Retreat => self.want_retreat(c),
            St::Fight => self.fight_target(c).is_some(),
            St::Rotate => {
                if self.strategy == Strategy::Roof && self.roof.is_some() && !self.on_roof(c) {
                    return false;
                }
                let (x, y, _) = self.ensure_spot(c);
                let reach = self.p.spot_reach * if self.state == St::Hold { 2.0 } else { 1.0 };
                (x - c.x).hypot(y - c.y) > reach
            }
            St::Drop => self.drop_target(c).is_some(),
            St::Loot => self.loot_target(c).is_some(),
            St::Farm => self.farm_target(c).is_some(),
            St::Hunt => !self.recovering && !self.passive && !self.final_ && !self.early,
            St::Break => self.want_break(c),
            St::Hold => true,
            St::Climb => self.roof_valid(c) && !self.on_roof(c),
            St::Perch => self.roof.is_some() && self.on_roof(c),
        }
    }

    fn want_retreat(&self, c: &Ctx) -> bool {
        let p = &self.p;
        let window = if self.state == St::Retreat {
            p.retreat_memory_s
        } else {
            p.threat_memory_s
        };
        if !self.threat_recent(c, window) {
            return false;
        }
        if self
            .attacker(c)
            .is_some_and(|a| a.0 < p.retreat_gives_up_dist)
        {
            return false; // cornered: fight it out
        }
        if self.state == St::Retreat {
            c.hp_frac < p.resume_frac
        } else {
            c.hp_frac < p.retreat_frac
        }
    }

    /// TACTICS #5: enough mechs seen lately within fire range, and no attacker close enough to corner us.
    fn want_break(&mut self, c: &Ctx) -> bool {
        let p = &self.p;
        if p.exposed_break_n < 1.5 || self.final_ {
            // the final's concealed hold already hides us; a crowded last circle would keep BREAK on
            self.break_spot = None;
            return false;
        }
        let seen = self
            .tracks
            .iter()
            .filter(|t| c.t - t.seen_t < p.break_hold_s.max(0.1) && t.dist(c.x, c.y) < p.fire_range)
            .count();
        let on = seen as f64 >= p.exposed_break_n.round()
            && !self
                .attacker(c)
                .is_some_and(|a| a.0 < p.retreat_gives_up_dist);
        if !on {
            self.break_spot = None;
        }
        on
    }

    /// The nearest walkable spot inside the zone that the remembered mechs cannot see,
    /// re-picked at most once a second.
    fn ensure_break_spot(&mut self, c: &Ctx) -> (f64, f64) {
        if let Some(spot) = self.break_spot.filter(|_| c.t < self.break_pick_t) {
            return spot;
        }
        self.break_pick_t = c.t + 1.0;
        let p = &self.p;
        let w = c.nav;
        let zone = &c.obs.zone;
        let my_comp = w.component(c.x, c.y);
        let span = (p.break_search_r / w.cs) as i64;
        let i0 = ((c.x - w.rx0) / w.cs) as i64;
        let j0 = ((c.y - w.ry0) / w.cs) as i64;
        let (nw, nh) = (w.nw as i64, w.nh as i64);
        let mut cands = Vec::new();
        for j in (j0 - span).max(1)..(j0 + span + 1).min(nh - 1) {
            for i in (i0 - span).max(1)..(i0 + span + 1).min(nw - 1) {
                let cc = (j * nw + i) as usize;
                if w.blocked[cc] != 0 || (my_comp != 0 && w.comp[cc] != my_comp) {
                    continue;
                }
                let (x, y) = w.ncenter(cc);
                let d = (x - c.x).hypot(y - c.y);
                if d > p.break_search_r
                    || (x - zone.center[0]).hypot(y - zone.center[1]) > zone.radius - 5.0
                {
                    continue;
                }
                let cov = f64::from(w.cover[cc]);
                let s = d
                    + p.spot_w_cover * w.cs * (cov - 1.0).powi(2) * 0.5
                    + 0.03 * c.dist_region_at(x, y).max(0.0);
                cands.push((s, x, y));
            }
        }
        let wc = p.spot_w_conceal.max(BREAK_W_CONCEAL);
        let spot = concealed_best(c, cands, &self.watchers(c), wc).unwrap_or((c.x, c.y));
        self.break_spot = Some(spot);
        spot
    }

    /// Aggro's own rules apply: the strategy is aggro and the free-for-all switch is not holding it back.
    fn aggro(&self) -> bool {
        self.strategy == Strategy::Aggro && !self.passive && !self.final_ && !self.early
    }

    /// Under the final or early free-for-all rules, a mech worth shooting unprovoked:
    /// low HP, and in the early game also isolated.
    fn worth_it(&self, c: &Ctx, m: &ActorObs) -> bool {
        if !self.finishable(m) {
            return false;
        }
        !self.early
            || !c.enemies.iter().any(|(_, o)| {
                o.id != m.id && (o.pos[0] - m.pos[0]).hypot(o.pos[1] - m.pos[1]) < self.p.isolated_r
            })
    }

    /// In the free-for-all final, a mech worth shooting without being attacked: shown HP under `finish_hp`.
    fn finishable(&self, m: &ActorObs) -> bool {
        m.hp.is_some_and(|hp| hp < self.p.finish_hp)
    }

    fn fight_target<'a>(&self, c: &Ctx<'a>) -> Option<Target<'a>> {
        if self.recovering {
            return None;
        }
        if let Some(att) = self.attacker(c).filter(|a| a.0 < self.p.fire_range) {
            return Some(att);
        }
        if self.final_ || self.early {
            return c
                .enemies
                .iter()
                .find(|(d, m)| *d < self.p.fire_range && self.worth_it(c, m))
                .copied();
        }
        c.enemies
            .iter()
            .find(|(d, _)| *d < self.p.engage_range || self.aggro())
            .copied()
    }

    fn drop_target<'a>(&self, c: &Ctx<'a>) -> Option<&'a DropObs> {
        let p = &self.p;
        if c.t - self.last_enemy_t < p.quiet_s {
            return None;
        }
        c.obs.drops.iter().find(|d| {
            if d.holder.is_some_and(|h| h != c.me.id) {
                return false;
            }
            let (x, y) = (d.pos[0], d.pos[1]);
            let dist = (x - c.x).hypot(y - c.y);
            if dist > p.drop_max_dist || c.dist_region_at(x, y) > 0.0 {
                return false;
            }
            if dist / c.me.stats.walk_speed + 3.0 < d.lands_in {
                return false;
            }
            !c.enemies
                .iter()
                .any(|(_, m)| (m.pos[0] - x).hypot(m.pos[1] - y) < p.drop_enemy_clear)
        })
    }

    fn loot_target(&self, c: &Ctx) -> Option<[f64; 3]> {
        let p = &self.p;
        if c.enemies.iter().any(|(d, _)| *d < p.loot_enemy_clear) {
            return None;
        }
        let (mut best, mut bd) = (None, 1e18);
        for pk in &c.obs.pickups {
            if !wanted_weapon(pk.kind, c.me, self.p.no_mortar > 0.5) {
                continue;
            }
            let d = (pk.pos[0] - c.x).hypot(pk.pos[1] - c.y);
            if d < p.loot_detour && (pk.pos[2] - c.z).abs() < 4.0 && d < bd {
                best = Some(pk.pos);
                bd = d;
            }
        }
        for cu in &c.obs.cubes {
            let d = (cu.pos[0] - c.x).hypot(cu.pos[1] - c.y);
            if d < p.cube_detour && (cu.pos[2] - c.z).abs() < 4.0 && d < bd {
                best = Some(cu.pos);
                bd = d;
            }
        }
        best
    }

    fn farm_target<'a>(&self, c: &Ctx<'a>) -> Option<Target<'a>> {
        let p = &self.p;
        if self.recovering
            || c.hp_frac < p.farm_min_hp
            || c.t > p.farm_until_t
            || c.t - self.last_enemy_t < p.quiet_s
        {
            return None;
        }
        c.npcs.iter().find(|(d, _)| *d < p.farm_range).copied()
    }

    // ================= state behaviours: (move, boost, jump, look yaw) =================
    fn threat_look(&self, c: &Ctx, fallback: f64) -> f64 {
        if self.threat_recent(c, self.p.face_threat_s) {
            return self.threat.expect("recent").0;
        }
        if let Some((b, _)) = self.alert.filter(|a| c.t - a.1 < self.p.alert_s) {
            return b;
        }
        fallback
    }

    fn handle(&mut self, s: St, c: &Ctx) -> ((f64, f64), bool, bool, f64) {
        let aim0 = c.me.aim[0];
        let energy = c.me.energy;
        match s {
            St::Escape => {
                let (x, y, _) = self.ensure_spot(c);
                let mv = self.go(c, x, y, 3.0);
                let boost = !c.me.in_zone || energy > 50.0;
                (mv, boost, false, self.threat_look(c, heading_or(mv, aim0)))
            }
            St::Dodge => {
                let (ax, ay) = if let Some((lx, ly, _)) = self.shell_threat {
                    let (ax, ay) = unit(c.x - lx, c.y - ly);
                    if ax == 0.0 && ay == 0.0 {
                        ((aim0 + 1.57).cos(), (aim0 + 1.57).sin())
                    } else {
                        (ax, ay)
                    }
                } else {
                    let b = self.laser_threat.expect("dodge needs a threat");
                    let sign = f64::from(self.strafe_sign);
                    ((b + PI / 2.0).cos() * sign, (b + PI / 2.0).sin() * sign)
                };
                let mv = best_heading(
                    c,
                    |dx, dy| dx * ax + dy * ay - c.region_penalty(dx, dy),
                    8.0,
                );
                (mv, true, false, self.threat_look(c, aim0))
            }
            St::Retreat => {
                let b = self.threat.expect("retreat needs a threat").0;
                let (ax, ay) = (-b.cos(), -b.sin());
                let w = c.nav;
                let mv = best_heading(
                    c,
                    |dx, dy| {
                        let nc = w.ncell(c.x + dx * 12.0, c.y + dy * 12.0);
                        let cov = if nc >= 0 {
                            f64::from(w.cover[nc as usize])
                        } else {
                            15.0
                        };
                        dx * ax + dy * ay + 0.08 * (3.0 - cov.min(6.0)) - c.region_penalty(dx, dy)
                    },
                    10.0,
                );
                (mv, energy > 25.0, false, b)
            }
            St::Fight => self.do_fight(c),
            St::Break => {
                let (x, y) = self.ensure_break_spot(c);
                let d = (x - c.x).hypot(y - c.y);
                let mv = if d < 2.0 {
                    (0.0, 0.0)
                } else {
                    self.go(c, x, y, 2.0)
                };
                let near = c.enemies.first().map_or(heading_or(mv, aim0), |(_, m)| {
                    (m.pos[1] - c.y).atan2(m.pos[0] - c.x)
                });
                (
                    mv,
                    d > 10.0 && energy > 30.0,
                    false,
                    self.threat_look(c, near),
                )
            }
            St::Rotate => self.do_rotate(c),
            St::Hold => {
                let (x, y, watch) = self.ensure_spot(c);
                let d = (x - c.x).hypot(y - c.y);
                if self.braking(c) {
                    // GUESS: full speed until 6 m out, then proportional, stopping within 0.5 m
                    let (ux, uy) = unit(x - c.x, y - c.y);
                    let k = if d < 0.5 { 0.0 } else { (d / 6.0).min(1.0) };
                    return (
                        (ux * k, uy * k),
                        false,
                        false,
                        self.threat_look(c, self.watch_yaw(c, watch)),
                    );
                }
                let mut mv = if d < 2.0 {
                    (0.0, 0.0)
                } else {
                    self.go(c, x, y, 2.0)
                };
                if d < 4.0 {
                    mv = (mv.0 * 0.3, mv.1 * 0.3); // under step_speed: silent
                }
                (
                    mv,
                    false,
                    false,
                    self.threat_look(c, self.watch_yaw(c, watch)),
                )
            }
            St::Drop => {
                let d = self.drop_target(c).expect("drop needs a target");
                let (x, y) = (d.pos[0], d.pos[1]);
                let dist = (x - c.x).hypot(y - c.y);
                let mv = if dist < d.radius * 0.5 {
                    (0.0, 0.0)
                } else {
                    self.go(c, x, y, 2.0)
                };
                (
                    mv,
                    dist > 40.0 && energy > 60.0,
                    false,
                    self.threat_look(c, self.watch_yaw(c, aim0)),
                )
            }
            St::Loot => {
                let [x, y, _] = self.loot_target(c).expect("loot needs a target");
                let mv = if (x - c.x).hypot(y - c.y) < 10.0 {
                    unit(x - c.x, y - c.y)
                } else {
                    self.go(c, x, y, 1.5)
                };
                (mv, false, false, self.threat_look(c, heading_or(mv, aim0)))
            }
            St::Farm => {
                let (d, m) = self.farm_target(c).expect("farm needs a target");
                let b = (m.pos[1] - c.y).atan2(m.pos[0] - c.x);
                let mv = if d > 60.0 {
                    self.go(c, m.pos[0], m.pos[1], 50.0)
                } else {
                    (0.0, 0.0)
                };
                (mv, false, false, b)
            }
            St::Hunt => {
                if let Some((_, m)) = c.enemies.first() {
                    let (gx, gy) = (snap20(m.pos[0]), snap20(m.pos[1]));
                    let mv = self.go(c, gx, gy, 20.0);
                    return (
                        mv,
                        energy > 60.0,
                        false,
                        (m.pos[1] - c.y).atan2(m.pos[0] - c.x),
                    );
                }
                if let Some((b, _)) = self.alert.filter(|a| c.t - a.1 < 8.0) {
                    let mv = best_heading(
                        c,
                        |dx, dy| dx * b.cos() + dy * b.sin() - c.region_penalty(dx, dy),
                        10.0,
                    );
                    return (mv, false, false, b);
                }
                self.do_rotate(c)
            }
            St::Climb => self.do_climb(c),
            St::Perch => {
                let (tx, ty) = c.nav.roofs[self.roof.as_ref().expect("perch needs a roof").i].c;
                let d = (tx - c.x).hypot(ty - c.y);
                let mut mv = if d < 3.0 {
                    (0.0, 0.0)
                } else {
                    unit(tx - c.x, ty - c.y)
                };
                if d < 6.0 {
                    mv = (mv.0 * 0.3, mv.1 * 0.3);
                }
                let (cx, cy, _) = c.region;
                (
                    mv,
                    false,
                    false,
                    self.threat_look(c, self.watch_yaw(c, (c.y - cy).atan2(c.x - cx))),
                )
            }
        }
    }

    fn do_fight(&mut self, c: &Ctx) -> ((f64, f64), bool, bool, f64) {
        let (d, m) = self.fight_target(c).expect("fight needs a target");
        let (bx, by) = unit(m.pos[0] - c.x, m.pos[1] - c.y);
        if c.t > self.strafe_until {
            self.strafe_sign = self.rng.sign();
            self.strafe_until = c.t + self.rng.uniform(self.p.strafe_s_min, self.p.strafe_s_max);
        }
        let sign = f64::from(self.strafe_sign);
        let (px, py) = (-by * sign, bx * sign);
        let aggro = self.aggro();
        let (mut lo, mut hi) = (self.p.fight_min_dist, self.p.fight_max_dist);
        if m.weapons
            .as_ref()
            .is_some_and(|w| w.contains(&WeaponKind::Lightning))
        {
            lo = lo.max(self.p.lightning_keep_away);
        }
        if aggro {
            (lo, hi) = (10.0, self.p.aggro_close_dist);
            let held = |k: WeaponKind| c.me.weapons.iter().flatten().any(|w| w.kind() == k);
            if self.p.range_by_loadout_on > 0.5
                && !held(WeaponKind::Lightning)
                && !held(WeaponKind::Mortar)
            {
                (lo, hi) = if held(WeaponKind::Laser) {
                    (self.p.aggro_laser_min, self.p.aggro_laser_max)
                } else {
                    (self.p.aggro_gun_min, self.p.aggro_gun_max)
                };
            }
        }
        let mut radial = 0.0;
        if d < lo {
            radial = -1.0;
        } else if d > hi {
            radial = if aggro { 1.0 } else { 0.0 };
        }
        let look = by.atan2(bx);
        if aggro && d > hi {
            let mv = self.go(c, snap20(m.pos[0]), snap20(m.pos[1]), hi * 0.8);
            return (mv, c.me.energy > self.p.chase_boost_energy, false, look);
        }
        let (gx, gy) = (px + radial * bx, py + radial * by);
        let mv = best_heading(
            c,
            |dx, dy| dx * gx + dy * gy - c.region_penalty(dx, dy),
            8.0,
        );
        (mv, false, false, look)
    }

    fn do_rotate(&mut self, c: &Ctx) -> ((f64, f64), bool, bool, f64) {
        let (x, y, watch) = self.ensure_spot(c);
        let mv = self.go(c, x, y, 3.0);
        let boost = self.zone_urgent(c) || (c.me.energy > 70.0 && (x - c.x).hypot(y - c.y) > 60.0);
        (mv, boost, false, self.threat_look(c, heading_or(mv, watch)))
    }

    fn do_climb(&mut self, c: &Ctx) -> ((f64, f64), bool, bool, f64) {
        let roof = self.roof.as_ref().expect("climb needs a roof");
        let rf = &c.nav.roofs[roof.i];
        let (qx, qy, ex, ey) = roof.ap;
        let (tx, ty) = rf.c;
        let top = rf.top;
        let need = (top - c.ground) * self.p.roof_energy_per_m + 10.0;
        let d_ap = (qx - c.x).hypot(qy - c.y);
        let airborne = !c.me.grounded;
        let near_wall = (ex - c.x).hypot(ey - c.y) < 9.0;
        let to_center = (ty - c.y).atan2(tx - c.x);
        if airborne && c.z > c.ground + 1.0 {
            // in the climb: jet until the feet clear the roof, then move over it
            let mv = unit(tx - c.x, ty - c.y);
            let clear = c.z > top + 0.4;
            if !clear && c.me.energy < 2.0 && !self.climb_marked {
                self.climb_marked = true;
                self.climb_failed();
            }
            let mv = if clear || near_wall {
                mv
            } else {
                unit(ex - c.x, ey - c.y)
            };
            return (mv, !clear, false, to_center);
        }
        self.climb_marked = false;
        if d_ap > 4.0 && !near_wall {
            let mv = self.go(c, qx, qy, 2.0);
            return (mv, false, false, self.threat_look(c, heading_or(mv, 0.0)));
        }
        if c.me.energy < need {
            return (
                (0.0, 0.0),
                false,
                false,
                self.threat_look(c, (qy - ty).atan2(qx - tx)),
            );
        }
        (unit(ex - c.x, ey - c.y), true, true, to_center)
    }

    fn climb_failed(&mut self) {
        let Some(roof) = &self.roof else { return };
        let i = roof.i;
        self.roof_fail[i] += 1.0;
        if self.roof_fail[i] >= self.p.roof_max_tries {
            self.roof = None;
        }
    }

    // ================= gunnery =================
    fn pick_target<'a>(&self, c: &Ctx<'a>) -> Option<(f64, &'a ActorObs, [f64; 3])> {
        let p = &self.p;
        if self.recovering {
            return None;
        }
        let mode = if self.passive {
            FireMode::Engaged
        } else {
            self.strategy.fire()
        };
        let attacked = self.threat_recent(c, p.threat_memory_s);
        if mode == FireMode::Attacked && !attacked {
            return None;
        }
        if self.state == St::Retreat && c.t - self.last_damage_t > p.retreat_quiet_fire_s {
            return None; // once the hits stop, dealing damage would only reset the regen clock
        }
        let att = self.attacker(c);
        let mut cands: Vec<Target> = Vec::new();
        if let Some(a) = att {
            cands.push(a);
        }
        cands.extend(
            c.enemies
                .iter()
                .filter(|e| att.is_none_or(|a| e.1.id != a.1.id)),
        );
        if self.state == St::Farm {
            if let Some(ft) = self.farm_target(c) {
                cands.insert(0, ft);
            }
        }
        if p.target_score_on > 0.5 {
            let score = |&(d, m): &Target| {
                let hp = match (m.hp, m.max_hp) {
                    (Some(hp), Some(max)) if max > 0.0 => hp / max,
                    _ => 1.0,
                };
                let to_me = (c.y - m.pos[1]).atan2(c.x - m.pos[0]);
                let facing = wrap(m.aim[0] - to_me).abs() < p.aim_at_me_rad;
                p.w_dist * d / 100.0 + p.w_hp * hp
                    - if self.last_target == Some(m.id) {
                        p.w_sticky
                    } else {
                        0.0
                    }
                    - if facing { p.w_facing } else { 0.0 }
            };
            cands.sort_by(|a, b| score(a).total_cmp(&score(b)));
        }
        let mut checks = 0.0;
        for (d, m) in cands {
            if d > p.fire_range {
                continue;
            }
            if (self.final_ || self.early)
                && !att.is_some_and(|a| a.1.id == m.id)
                && !self.worth_it(c, m)
            {
                continue;
            }
            if mode == FireMode::Engaged
                && m.kind == ActorKind::Mech
                && !att.is_some_and(|a| a.1.id == m.id)
                && !matches!(self.state, St::Fight | St::Farm)
            {
                let to_me = (c.y - m.pos[1]).atan2(c.x - m.pos[0]);
                if d > p.engage_range && wrap(m.aim[0] - to_me).abs() > p.aim_at_me_rad {
                    continue;
                }
            }
            if checks >= p.max_los_checks {
                break;
            }
            checks += 1.0;
            let center = [
                m.pos[0],
                m.pos[1],
                m.pos[2] + if m.kind == ActorKind::Mech { 4.5 } else { 1.75 },
            ];
            if c.nav.los(c.muzzle, center) {
                return Some((d, m, center));
            }
        }
        None
    }

    fn gunnery(&mut self, c: &Ctx, look: f64) -> ([f64; 2], [bool; 2]) {
        let (aim, fire, on, target) = self.aim_and_fire(c, look);
        self.last_on = on;
        self.last_target = target;
        (aim, fire)
    }

    /// Torso aim and triggers, plus the id the gun was on target at and the id aimed at.
    fn aim_and_fire(&self, c: &Ctx, look: f64) -> ([f64; 2], [bool; 2], Option<u32>, Option<u32>) {
        let p = &self.p;
        let (mut on_id, mut target_id) = (None, None);
        let (mut aim_yaw, mut aim_pitch) = (look, 0.0);
        let mut fire = [false, false];
        let weapons = &c.me.weapons;
        let slot_of = |kind: WeaponKind| {
            weapons
                .iter()
                .rposition(|w| w.as_ref().is_some_and(|w| w.kind() == kind))
        };
        let laser_slot = weapons
            .iter()
            .position(|w| w.as_ref().is_some_and(|w| w.kind() == WeaponKind::Laser));
        if let Some((d, m, center)) = self.pick_target(c) {
            target_id = Some(m.id);
            let [mx, my, mz] = c.muzzle;
            if let Some(ms) = slot_of(WeaponKind::Mortar) {
                if let Some(WeaponObs::Mortar { cooldown, stats }) = &weapons[ms] {
                    let ready =
                        *cooldown <= 0.05 && p.mortar_min_range < d && d < p.mortar_max_range;
                    if let Some((yaw, pitch)) = ready
                        .then(|| self.mortar_solution(c, stats.speed, stats.gravity, m))
                        .flatten()
                    {
                        if aimed(c, yaw, pitch, 0.03) {
                            fire[ms] = true;
                        }
                        return ([yaw, pitch], fire, None, target_id);
                    }
                }
            }
            let gun = slot_of(WeaponKind::Gun);
            let speed = match gun.and_then(|g| weapons[g].as_ref()) {
                Some(WeaponObs::Gun { stats, .. }) => stats.speed,
                _ => 252.0,
            };
            let [px, py, pz] = center;
            let [vx, vy, vz] = m.vel;
            let mut tt = d / speed;
            let (mut ax, mut ay, mut az) = (px, py, pz);
            for _ in 0..2 {
                let lead = tt + p.lead_lag_s;
                (ax, ay, az) = (
                    px + vx * lead,
                    py + vy * lead,
                    pz + vz * lead * p.lead_vz_frac,
                );
                tt = ((ax - mx).powi(2) + (ay - my).powi(2) + (az - mz).powi(2)).sqrt() / speed;
            }
            aim_yaw = (ay - my).atan2(ax - mx);
            aim_pitch = (az - mz).atan2((ax - mx).hypot(ay - my));
            let half_w = if p.fire_tol_radius_frac > 0.0 {
                p.fire_tol_radius_frac * self.cfg.mech.radius
            } else {
                2.0
            };
            let tol = p.aim_tol_min.max(half_w.atan2(d.max(1.0)));
            let on = aimed(c, aim_yaw, aim_pitch, tol);
            if on {
                on_id = Some(m.id);
            }
            let settled = p.slew_gate <= 0.5 || self.last_on == Some(m.id);
            if let Some(g) = gun {
                if let Some(WeaponObs::Gun {
                    ammo,
                    reload,
                    stats,
                    ..
                }) = &weapons[g]
                {
                    if on && settled && *ammo > 0.0 && *reload <= 0.0 && d < stats.range {
                        fire[g] = true;
                    }
                }
            }
            if let Some((lt, Some(WeaponObs::Lightning { stats, .. }))) =
                slot_of(WeaponKind::Lightning).map(|i| (i, &weapons[i]))
            {
                let direct = (py - my).atan2(px - mx);
                if d < stats.range * 0.95
                    && wrap(c.me.aim[0] - direct).abs() < stats.cone_deg.to_radians() * 0.4
                    && c.me.energy > 10.0
                {
                    fire[lt] = true;
                }
            }
            if let Some((
                ls,
                Some(WeaponObs::Laser {
                    cooldown,
                    charge,
                    stats,
                }),
            )) = laser_slot.map(|i| (i, &weapons[i]))
            {
                let direct_yaw = (py - my).atan2(px - mx);
                let direct_pitch = (pz - mz).atan2((px - mx).hypot(py - my));
                let finish = p.laser_finish_hp > 0.0
                    && m.hp.is_some_and(|hp| hp < p.laser_finish_hp)
                    && *charge >= stats.min_charge * 1.05;
                let charged = *charge >= stats.charge * 0.95 || finish;
                if *charge > 0.0 || *cooldown <= 0.0 {
                    (aim_yaw, aim_pitch) = (direct_yaw, direct_pitch);
                    let on_direct = aimed(c, direct_yaw, direct_pitch, tol);
                    fire[ls] = !(charged && on_direct);
                    if let Some(g) = gun.filter(|_| fire[ls]) {
                        fire[g] = fire[g] && on_direct;
                    }
                }
            }
        }
        let pitch = finite(aim_pitch).clamp(c.me.stats.min_pitch, c.me.stats.max_pitch);
        ([wrap(finite(aim_yaw)), pitch], fire, on_id, target_id)
    }

    fn mortar_solution(&self, c: &Ctx, v: f64, g: f64, m: &ActorObs) -> Option<(f64, f64)> {
        let (mx, my) = (c.x, c.y);
        let mz = c.z + c.me.stats.height + 0.5;
        let [tx, ty, tz] = m.pos;
        let (vx, vy) = (m.vel[0], m.vel[1]);
        let high = !c.nav.los(c.muzzle, [tx, ty, tz + 4.5]);
        let mut sol = None;
        let mut t_f = 1.0;
        for _ in 0..2 {
            let (px, py) = (tx + vx * t_f, ty + vy * t_f);
            let d = (px - mx).hypot(py - my);
            let dz = tz - mz;
            let disc = v.powf(4.0) - g * (g * d * d + 2.0 * dz * v * v);
            if disc < 0.0 || d < 1.0 {
                return None;
            }
            let root = disc.sqrt();
            let mut th = if high {
                ((v * v + root) / (g * d)).atan()
            } else {
                ((v * v - root) / (g * d)).atan()
            };
            if th > c.me.stats.max_pitch {
                th = ((v * v - root) / (g * d)).atan();
            }
            t_f = d / (v * th.cos()).max(1e-3);
            sol = Some(((py - my).atan2(px - mx), th));
        }
        sol
    }

    // ================= items =================
    fn items(&self, c: &Ctx, act: &mut Action) {
        if let Some(offer) = c.obs.offer.as_ref().filter(|o| !o.choices.is_empty()) {
            let rank = |id: &str| UPGRADE_PREF.iter().position(|u| *u == id).unwrap_or(99);
            let best = (0..offer.choices.len()).min_by_key(|&i| (rank(&offer.choices[i].id), i));
            act.pick = best.map(|i| i as u8);
        }
        let reach = self.cfg.loot.pickup_radius;
        let weapons = &c.me.weapons;
        for pk in &c.obs.pickups {
            if (pk.pos[0] - c.x).hypot(pk.pos[1] - c.y) < reach
                && (pk.pos[2] - c.z).abs() < 4.0
                && wanted_weapon(pk.kind, c.me, self.p.no_mortar > 0.5)
            {
                act.take = Some(pk.id);
                if let Some(s1) = &weapons[1] {
                    if matches!(s1, WeaponObs::Laser { charge, .. } if *charge > 0.0) {
                        act.take = None;
                        continue;
                    }
                    act.drop = Some(1);
                }
                break;
            }
        }
        if let Some(WeaponObs::Gun {
            ammo,
            reload,
            stats,
            ..
        }) = &weapons[0]
        {
            if !act.fire[0] && *reload <= 0.0 && *ammo < stats.mag * 0.5 && c.enemies.is_empty() {
                act.reload = true;
            }
        }
    }
}

/// `round(v / 20) * 20`, Python's rounding.
fn snap20(v: f64) -> f64 {
    (pround(v / 20.0) * 20) as f64
}

fn aimed(c: &Ctx, yaw: f64, pitch: f64, tol: f64) -> bool {
    let [ay, ap] = c.me.aim;
    wrap(ay - yaw).abs() < tol && (ap - pitch).abs() < tol.max(0.03)
}

/// Python's `wanted_weapon` over `weapon_pref`; `no_mortar` drops the mortar
/// from the list, as `weapon_pref = ["lightning", "laser"]` would.
fn wanted_weapon(kind: WeaponKind, me: &SelfObs, no_mortar: bool) -> bool {
    let pref = &WEAPON_PREF[..if no_mortar { 2 } else { 3 }];
    let Some(rank) = pref.iter().position(|k| *k == kind) else {
        return false;
    };
    if me.weapons.iter().flatten().any(|w| w.kind() == kind) {
        return false;
    }
    match &me.weapons[1] {
        None => true,
        Some(s1) => pref
            .iter()
            .position(|k| *k == s1.kind())
            .is_some_and(|r1| rank < r1),
    }
}

/// The spot of lowest score plus `w` per watcher that can see it, among the
/// [`CONCEAL_TESTS`] best by score alone (ties keep the earlier candidate).
fn concealed_best(
    c: &Ctx,
    mut cands: Vec<(f64, f64, f64)>,
    watchers: &[[f64; 3]],
    w: f64,
) -> Option<(f64, f64)> {
    if watchers.is_empty() {
        return None;
    }
    cands.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (mut best, mut bs) = (None, f64::INFINITY);
    for &(s, x, y) in cands.iter().take(CONCEAL_TESTS) {
        let at = [x, y, c.nav.ground(x, y) + 4.5];
        let seen = watchers.iter().filter(|e| c.nav.los(**e, at)).count();
        let s = s + w * seen as f64;
        if s < bs {
            best = Some((x, y));
            bs = s;
        }
    }
    best
}

/// The best of 16 compass headings whose first `length` meters are walkable, or (0, 0).
fn best_heading(c: &Ctx, score: impl Fn(f64, f64) -> f64, length: f64) -> (f64, f64) {
    let n = 16;
    let (mut best, mut bs) = ((0.0, 0.0), -1e18);
    for k in 0..n {
        let a = 2.0 * PI * f64::from(k) / f64::from(n);
        let (dx, dy) = (a.cos(), a.sin());
        if !c.nav.clear_ray(c.x, c.y, dx, dy, length, 2.0) {
            continue;
        }
        let s = score(dx, dy);
        if s > bs {
            best = (dx, dy);
            bs = s;
        }
    }
    best
}

/// Where a mortar shell lands, integrated at 50 ms: `(x, y, seconds)`.
fn shell_landing(nav: &NavMap, pos: [f64; 3], vel: [f64; 3], g: f64) -> Option<(f64, f64, f64)> {
    let [mut x, mut y, mut z] = pos;
    let [vx, vy, mut vz] = vel;
    let (dt, ttl) = (0.05, 7.0);
    let mut t = 0.0;
    while t < ttl {
        x += vx * dt;
        y += vy * dt;
        vz -= g * dt;
        z += vz * dt;
        t += dt;
        if vz < 0.0 {
            let top = nav.roof_top(x, y);
            if z <= nav.ground(x, y) || (top > NONE_TOP / 10.0 && z <= top && z >= top - 3.0) {
                return Some((x, y, t));
            }
        }
    }
    None
}

impl Policy for Cbot {
    fn name(&self) -> &str {
        "cbot"
    }

    fn on_hello(&mut self, hello: &Hello) {
        self.cfg = hello.config.clone();
        self.slot = hello.slot;
        self.rng = PyRandom::new(hello.seed);
        self.load_nav(&hello.map.dir);
    }

    fn act(&mut self, obs: &Obs) -> Action {
        let nav = match &self.nav {
            Some(nav) => nav.clone(),
            None => self.load_nav(""),
        };
        let c = Ctx::new(obs, &nav, &self.p);
        self.remember(&c);
        self.reflex_scan(&c);
        self.read_mode(&c);
        self.recover(&c);
        for &s in self.strategy.states() {
            if self.wants(s, &c) {
                self.state = s;
                break;
            }
        }
        self.jet_hint = false;
        let (mut mv, mut boost, mut jump, look) = self.handle(self.state, &c);
        let silent = self.silent(&c);
        if silent {
            let cap = self.p.silent_speed_frac * self.cfg.sound.step_speed
                / c.me.stats.walk_speed.max(1e-3);
            let len = mv.0.hypot(mv.1);
            if len > cap {
                mv = (mv.0 * cap / len, mv.1 * cap / len);
            }
            (boost, jump) = (false, false);
        }
        if self.jet_hint && !silent {
            if c.me.grounded && c.me.energy > 60.0 && !nav.clear_ray(c.x, c.y, mv.0, mv.1, 6.0, 2.0)
            {
                jump = true;
                boost = true;
            } else if !c.me.grounded && c.me.energy > 3.0 {
                boost = true;
            }
        }
        (mv, boost, jump) = self.unstick(&c, mv, boost, jump);
        let (aim, fire) = self.gunnery(&c, look);
        let mut act = Action::new(obs.tick);
        act.move_dir = [finite(mv.0), finite(mv.1)];
        act.boost = boost;
        act.jump = jump;
        act.aim = Some(aim);
        act.fire = fire;
        self.items(&c, &mut act);
        self.flow.pump(&nav, self.p.flow_pops.max(1.0) as u64);
        if self.debug {
            act.debug = Some(proto::DebugDraw {
                label: Some(format!(
                    "{}:{} hp{:.2}",
                    self.strategy.name(),
                    self.state_name(),
                    c.hp_frac
                )),
                ..Default::default()
            });
        }
        act
    }
}
