//! [`AimControl`]: where the torso points and which triggers are held.
//!
//! One call per decision. In order of precedence:
//! - a farm target (crate or NPC) when there is no mech to fight;
//! - the fight target: exposed-body-point aiming, gun lead per
//!   [`GunLeadMode`], a turn-rate predicted slew gate, the laser package
//!   (track, blend, release, lethal partial, hold cap, retreat release,
//!   lightning short-risk start veto), and
//!   lightning (preferring an enemy whose laser is charging);
//! - in retreat, a rear-armor torso turn toward an attacker behind us;
//! - laser peek precharge, the opening scan, then idle aim.
//!
//! Laser trigger semantics: holding `fire` charges, letting go fires once
//! `min_charge` is reached, and the sim fires on its own `hold_s` after a full
//! charge.

use super::genome::{ApexGenome, GunLeadMode};
use super::memory::BotMemory;
use super::types::{clamp, dist2d, dist3d, wrap_pi, AimDecision, ItemGoal, ItemKind, Mode, Rng64, ThreatInfo};
use proto::{GunStats, LaserStats, LightningStats, Obs, SelfObs, WeaponKind, WeaponObs};
use std::f64::consts::PI;
use world::World;

type V3 = [f64; 3];

/// Seconds the torso turns between this decision and the shot.
const STEP: f64 = 0.05;
/// Shots leave this far out from the muzzle along the aim yaw (× cos pitch).
const BARREL: f64 = 3.0;
/// Size growth per carried cube.
const SIZE_PER_CUBE: f64 = 0.05;
/// Gun shots past this need a weak target to be worth the ammo.
const GUN_LONG_RANGE: f64 = 155.0;
/// Below this target HP, long gun shots are allowed.
const GUN_LONG_HP: f64 = 300.0;
/// Latency for hitscan (laser) and direct (lightning) aim: target and muzzle
/// are both advanced this far.
const HITSCAN_LATENCY: f64 = 0.075;
/// Gun lead latency bounds; the gun cooldown is clamped into this window.
const GUN_LAT_MIN: f64 = 0.05;
const GUN_LAT_MAX: f64 = 0.075;
/// Aim blends from lead to hitscan over the charge window
/// `[START_FRAC * charge, charge - END]`.
const LASER_BLEND_START_FRAC: f64 = 0.50;
const LASER_BLEND_END: f64 = 0.12;
/// Hold limit past full charge while peeking unhit behind cover.
const LASER_PEEK_OVERHOLD: f64 = 1.0;
/// Hold limit past full charge otherwise.
const LASER_OVERHOLD: f64 = 0.35;
/// No damage taken for this long counts as unhit.
const UNHIT_S: f64 = 1.5;
/// Lightning short risk: an aimed lightning enemy this close blocks a charge start.
const SHORT_RISK_DIST: f64 = 50.0;
const SHORT_RISK_AIM: f64 = 0.6;
/// A peek precharge needs more than this HP fraction.
const PEEK_MIN_HP_FRAC: f64 = 0.50;
/// A threat seen this recently is worth holding a charge for.
const LASER_HOLD_SEEN_S: f64 = 2.5;
/// Crate and NPC aim heights above their base.
const CRATE_MID: f64 = 1.2;
const NPC_MID: f64 = 1.75;

/// The weapons we hold, by kind, with their slot.
#[derive(Clone, Debug, Default)]
struct Arms {
    gun: Option<(usize, f64, GunStats, f64, f64)>,
    laser: Option<(usize, f64, f64, LaserStats)>,
    lightning: Option<(usize, f64, LightningStats)>,
}

impl Arms {
    fn of(me: &SelfObs) -> Self {
        let mut a = Arms::default();
        for (slot, w) in me.weapons.iter().enumerate() {
            match w {
                Some(WeaponObs::Gun { cooldown, stats, ammo, reload }) if a.gun.is_none() => {
                    a.gun = Some((slot, *cooldown, stats.clone(), *ammo, *reload));
                }
                Some(WeaponObs::Laser { cooldown, charge, stats }) if a.laser.is_none() => {
                    a.laser = Some((slot, *cooldown, *charge, stats.clone()));
                }
                Some(WeaponObs::Lightning { cooldown, stats }) if a.lightning.is_none() => {
                    a.lightning = Some((slot, *cooldown, stats.clone()));
                }
                _ => {}
            }
        }
        a
    }

    fn laser_charge(&self) -> f64 {
        self.laser.as_ref().map_or(0.0, |l| l.2)
    }
}

/// Unscaled mech body size `(radius, height)`, recovered from our own stats.
fn base_body(me: &SelfObs) -> (f64, f64) {
    let s = 1.0 + SIZE_PER_CUBE * f64::from(me.cubes);
    (me.stats.radius / s, me.stats.height / s)
}

fn muzzle_of(me: &SelfObs) -> V3 {
    [me.pos[0], me.pos[1], me.pos[2] + me.stats.muzzle_height]
}

/// The muzzle advanced along our planar velocity by `lat` seconds.
fn muzzle_at(me: &SelfObs, lat: f64) -> V3 {
    [me.pos[0] + me.vel[0] * lat, me.pos[1] + me.vel[1] * lat, me.pos[2] + me.stats.muzzle_height]
}

/// Where a shot along `aim` starts.
fn barrel_of(muzzle: V3, aim: [f64; 2]) -> V3 {
    let reach = BARREL * aim[1].cos();
    [muzzle[0] + reach * aim[0].cos(), muzzle[1] + reach * aim[0].sin(), muzzle[2]]
}

/// `[yaw, pitch]` that sends a shot from the barrel through `pt`.
fn aim_at(muzzle: V3, pt: V3, min_pitch: f64, max_pitch: f64) -> [f64; 2] {
    let solve = |o: V3| {
        let (dx, dy, dz) = (pt[0] - o[0], pt[1] - o[1], pt[2] - o[2]);
        [dy.atan2(dx), dz.atan2(dx.hypot(dy))]
    };
    let mut a = solve(muzzle);
    if dist2d(muzzle, pt) > BARREL + 1.0 {
        for _ in 0..2 {
            a = solve(barrel_of(muzzle, a));
        }
    }
    [a[0], clamp(a[1], min_pitch, max_pitch)]
}

/// The aim after one step of torso turning toward `want`.
fn predict_aim(cur: [f64; 2], want: [f64; 2], yaw_rate: f64, pitch_rate: f64) -> [f64; 2] {
    let (ys, ps) = (yaw_rate * STEP, pitch_rate * STEP);
    let dy = clamp(wrap_pi(want[0] - cur[0]), -ys, ys);
    let dp = clamp(want[1] - cur[1], -ps, ps);
    [wrap_pi(cur[0] + dy), cur[1] + dp]
}

/// `(yaw_err, pitch_err)` between two aims.
fn aim_err(a: [f64; 2], b: [f64; 2]) -> (f64, f64) {
    (wrap_pi(a[0] - b[0]).abs(), (a[1] - b[1]).abs())
}

/// Slew-gate tolerances `(yaw, pitch)` for a body of `scale` at `d` meters.
fn gate_tol(r: f64, h: f64, scale: f64, d: f64) -> (f64, f64) {
    let d = d.max(1.0);
    let mut yaw = (0.75 * r * scale).max(0.45).atan2(d);
    let mut pitch = (0.40 * h * scale).max(0.55).atan2(d);
    if d < 25.0 {
        yaw *= 1.5;
        pitch *= 1.5;
    }
    (yaw, pitch)
}

/// The first exposed point on a mech standing at `pos`, probed center, head,
/// low, then both shoulders. Returns the center and `false` when none is clear.
fn exposed_point(world: &World, muzzle: V3, pos: V3, r: f64, h: f64, cubes: u32) -> (V3, bool) {
    let scale = 1.0 + SIZE_PER_CUBE * f64::from(cubes);
    let (dx, dy) = (pos[0] - muzzle[0], pos[1] - muzzle[1]);
    let n = dx.hypot(dy).max(1e-9);
    let perp = [-dy / n, dx / n];
    let side = 0.75 * r * scale;
    let probes = [(0.0, 0.5 * h), (0.0, 0.85 * h), (0.0, 0.20 * h), (side, 0.55 * h), (-side, 0.55 * h)];
    for (off, z) in probes {
        let pt = [pos[0] + perp[0] * off, pos[1] + perp[1] * off, pos[2] + z];
        if world.los(muzzle, pt) {
            return (pt, true);
        }
    }
    ([pos[0], pos[1], pos[2] + 0.5 * h], false)
}

/// Where to aim a bullet of `speed` at a body point `c` moving with `v`
/// (and `acc`), scaled by `km`, with `lat` seconds of latency. Four
/// fixed-point iterations, flight time measured from the barrel tip. An
/// airborne target's vertical velocity counts half (jumps arc back).
#[allow(clippy::too_many_arguments)]
fn lead_point(muzzle: V3, c: V3, v: V3, acc: V3, km: f64, speed: f64, airborne: bool, lat: f64) -> V3 {
    let speed = speed.max(1.0);
    let vz = if airborne { 0.5 * clamp(v[2], -15.0, 10.0) } else { v[2] };
    let v = [v[0], v[1], vz];
    let mut p = c;
    for _ in 0..4 {
        let a = [yaw_to(muzzle, p), (p[2] - muzzle[2]).atan2(dist2d(muzzle, p))];
        let tt = (dist3d(barrel_of(muzzle, a), p) / speed).max(0.0) + lat;
        let k = |i: usize| (v[i] * tt + 0.5 * acc[i] * tt * tt) * km;
        p = [c[0] + k(0), c[1] + k(1), c[2] + clamp(k(2), -8.0, 8.0)];
    }
    p
}

/// Blend two aims, `t` in 0..=1, along the short way round in yaw.
fn blend_aim(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    let t = clamp(t, 0.0, 1.0);
    [wrap_pi(a[0] + wrap_pi(b[0] - a[0]) * t), a[1] + (b[1] - a[1]) * t]
}

/// Yaw from `a` to `b`.
fn yaw_to(a: V3, b: V3) -> f64 {
    (b[1] - a[1]).atan2(b[0] - a[0])
}

/// One target's firing solution.
struct Solution {
    /// Gun-lead aim.
    lead: [f64; 2],
    /// Hitscan aim (laser, with latency).
    hitscan: [f64; 2],
    /// Direct aim at the body point (lightning).
    direct: [f64; 2],
    /// A body point is in the clear from the muzzle.
    los: bool,
    /// 3-D distance from the muzzle to the body point.
    dist: f64,
    tol: (f64, f64),
}

/// The firing solution on `t`.
fn solve(obs: &Obs, world: &World, mem: &BotMemory, genome: &ApexGenome, t: &ThreatInfo, gun_speed: f64) -> Solution {
    let me = &obs.me;
    let muzzle = muzzle_of(me);
    let (r, h) = base_body(me);
    let scale = 1.0 + SIZE_PER_CUBE * f64::from(t.cubes);
    let (pt, los) = exposed_point(world, muzzle, t.pos, r, h * scale, t.cubes);
    let (km, acc) = match genome.gun_lead_mode {
        GunLeadMode::Standard10 => (1.0, [0.0; 3]),
        GunLeadMode::Damped082 => (0.82, [0.0; 3]),
        GunLeadMode::AccelHitfbBandit => (mem.hit_bandit().current_km(), mem.track_accel(t.id)),
    };
    let airborne = t.vel[2].abs() > 0.5;
    let gun_cd = Arms::of(me).gun.map_or(0.0, |g| g.1);
    let gun_lat = gun_cd.clamp(GUN_LAT_MIN, GUN_LAT_MAX);
    let gun_muzzle = muzzle_at(me, gun_lat);
    let lp = lead_point(gun_muzzle, pt, t.vel, acc, km, gun_speed, airborne, gun_lat);
    let hs_muzzle = muzzle_at(me, HITSCAN_LATENCY);
    let hp = [
        pt[0] + t.vel[0] * HITSCAN_LATENCY,
        pt[1] + t.vel[1] * HITSCAN_LATENCY,
        pt[2] + t.vel[2] * HITSCAN_LATENCY,
    ];
    let (lo, hi) = (me.stats.min_pitch, me.stats.max_pitch);
    let dist = dist3d(muzzle, pt);
    Solution {
        lead: aim_at(gun_muzzle, lp, lo, hi),
        hitscan: aim_at(hs_muzzle, hp, lo, hi),
        direct: aim_at(hs_muzzle, hp, lo, hi),
        los,
        dist,
        tol: gate_tol(r, h, scale, dist),
    }
}

pub struct AimControl {
    /// The laser charge was started as a peek precharge.
    peeking: bool,
}

impl Default for AimControl {
    fn default() -> Self {
        Self::new()
    }
}

impl AimControl {
    pub fn new() -> Self {
        Self { peeking: false }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decide(
        &mut self,
        obs: &Obs,
        world: &World,
        mem: &BotMemory,
        genome: &ApexGenome,
        mode: Mode,
        target: Option<&ThreatInfo>,
        farm_target: Option<&ItemGoal>,
        threats: &[ThreatInfo],
        move_dir: [f64; 2],
        _rng: &mut Rng64,
    ) -> AimDecision {
        let me = &obs.me;
        let arms = Arms::of(me);
        let charge = arms.laser_charge();
        if charge <= 0.0 {
            self.peeking = false;
        }
        let mut out = AimDecision::default();
        let visible_threat = threats.iter().any(|t| t.visible_now && !t.is_ghost);

        // Lightning shorts a charging laser: prefer such an enemy in reach.
        let zap_pref = arms.lightning.as_ref().and_then(|(_, _, ls)| {
            threats
                .iter()
                .filter(|t| t.charging && t.visible_now && t.los && !t.is_ghost && t.dist <= ls.range - 1.0)
                .min_by(|a, b| a.dist.total_cmp(&b.dist))
        });
        let fight = zap_pref.or(target.filter(|t| t.visible_now && !t.is_ghost));

        if let Some(t) = fight {
            self.engage(obs, world, mem, genome, mode, t, threats, &arms, &mut out);
        } else if let Some(f) = farm_target.filter(|f| matches!(f.kind, ItemKind::Crate | ItemKind::Npc)) {
            self.farm(obs, world, f, &arms, &mut out);
        }

        let firing = about_to_fire(&arms, &out);
        let last_dmg = mem.last_damage_t();
        if genome.rear_turn && !firing && (mode == Mode::Retreat || (obs.t - last_dmg < 3.5 && out.aim.is_none())) {
            self.rear_turn(obs, world, mem, genome, mode, threats, &arms, &mut out);
        }

        // Laser state when nothing above took charge of it.
        if let Some((slot, cd, _, ls)) = arms.laser.as_ref() {
            let decided = out.fire_target_id.is_some() || out.aim.is_some();
            let unhit = obs.t - mem.last_damage_t() >= UNHIT_S;
            if mode == Mode::Retreat && charge > 0.0 {
                // Never hold a charge while retreating: walk and boost come back now.
                out.fire[*slot] = false;
            } else if !decided {
                if charge > 0.0 {
                    // Hold through a brief sight drop while a threat was seen recently.
                    let recent = threats
                        .iter()
                        .any(|t| !t.is_ghost && obs.t - t.last_seen_t <= LASER_HOLD_SEEN_S && t.dist <= ls.range);
                    out.fire[*slot] = charge < hold_limit(ls, self.peeking, unhit) && (self.peeking || recent);
                } else if mode == Mode::Fight
                    && genome.laser_peek
                    && *cd <= 0.05
                    && me.hp / me.max_hp.max(1.0) > PEEK_MIN_HP_FRAC
                    && unhit
                    && !short_risk(me, threats)
                {
                    if let Some(t) = peek_target(obs, threats, ls) {
                        self.peeking = true;
                        out.fire[*slot] = true;
                        let (_, h) = base_body(me);
                        let pt = [t.pos[0], t.pos[1], t.pos[2] + 0.5 * h];
                        out.aim = Some(aim_at(muzzle_of(me), pt, me.stats.min_pitch, me.stats.max_pitch));
                    }
                }
            }
        }

        // Unarmed with nothing to fetch: sweep harder to spot weapon pickups.
        let unhit_scan = obs.t - last_dmg >= 4.5 && mem.unseen_hit_bearing(obs.t).is_none();
        let has_wanted_weapon_mem = mem
            .items()
            .iter()
            .any(|it| it.kind == ItemKind::Weapon && it.weapon_kind.is_some_and(|k| super::economy::weapon_rank(genome.weapon_pref, k) > 0));
        let unarmed_scan =
            me.weapons[1].is_none() && target.is_none() && farm_target.is_none() && unhit_scan && !has_wanted_weapon_mem && (obs.t < 4.0 || obs.t % 6.0 < 1.5);
        if out.aim.is_none() {
            out.aim = Some(if let Some(t) = target {
                let (_, h) = base_body(me);
                aim_at(muzzle_of(me), [t.pos[0], t.pos[1], t.pos[2] + 0.5 * h], me.stats.min_pitch, me.stats.max_pitch)
            } else if (obs.t < 2.5 && unhit_scan) || unarmed_scan {
                [wrap_pi(me.aim[0] + 1.5), 0.0]
            } else {
                idle_aim(obs, mem, threats, move_dir)
            });
        }

        // Top-off reload.
        if let Some((_, _, gs, ammo, reload)) = arms.gun.as_ref() {
            let laser_active = charge > 0.0;
            let duel = fight.is_some();
            let calm = !visible_threat || obs.t - mem.last_damage_t() > 3.0;
            let empty = *ammo <= 0.0 && !laser_active;
            let top_off = !duel && calm && *ammo < 0.75 * gs.mag;
            out.reload = *reload <= 0.0 && (empty || top_off);
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn engage(
        &mut self,
        obs: &Obs,
        world: &World,
        mem: &BotMemory,
        genome: &ApexGenome,
        mode: Mode,
        t: &ThreatInfo,
        threats: &[ThreatInfo],
        arms: &Arms,
        out: &mut AimDecision,
    ) {
        let me = &obs.me;
        let muzzle = muzzle_of(me);
        let gun_speed = arms.gun.as_ref().map_or(GunStats::default().speed, |g| g.2.speed);
        let sol = solve(obs, world, mem, genome, t, gun_speed);
        let charge = arms.laser_charge();
        let gun_in_range = |gs: &GunStats| sol.dist <= 0.95 * gs.range && (sol.dist <= GUN_LONG_RANGE || t.hp < GUN_LONG_HP);
        let gun_ok = arms
            .gun
            .as_ref()
            .is_some_and(|(_, _, gs, ammo, reload)| *ammo > 0.0 && *reload <= 0.0 && gun_in_range(gs));

        // Desired aim: laser tracking/blend > lightning-only > gun lead.
        let mut yaw_rate = me.stats.turn_rate;
        let want = if let Some((_, _, c, ls)) = arms.laser.as_ref().filter(|l| l.2 > 0.0) {
            yaw_rate *= ls.charge_turn_mult;
            if gun_ok {
                let start = LASER_BLEND_START_FRAC * ls.charge;
                let end = ls.charge - LASER_BLEND_END;
                blend_aim(sol.lead, sol.hitscan, (c - start) / (end - start).max(1e-3))
            } else {
                sol.hitscan
            }
        } else if arms.gun.is_none() && arms.lightning.is_some() && arms.laser.is_none() {
            sol.direct
        } else {
            sol.lead
        };
        out.aim = Some(want);
        out.fire_target_id = Some(t.id);
        let pred = predict_aim(me.aim, want, yaw_rate, me.stats.pitch_rate);
        let (ye, _pe) = aim_err(want, pred);
        let barrel_clear = world.los(muzzle, barrel_of(muzzle, pred));

        // Gun.
        if let Some((slot, cd, gs, ammo, reload)) = arms.gun.as_ref() {
            let in_range = gun_in_range(gs);
            let (ge, gp) = aim_err(sol.lead, pred);
            let near_now = wrap_pi(me.aim[0] - sol.lead[0]).abs() <= 0.30;
            let gun_gated = ge < sol.tol.0 && gp < sol.tol.1 && near_now;
            if *ammo > 0.0 && *reload <= 0.0 && *cd <= 2.0 * STEP && in_range && sol.los && barrel_clear && gun_gated {
                out.fire[*slot] = true;
            }
        }

        // Lightning.
        if let Some((slot, cd, ls)) = arms.lightning.as_ref() {
            let reserve = if mode == Mode::Retreat { 20.0 } else { 4.0 };
            let half_cone = ls.cone_deg.to_radians();
            let (de, _) = aim_err(sol.direct, pred);
            if sol.dist <= ls.range - 1.0 && de <= 0.45 * half_cone && me.energy >= ls.energy + reserve && *cd <= 2.0 * STEP && sol.los {
                out.fire[*slot] = true;
            }
        }

        // Laser.
        if let Some((slot, cd, _, ls)) = arms.laser.as_ref() {
            if charge > 0.0 {
                let was_peeking = self.peeking;
                self.peeking = false;
                if mode == Mode::Retreat {
                    // Never hold while retreating: let go now (fires at or past
                    // `min_charge`, cancels below) so walk speed and boost return.
                    out.fire[*slot] = false;
                } else {
                    let (le, lp) = aim_err(sol.hitscan, pred);
                    let aligned = le < 0.9 * sol.tol.0 && lp < sol.tol.1 && sol.los && barrel_clear;
                    let full = charge >= ls.charge - 0.055;
                    let lethal = genome.laser_lethal_k > 0.0 && charge >= ls.min_charge && genome.laser_lethal_k * laser_damage(me, ls, charge, t) >= t.hp;
                    let unhit = obs.t - mem.last_damage_t() >= UNHIT_S;
                    let cap = charge >= hold_limit(ls, was_peeking, unhit);
                    let release = cap || (aligned && (full || lethal));
                    // Briefly occluded: keep the charge rather than fire into cover.
                    let hold = !sol.los && !cap;
                    out.fire[*slot] = hold || !release;
                }
            } else if *cd <= 0.10 && mode != Mode::Retreat && sol.los && sol.dist <= ls.range && ye < 0.6 && !short_risk(me, threats) {
                out.fire[*slot] = true;
            }
        }
    }

    fn farm(&mut self, obs: &Obs, world: &World, f: &ItemGoal, arms: &Arms, out: &mut AimDecision) {
        let me = &obs.me;
        let muzzle = muzzle_of(me);
        let (base, vel, mid, r, h) = match f.kind {
            ItemKind::Npc => {
                let a = obs.mechs.iter().find(|m| m.id == f.id);
                (a.map_or(f.pos, |a| a.pos), a.map_or([0.0; 3], |a| a.vel), NPC_MID, 1.8, 3.5)
            }
            _ => (f.pos, [0.0; 3], CRATE_MID, 1.4, 2.4),
        };
        let c = [base[0], base[1], base[2] + mid];
        let gun_speed = arms.gun.as_ref().map_or(GunStats::default().speed, |g| g.2.speed);
        let lp = lead_point(muzzle, c, vel, [0.0; 3], 1.0, gun_speed, false, GUN_LAT_MIN);
        let (lo, hi) = (me.stats.min_pitch, me.stats.max_pitch);
        let want = aim_at(muzzle, lp, lo, hi);
        out.aim = Some(want);
        out.fire_target_id = Some(f.id);
        let pred = predict_aim(me.aim, want, me.stats.turn_rate, me.stats.pitch_rate);
        let (ye, pe) = aim_err(want, pred);
        let d = dist3d(muzzle, c);
        let tol = gate_tol(r, h, 1.0, d);
        let los = world.los(muzzle, c) && world.los(muzzle, barrel_of(muzzle, pred));
        if !(los && ye < tol.0 && pe < tol.1) {
            return;
        }
        if let Some((slot, cd, ls)) = arms.lightning.as_ref() {
            if d <= ls.range - 1.0 && me.energy > 35.0 && *cd <= 2.0 * STEP {
                out.fire[*slot] = true;
                return;
            }
        }
        if let Some((slot, cd, gs, ammo, reload)) = arms.gun.as_ref() {
            if *ammo > 0.0 && *reload <= 0.0 && *cd <= 2.0 * STEP && d <= 0.95 * gs.range {
                out.fire[*slot] = true;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rear_turn(
        &mut self,
        obs: &Obs,
        world: &World,
        mem: &BotMemory,
        genome: &ApexGenome,
        mode: Mode,
        threats: &[ThreatInfo],
        arms: &Arms,
        out: &mut AimDecision,
    ) {
        let me = &obs.me;
        let half_arc = 0.5 * me.stats.front_arc_deg.to_radians() - 0.35;
        let behind = |b: f64| wrap_pi(b - me.aim[0]).abs() > half_arc;
        let attacker = threats
            .iter()
            .filter(|t| obs.t - t.last_hit_us_t < 3.5)
            .min_by(|a, b| a.dist.total_cmp(&b.dist));
        if let Some(t) = attacker {
            let b = yaw_to(me.pos, t.pos);
            if mode == Mode::Retreat && t.visible_now && !t.is_ghost && behind(b) {
                let mut sub = AimDecision::default();
                // Keep any laser charge we already hold as it was.
                self.engage(obs, world, mem, genome, mode, t, threats, arms, &mut sub);
                *out = sub;
            } else if out.aim.is_none() || behind(b) {
                out.aim = Some([b, 0.0]);
            }
            return;
        }
        if let Some((b, _)) = mem.unseen_hit_bearing(obs.t) {
            if out.aim.is_none() || behind(b) {
                out.aim = Some([b, 0.0]);
            }
            return;
        }
        if let Some(t) = threats
            .iter()
            .filter(|t| !t.is_ghost && (t.visible_now || obs.t - t.last_seen_t < 3.5))
            .min_by(|a, b| a.dist.total_cmp(&b.dist))
        {
            let b = yaw_to(me.pos, t.pos);
            if out.aim.is_none() || behind(b) {
                out.aim = Some([b, 0.0]);
            }
        }
    }
}

/// A shot leaves this decision: gun or lightning held, or a charged laser let go.
fn about_to_fire(arms: &Arms, out: &AimDecision) -> bool {
    let laser_slot = arms.laser.as_ref().map(|l| l.0);
    let release = arms.laser_charge() > 0.0 && laser_slot.is_some_and(|ls| !out.fire[ls]) && out.fire_target_id.is_some();
    let other = out.fire.iter().enumerate().any(|(i, &f)| f && Some(i) != laser_slot);
    release || other
}

/// Laser damage at `charge` against `t`.
fn laser_damage(me: &SelfObs, ls: &LaserStats, charge: f64, t: &ThreatInfo) -> f64 {
    let frac = ls.min_frac + (1.0 - ls.min_frac) * (charge / ls.charge.max(1e-9)).min(1.0);
    let cubes = 1.0 + 0.04 * f64::from(me.cubes.min(10));
    let cook = if t.weapons.contains(&Some(WeaponKind::Mortar)) {
        1.0 + ls.cookoff
    } else {
        1.0
    };
    let toward_me = yaw_to(t.pos, me.pos);
    let away = if wrap_pi(t.aim[0] - toward_me).abs() > 0.5 * PI { 1.5 } else { 1.0 };
    ls.damage * frac * cubes * cook * away
}

/// The highest charge a laser may be held to: longer only when peeking
/// behind cover unhit.
fn hold_limit(ls: &LaserStats, peeking: bool, unhit: bool) -> f64 {
    ls.charge + if peeking && unhit { LASER_PEEK_OVERHOLD } else { LASER_OVERHOLD }
}

/// A visible lightning enemy close by and aimed at us would short a charge.
fn short_risk(me: &SelfObs, threats: &[ThreatInfo]) -> bool {
    threats.iter().any(|th| {
        th.visible_now
            && !th.is_ghost
            && th.weapons.contains(&Some(WeaponKind::Lightning))
            && th.dist < SHORT_RISK_DIST
            && wrap_pi(th.aim[0] - yaw_to(th.pos, me.pos)).abs() < SHORT_RISK_AIM
    })
}

/// The nearest threat worth a laser peek: in reach, seen in the last 1.5 s, out of sight now.
fn peek_target<'a>(obs: &Obs, threats: &'a [ThreatInfo], ls: &LaserStats) -> Option<&'a ThreatInfo> {
    threats
        .iter()
        .filter(|t| !t.is_ghost && !t.los && t.dist <= ls.range - 20.0 && obs.t - t.last_seen_t < UNHIT_S)
        .min_by(|a, b| a.dist.total_cmp(&b.dist))
}

/// Idle look direction: unseen damage, then sound, then the last seen threat, then a sweep.
fn idle_aim(obs: &Obs, mem: &BotMemory, threats: &[ThreatInfo], move_dir: [f64; 2]) -> [f64; 2] {
    let me = &obs.me;
    if let Some((b, _)) = mem.unseen_hit_bearing(obs.t) {
        return [b, 0.0];
    }
    if let Some((b, _, _)) = mem.best_sound(obs.t) {
        return [b, 0.0];
    }
    if let Some(t) = threats.iter().filter(|t| !t.is_ghost).max_by(|a, b| a.last_seen_t.total_cmp(&b.last_seen_t)) {
        return [yaw_to(me.pos, t.pos), 0.0];
    }
    let base = if move_dir[0].hypot(move_dir[1]) > 1e-6 {
        move_dir[1].atan2(move_dir[0])
    } else {
        me.aim[0]
    };
    [wrap_pi(base + 1.1 * (1.3 * obs.t).sin()), 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::{MechStats, WeaponKind};
    use world::{Prism, PrismKind};

    fn me_with(weapons: [Option<WeaponObs>; 2]) -> Obs {
        let mut obs = Obs { t: 30.0, ..Default::default() };
        obs.me.stats = MechStats::default();
        obs.me.hp = 1400.0;
        obs.me.max_hp = 1400.0;
        obs.me.energy = 100.0;
        obs.me.weapons = weapons;
        obs
    }

    fn gun(ammo: f64) -> Option<WeaponObs> {
        Some(WeaponObs::Gun {
            cooldown: 0.0,
            stats: GunStats::default(),
            ammo,
            reload: 0.0,
        })
    }

    fn laser(charge: f64) -> Option<WeaponObs> {
        Some(WeaponObs::Laser {
            cooldown: 0.0,
            charge,
            stats: LaserStats::default(),
        })
    }

    fn zap() -> Option<WeaponObs> {
        Some(WeaponObs::Lightning {
            cooldown: 0.0,
            stats: LightningStats::default(),
        })
    }

    fn threat(pos: V3, vel: V3) -> ThreatInfo {
        ThreatInfo {
            id: 7,
            pos,
            vel,
            aim: [PI, 0.0],
            hp: 1400.0,
            max_hp: 1400.0,
            cubes: 0,
            weapons: [Some(WeaponKind::Gun), None],
            charging: false,
            dist: dist2d([0.0; 3], pos),
            los: true,
            visible_now: true,
            last_seen_t: 30.0,
            last_hit_us_t: f64::NEG_INFINITY,
            last_beam_t: f64::NEG_INFINITY,
            is_ghost: false,
            is_aware: true,
            threat_dps: 100.0,
            target_score: 0.0,
        }
    }

    fn genome(mode: GunLeadMode) -> ApexGenome {
        let mut g = ApexGenome::practice_default();
        g.gun_lead_mode = mode;
        g
    }

    fn run(obs: &Obs, g: &ApexGenome, mode: Mode, t: Option<&ThreatInfo>, threats: &[ThreatInfo]) -> AimDecision {
        let world = World::flat(1000.0);
        let mem = BotMemory::new(0, &proto::Config::default());
        let mut rng = Rng64::new(1, 0);
        AimControl::new().decide(obs, &world, &mem, g, mode, t, None, threats, [1.0, 0.0], &mut rng)
    }

    #[test]
    fn exposed_point_prefers_center_then_head() {
        let world = World::flat(500.0);
        let (p, ok) = exposed_point(&world, [0.0, 0.0, 7.0], [50.0, 0.0, 0.0], 2.5, 9.0, 0);
        assert!(ok);
        assert!((p[2] - 4.5).abs() < 1e-9);
        // A low wall hides the center and low points but not the head.
        let mut w = World::flat(500.0);
        w.prisms.push(Prism::rect(PrismKind::Building, [24.0, -10.0], [26.0, 10.0], 0.0, 6.0));
        let (p, ok) = exposed_point(&w, [0.0, 0.0, 7.0], [50.0, 0.0, 0.0], 2.5, 9.0, 0);
        assert!(ok);
        assert!((p[2] - 0.85 * 9.0).abs() < 1e-9);
    }

    #[test]
    fn damped_lead_leads_less_than_standard() {
        let obs = me_with([gun(12.0), None]);
        let t = threat([100.0, 0.0, 0.0], [0.0, 10.0, 0.0]);
        let a = run(&obs, &genome(GunLeadMode::Standard10), Mode::Fight, Some(&t), &[t]);
        let b = run(&obs, &genome(GunLeadMode::Damped082), Mode::Fight, Some(&t), &[t]);
        let (ya, yb) = (a.aim.unwrap()[0], b.aim.unwrap()[0]);
        assert!(ya > yb && yb > 0.0, "{ya} {yb}");
        // Ratio of leads is about 0.82.
        assert!((yb / ya - 0.82).abs() < 0.03);
    }

    #[test]
    fn gun_fires_when_aligned_and_holds_when_not() {
        let obs = me_with([gun(12.0), None]);
        let t = threat([60.0, 0.0, 0.0], [0.0; 3]);
        let d = run(&obs, &genome(GunLeadMode::Standard10), Mode::Fight, Some(&t), &[t]);
        assert!(d.fire[0]);
        assert_eq!(d.fire_target_id, Some(7));
        let side = threat([0.0, 60.0, 0.0], [0.0; 3]);
        let d = run(&obs, &genome(GunLeadMode::Standard10), Mode::Fight, Some(&side), &[side]);
        assert!(!d.fire[0]);
        assert!((d.aim.unwrap()[0] - PI / 2.0).abs() < 0.05);
    }

    #[test]
    fn gun_range_discipline() {
        let obs = me_with([gun(12.0), None]);
        let far = threat([180.0, 0.0, 0.0], [0.0; 3]);
        assert!(!run(&obs, &genome(GunLeadMode::Standard10), Mode::Fight, Some(&far), &[far]).fire[0]);
        let mut weak = far;
        weak.hp = 200.0;
        assert!(run(&obs, &genome(GunLeadMode::Standard10), Mode::Fight, Some(&weak), &[weak]).fire[0]);
    }

    #[test]
    fn laser_charges_then_releases_at_full_charge() {
        let g = genome(GunLeadMode::Standard10);
        let t = threat([80.0, 0.0, 0.0], [0.0; 3]);
        let mut g2 = g.clone();
        g2.laser_lethal_k = 0.0;
        assert!(run(&me_with([None, laser(0.0)]), &g2, Mode::Fight, Some(&t), &[t]).fire[1], "starts charging");
        assert!(run(&me_with([None, laser(0.3)]), &g2, Mode::Fight, Some(&t), &[t]).fire[1], "keeps charging");
        assert!(!run(&me_with([None, laser(0.70)]), &g2, Mode::Fight, Some(&t), &[t]).fire[1], "releases");
        // Hold cap releases even when misaligned.
        let side = threat([0.0, 80.0, 0.0], [0.0; 3]);
        assert!(run(&me_with([None, laser(0.70)]), &g2, Mode::Fight, Some(&side), &[side]).fire[1]);
        assert!(!run(&me_with([None, laser(2.6)]), &g2, Mode::Fight, Some(&side), &[side]).fire[1]);
    }

    #[test]
    fn laser_lethal_partial_release() {
        let mut g = genome(GunLeadMode::Standard10);
        g.laser_lethal_k = 1.0;
        let mut t = threat([80.0, 0.0, 0.0], [0.0; 3]);
        t.hp = 150.0;
        assert!(!run(&me_with([None, laser(0.3)]), &g, Mode::Fight, Some(&t), &[t]).fire[1]);
        t.hp = 1000.0;
        assert!(run(&me_with([None, laser(0.3)]), &g, Mode::Fight, Some(&t), &[t]).fire[1]);
        // Facing away and a mortar on board raise the estimate.
        let me = me_with([None, None]).me;
        let ls = LaserStats::default();
        let mut back = threat([80.0, 0.0, 0.0], [0.0; 3]);
        back.aim = [0.0, 0.0];
        back.weapons = [Some(WeaponKind::Mortar), None];
        let full = laser_damage(&me, &ls, 0.75, &back);
        assert!((full - 315.0 * 1.6 * 1.5).abs() < 1e-6);
    }

    #[test]
    fn retreat_dump_releases_laser() {
        let mut g = genome(GunLeadMode::Standard10);
        g.laser_lethal_k = 0.0;
        g.laser_retreat_dump = true;
        let t = threat([80.0, 0.0, 0.0], [0.0; 3]);
        assert!(!run(&me_with([None, laser(0.3)]), &g, Mode::Retreat, Some(&t), &[t]).fire[1]);
        g.laser_retreat_dump = false;
        assert!(!run(&me_with([None, laser(0.3)]), &g, Mode::Retreat, Some(&t), &[t]).fire[1]);
    }

    #[test]
    fn retreat_never_holds_laser_without_dump() {
        let mut g = genome(GunLeadMode::Standard10);
        g.laser_lethal_k = 0.0;
        g.laser_retreat_dump = false;
        g.rear_turn = false;
        let side = threat([0.0, 80.0, 0.0], [0.0; 3]);
        for c in [0.05, 0.3, 0.7, 1.0] {
            let d = run(&me_with([None, laser(c)]), &g, Mode::Retreat, Some(&side), &[side]);
            assert!(!d.fire[1], "engaged retreat holds at charge {c}");
        }
        // Fallback path (target out of sight) too.
        let mut gone = side;
        gone.visible_now = false;
        gone.los = false;
        gone.last_seen_t = 29.9;
        assert!(!run(&me_with([None, laser(0.5)]), &g, Mode::Retreat, Some(&gone), &[gone]).fire[1]);
        // And no new charge starts in retreat.
        let ahead = threat([80.0, 0.0, 0.0], [0.0; 3]);
        assert!(!run(&me_with([None, laser(0.0)]), &g, Mode::Retreat, Some(&ahead), &[ahead]).fire[1]);
    }

    #[test]
    fn tight_gun_gate_rejects_small_error_at_120m() {
        let me = me_with([gun(12.0), None]).me;
        let (r, h) = base_body(&me);
        let (yaw, pitch) = gate_tol(r, h, 1.0, 120.0);
        assert!(yaw < 0.025, "yaw tol {yaw}");
        assert!(0.012 < yaw, "yaw tol {yaw}");
        assert!(pitch < 0.035, "pitch tol {pitch}");
        // Old formula with slop would have passed 0.025.
        assert!((0.85 * r).atan2(120.0) + 0.012 > 0.025);
        // Small-body floors hold.
        let (y, p) = gate_tol(0.1, 0.1, 1.0, 120.0);
        assert!((y - 0.45f64.atan2(120.0)).abs() < 1e-12 && (p - 0.55f64.atan2(120.0)).abs() < 1e-12);
        // Gun also refuses when the current aim is >0.30 rad off, even if one
        // fast step would reach the lead.
        let mut obs = me_with([gun(12.0), None]);
        obs.me.stats.turn_rate = 10.0;
        obs.me.aim = [0.35, 0.0];
        let t = threat([60.0, 0.0, 0.0], [0.0; 3]);
        let g = genome(GunLeadMode::Standard10);
        assert!(!run(&obs, &g, Mode::Fight, Some(&t), &[t]).fire[0], "current aim too far off");
        obs.me.aim = [0.25, 0.0];
        assert!(run(&obs, &g, Mode::Fight, Some(&t), &[t]).fire[0], "within 0.30 and slew reaches");
    }

    #[test]
    fn own_velocity_advances_muzzle_for_lead() {
        let g = genome(GunLeadMode::Standard10);
        let world = World::flat(1000.0);
        let mem = BotMemory::new(0, &proto::Config::default());
        let t = threat([100.0, 0.0, 0.0], [0.0; 3]);
        let mut obs = me_with([gun(12.0), None]);
        let still = solve(&obs, &world, &mem, &g, &t, GunStats::default().speed);
        assert!(still.lead[0].abs() < 1e-9 && still.hitscan[0].abs() < 1e-9);
        obs.me.vel = [0.0, 10.0, 0.0];
        let moving = solve(&obs, &world, &mem, &g, &t, GunStats::default().speed);
        // Gun: 0.05 s (cooldown 0 clamps up) -> muzzle 0.5 m to +y, aim swings to -y.
        let want_gun = (-0.5f64).atan2(100.0);
        assert!((moving.lead[0] - want_gun).abs() < 2e-4, "{} vs {want_gun}", moving.lead[0]);
        // Hitscan: 0.075 s -> 0.75 m.
        let want_hs = (-0.75f64).atan2(100.0);
        assert!((moving.hitscan[0] - want_hs).abs() < 2e-4, "{} vs {want_hs}", moving.hitscan[0]);
    }

    #[test]
    fn lightning_short_risk_blocks_laser_start() {
        let mut g = genome(GunLeadMode::Standard10);
        g.laser_lethal_k = 0.0;
        let mut t = threat([40.0, 0.0, 0.0], [0.0; 3]);
        t.weapons = [Some(WeaponKind::Lightning), None];
        t.aim = [PI, 0.0];
        assert!(
            !run(&me_with([None, laser(0.0)]), &g, Mode::Fight, Some(&t), &[t]).fire[1],
            "aimed close zapper"
        );
        // Looking away: safe to start.
        t.aim = [0.0, 0.0];
        assert!(
            run(&me_with([None, laser(0.0)]), &g, Mode::Fight, Some(&t), &[t]).fire[1],
            "zapper looking away"
        );
        // Far enough: safe to start.
        let mut far = threat([70.0, 0.0, 0.0], [0.0; 3]);
        far.weapons = [Some(WeaponKind::Lightning), None];
        assert!(run(&me_with([None, laser(0.0)]), &g, Mode::Fight, Some(&far), &[far]).fire[1], "zapper far");
    }

    #[test]
    fn lightning_needs_range_and_energy_and_prefers_chargers() {
        let g = genome(GunLeadMode::Standard10);
        let t = threat([30.0, 0.0, 0.0], [0.0; 3]);
        assert!(run(&me_with([None, zap()]), &g, Mode::Fight, Some(&t), &[t]).fire[1]);
        let mut low = me_with([None, zap()]);
        low.me.energy = 5.0;
        assert!(!run(&low, &g, Mode::Fight, Some(&t), &[t]).fire[1]);
        let far = threat([60.0, 0.0, 0.0], [0.0; 3]);
        assert!(!run(&me_with([None, zap()]), &g, Mode::Fight, Some(&far), &[far]).fire[1]);
        let mut charger = threat([0.0, 20.0, 0.0], [0.0; 3]);
        charger.id = 9;
        charger.charging = true;
        let d = run(&me_with([None, zap()]), &g, Mode::Fight, Some(&far), &[far, charger]);
        assert_eq!(d.fire_target_id, Some(9));
    }

    #[test]
    fn reload_rules() {
        let g = genome(GunLeadMode::Standard10);
        let calm = me_with([gun(5.0), None]);
        assert!(run(&calm, &g, Mode::Explore, None, &[]).reload);
        assert!(!run(&me_with([gun(10.0), None]), &g, Mode::Explore, None, &[]).reload);
        let t = threat([60.0, 0.0, 0.0], [0.0; 3]);
        assert!(!run(&calm, &g, Mode::Fight, Some(&t), &[t]).reload, "no reload mid-duel with ammo");
        assert!(run(&me_with([gun(0.0), None]), &g, Mode::Fight, Some(&t), &[t]).reload);
    }

    #[test]
    fn opening_scan_and_idle_sweep() {
        let g = genome(GunLeadMode::Standard10);
        let mut obs = me_with([gun(12.0), None]);
        obs.t = 1.0;
        obs.me.aim = [0.2, 0.0];
        let d = run(&obs, &g, Mode::Explore, None, &[]);
        assert!((d.aim.unwrap()[0] - 1.7).abs() < 1e-9);
        obs.t = 10.0;
        let d = run(&obs, &g, Mode::Explore, None, &[]);
        assert!((d.aim.unwrap()[0] - wrap_pi(1.1 * (13.0f64).sin())).abs() < 1e-9);
    }

    #[test]
    fn farm_uses_gun_never_laser() {
        let g = genome(GunLeadMode::Standard10);
        let obs = me_with([gun(12.0), laser(0.0)]);
        let world = World::flat(500.0);
        let mem = BotMemory::new(0, &proto::Config::default());
        let crate_ = ItemGoal {
            kind: ItemKind::Crate,
            id: 3,
            pos: [40.0, 0.0, 0.0],
            score: 1.0,
            stand_off: 20.0,
        };
        let mut rng = Rng64::new(1, 0);
        let d = AimControl::new().decide(&obs, &world, &mem, &g, Mode::Farm, None, Some(&crate_), &[], [1.0, 0.0], &mut rng);
        assert!(d.fire[0]);
        assert!(!d.fire[1]);
    }

    #[test]
    fn rear_turn_faces_attacker_behind() {
        let mut g = genome(GunLeadMode::Standard10);
        g.rear_turn = true;
        let obs = me_with([gun(0.0), None]);
        let mut t = threat([-50.0, 0.0, 0.0], [0.0; 3]);
        t.visible_now = false;
        t.los = false;
        t.last_hit_us_t = 29.0;
        let d = run(&obs, &g, Mode::Retreat, None, &[t]);
        assert!((wrap_pi(d.aim.unwrap()[0] - PI)).abs() < 1e-9);
    }

    #[test]
    fn unseen_hit_suppresses_unarmed_scan_and_rear_turn_holds_bearing() {
        let mut g = genome(GunLeadMode::Standard10);
        g.rear_turn = true;
        let world = World::flat(1000.0);
        let mut mem = BotMemory::new(0, &proto::Config::default());
        let mut rng = Rng64::new(1, 0);

        // Remember a Mortar only: ignored by unarmed_scan when unhit.
        let mut mortar_obs = me_with([gun(12.0), None]);
        mortar_obs.t = 6.2;
        mortar_obs.me.aim = [0.2, 0.0];
        mortar_obs.pickups.push(proto::PickupObs {
            id: 99,
            kind: WeaponKind::Mortar,
            pos: [50.0, 0.0, 0.0],
        });
        mem.update(&mortar_obs, &world, &mut rng);
        let d_scan = AimControl::new().decide(&mortar_obs, &world, &mem, &g, Mode::Explore, None, None, &[], [1.0, 0.0], &mut rng);
        assert!(
            (d_scan.aim.unwrap()[0] - wrap_pi(0.2 + 1.5)).abs() < 1e-9,
            "mortar does not suppress unarmed_scan"
        );

        // Now take damage from an unseen attacker at bearing 0.5 rad (within 70 deg of aim 0.2).
        let mut hit_obs = me_with([gun(12.0), None]);
        hit_obs.t = 6.5;
        hit_obs.me.aim = [0.2, 0.0];
        hit_obs.events.push(proto::GameEvent::Damaged {
            amount: 50.0,
            weapon: proto::DamageKind::Gun,
            bearing: Some(0.5),
            by: Some(7),
            rear: false,
        });
        mem.update(&hit_obs, &world, &mut rng);

        // Even in Mode::Explore (not Retreat) and within 70 deg, rear_turn holds aim on 0.5
        // and unarmed_scan does NOT spin away by +1.5 rad.
        let d_hold = AimControl::new().decide(&hit_obs, &world, &mem, &g, Mode::Explore, None, None, &[], [1.0, 0.0], &mut rng);
        assert!(
            (wrap_pi(d_hold.aim.unwrap()[0] - 0.5)).abs() < 1e-9,
            "holds unseen hit bearing, got {:?}",
            d_hold.aim
        );
    }

    #[test]
    fn laser_charge_held_across_brief_target_loss() {
        let mut g = genome(GunLeadMode::Standard10);
        g.laser_lethal_k = 0.0;
        let mut t = threat([80.0, 0.0, 0.0], [0.0; 3]);
        t.visible_now = false;
        t.los = false;
        t.last_seen_t = 29.8;
        assert!(
            run(&me_with([None, laser(0.5)]), &g, Mode::Fight, Some(&t), &[t]).fire[1],
            "holds non-peek charge"
        );
        t.last_seen_t = 26.0;
        assert!(!run(&me_with([None, laser(0.5)]), &g, Mode::Fight, Some(&t), &[t]).fire[1], "stale: let go");
        t.last_seen_t = 29.8;
        assert!(!run(&me_with([None, laser(0.5)]), &g, Mode::Retreat, Some(&t), &[t]).fire[1], "retreat: let go");
    }

    #[test]
    fn laser_release_aims_pure_hitscan() {
        let mut g = genome(GunLeadMode::Standard10);
        g.laser_lethal_k = 0.0;
        let t = threat([80.0, 0.0, 0.0], [0.0, 12.0, 0.0]);
        let ls = LaserStats::default();
        let mut obs = me_with([None, laser(ls.charge - 0.055)]);
        let world = World::flat(1000.0);
        let mem = BotMemory::new(0, &proto::Config::default());
        let sol = solve(&obs, &world, &mem, &g, &t, GunStats::default().speed);
        assert!(aim_err(sol.lead, sol.hitscan).0 > 1e-3, "lead and hitscan differ");
        obs.me.aim = sol.hitscan;
        let d = run(&obs, &g, Mode::Fight, Some(&t), &[t]);
        let (ye, pe) = aim_err(d.aim.unwrap(), sol.hitscan);
        assert!(ye < 1e-12 && pe < 1e-12, "aim is pure hitscan");
        assert!(!d.fire[1], "released when aligned on hitscan");
    }

    #[test]
    fn slew_prediction_is_rate_limited() {
        let p = predict_aim([0.0, 0.0], [1.0, 1.0], 4.0, 3.0);
        assert!((p[0] - 0.2).abs() < 1e-12 && (p[1] - 0.15).abs() < 1e-12);
    }
}
