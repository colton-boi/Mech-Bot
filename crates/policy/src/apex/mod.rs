//! The apex bot: a genome-driven policy.
//!
//! Each decision picks one [`Mode`] in priority order (escape, zone, retreat,
//! drop, fight, loot/farm, zone drift, home, explore), turns it into a nav goal,
//! follows a cached path toward it, and hands the waypoint to [`MovePlanner`]
//! and the target to [`AimControl`]. Every timer runs on `obs.t`, and threats
//! are sorted by id, so a replay of the same stream gives the same actions.

pub mod aim;
pub mod economy;
pub mod genome;
pub mod memory;
pub mod move_plan;
pub mod nav;
pub mod types;

pub use genome::ApexGenome;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use proto::{Action, Config, DamageKind, GameEvent, Hello, Obs, WeaponObs};

use crate::Policy;
use aim::AimControl;
use genome::{DropContest, RetreatMode};
use memory::BotMemory;
use move_plan::MovePlanner;
use nav::{ApexNav, EscapePlan};
use types::{dist2d, dist_xy, wrap_pi, ItemGoal, ItemKind, Mode, MoveDecision, NavGoal, Rng64, ThreatInfo};

/// Replan when the goal moved this far, meters.
const REPLAN_MOVE_M: f64 = 8.0;
/// Replan at least this often, seconds.
const REPLAN_EVERY_S: f64 = 2.5;
/// Cover search result is reused this long, seconds.
const COVER_CACHE_S: f64 = 0.5;
/// How often to look for an escape while not escaping, seconds.
const ESCAPE_CHECK_S: f64 = 1.0;
/// An escape is abandoned after this long, seconds.
const ESCAPE_TIMEOUT_S: f64 = 15.0;
/// An escape walk that gets no closer to the launch spot for this long is abandoned, seconds.
const ESCAPE_STALL_S: f64 = 2.5;
/// After an abandoned escape, wait this long before looking for another, seconds.
const ESCAPE_COOLDOWN_S: f64 = 10.0;
/// Home spot re-pick interval, seconds.
const HOME_REPICK_S: f64 = 25.0;
/// A held home spot this exposed (0..1) is dropped at the next re-pick.
const HOME_POOR_EXPOSURE: f64 = 0.70;
/// A re-picked home spot must lower `home_spot_cost` by this much to replace the old one.
const HOME_MIN_COST_GAIN: f64 = 18.0;
/// Explore point re-pick interval, seconds.
const EXPLORE_REPICK_S: f64 = 20.0;
/// Stuck when less than this progress, meters ...
const STUCK_PROGRESS_M: f64 = 1.5;
/// ... over this long, seconds.
const STUCK_AFTER_S: f64 = 1.5;
/// Length of one unstuck maneuver, seconds.
const UNSTUCK_S: f64 = 0.85;
/// An item chased this long without being reached gets blacklisted, seconds.
const ITEM_GIVE_UP_S: f64 = 12.0;
/// A threat last seen this long ago is no longer a fight target, seconds.
const TARGET_FRESH_S: f64 = 2.5;
/// Window for "recently hit", seconds.
const RECENT_HIT_S: f64 = 3.0;
/// Visible NPCs this close get shot back at when there is nothing else to shoot, meters.
const NPC_RETURN_FIRE_M: f64 = 90.0;
/// A threat this close blocks the unarmed weapon rush, meters.
const RUSH_BLOCK_M: f64 = 45.0;
/// A threat below this hp is worth fighting instead of rushing a weapon.
const RUSH_WEAK_HP: f64 = 300.0;
/// While rushing a weapon, keep shooting a target in view only within this range, meters.
const RUSH_SHOOT_M: f64 = 65.0;
/// Visible threats this close make an idle bot break line of sight, meters.
const AVOID_LOS_M: f64 = 155.0;
/// Cover search radius when breaking line of sight, meters.
const AVOID_COVER_R: f64 = 52.0;
/// Match time until stealth opening ends before switching to aggressive practice play in non-FFA, seconds.
const OPENING_S: f64 = 45.0;
/// A latched retreat ends after this long without damage and with no threat near, seconds.
const RETREAT_CALM_EXIT_S: f64 = 7.0;

/// The apex policy: the FFA genome, the practice genome it plays in 1v1 duels
/// or after `OPENING_S` when `ffa_switch` is set, and, after `on_hello`, the match state.
pub struct Apex {
    ffa_genome: ApexGenome,
    /// Played in 1v1 duels or once `t >= OPENING_S` while FFA has not latched, when `ffa_genome.ffa_switch` is set.
    practice_genome: ApexGenome,
    state: Option<ApexState>,
}

/// Everything that lives for one match.
struct ApexState {
    nav: Arc<ApexNav>,
    mem: BotMemory,
    planner: MovePlanner,
    aim: AimControl,
    rng: Rng64,
    last_action: Action,
    panic_count: u32,

    mode: Mode,
    target_id: Option<u32>,

    retreat_latched: bool,
    retreat_pulse_until: f64,
    retreat_side: f64,
    retreat_side_until: f64,
    cover: Option<[f64; 3]>,
    cover_t: f64,

    path: Vec<[f64; 2]>,
    path_idx: usize,
    path_goal: Option<[f64; 2]>,
    path_t: f64,

    stuck_ref_pos: [f64; 2],
    stuck_ref_t: f64,
    unstuck_until: f64,
    unstuck_dir: [f64; 2],
    unstuck_jump_until: f64,

    escape: Option<EscapePlan>,
    escape_start_t: f64,
    escape_launched_t: Option<f64>,
    /// Earliest match time for the next `find_escape`.
    escape_check_t: f64,
    /// Closest the escape walk has come to the launch spot, meters.
    escape_best_d: f64,
    /// Match time `escape_best_d` last improved.
    escape_progress_t: f64,

    last_npc_hit_id: Option<u32>,
    last_npc_hit_t: f64,

    home: Option<[f64; 3]>,
    home_t: f64,
    explore: Option<[f64; 3]>,
    explore_t: f64,
    search_pt: Option<[f64; 3]>,
    search_t: f64,
    visited_cells: std::collections::HashSet<(i32, i32)>,

    item_id: Option<u32>,
    item_start_t: f64,
}

impl ApexState {
    fn new(nav: Arc<ApexNav>, seed: u64, slot: u32, config: &Config) -> Self {
        let mut rng = Rng64::new(seed, slot);
        let planner = MovePlanner::new(&mut rng);
        ApexState {
            nav,
            mem: BotMemory::new(slot, config),
            planner,
            aim: AimControl::new(),
            rng,
            last_action: Action::default(),
            panic_count: 0,
            mode: Mode::Explore,
            target_id: None,
            retreat_latched: false,
            retreat_pulse_until: f64::NEG_INFINITY,
            retreat_side: 1.0,
            retreat_side_until: f64::NEG_INFINITY,
            cover: None,
            cover_t: f64::NEG_INFINITY,
            path: Vec::new(),
            path_idx: 0,
            path_goal: None,
            path_t: f64::NEG_INFINITY,
            stuck_ref_pos: [0.0; 2],
            stuck_ref_t: 0.0,
            unstuck_until: f64::NEG_INFINITY,
            unstuck_dir: [0.0; 2],
            unstuck_jump_until: f64::NEG_INFINITY,
            escape: None,
            escape_start_t: 0.0,
            escape_launched_t: None,
            escape_check_t: f64::NEG_INFINITY,
            escape_best_d: f64::INFINITY,
            escape_progress_t: 0.0,
            last_npc_hit_id: None,
            last_npc_hit_t: f64::NEG_INFINITY,
            home: None,
            home_t: f64::NEG_INFINITY,
            explore: None,
            explore_t: f64::NEG_INFINITY,
            search_pt: None,
            search_t: f64::NEG_INFINITY,
            visited_cells: std::collections::HashSet::new(),
            item_id: None,
            item_start_t: 0.0,
        }
    }
}

impl Apex {
    pub fn new(genome: ApexGenome) -> Self {
        Self {
            ffa_genome: genome,
            practice_genome: ApexGenome::practice_default(),
            state: None,
        }
    }
}

impl Default for Apex {
    fn default() -> Self {
        Self::new(ApexGenome::default())
    }
}

impl Policy for Apex {
    fn name(&self) -> &str {
        "apex"
    }

    fn on_hello(&mut self, hello: &Hello) {
        let nav = ApexNav::shared(&hello.map.dir, &hello.config.mech, &hello.config.vision);
        self.state = Some(ApexState::new(nav, hello.seed, hello.slot, &hello.config));
    }

    fn act(&mut self, obs: &Obs) -> Action {
        let Some(state) = self.state.as_mut() else {
            return Action::new(obs.tick);
        };
        let (ffa, practice) = (&self.ffa_genome, &self.practice_genome);
        match catch_unwind(AssertUnwindSafe(|| step(ffa, practice, state, obs))) {
            Ok(action) => {
                state.last_action = action.clone();
                action
            }
            Err(_) => {
                state.panic_count += 1;
                state.last_action.coast(obs.tick)
            }
        }
    }
}

fn xy(p: [f64; 3]) -> [f64; 2] {
    [p[0], p[1]]
}

/// Unit vector from `a` to `b`, zero when they coincide.
fn unit_to(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let d = dx.hypot(dy);
    if d > 1e-6 {
        [dx / d, dy / d]
    } else {
        [0.0, 0.0]
    }
}

fn laser_ready(obs: &Obs) -> bool {
    obs.me
        .weapons
        .iter()
        .flatten()
        .any(|w| matches!(w, WeaponObs::Laser { cooldown, charge, .. } if *cooldown <= 0.0 || *charge > 0.0))
}

/// The zone circle to stay inside: the next one when announced, else the current.
fn safe_circle(obs: &Obs) -> ([f64; 2], f64) {
    let z = &obs.zone;
    if z.next_radius > 0.0 {
        (z.next_center, z.next_radius)
    } else if z.radius > 0.0 {
        (z.center, z.radius)
    } else {
        (xy(obs.me.pos), 60.0)
    }
}

/// FFA/stealth engage gating: whether `th` is worth starting or keeping a fight with.
/// In stealth FFA (`stealth_ffa`) a healthy bot engages for self-defense/retaliation,
/// weak prey, or clean armed engagements; full endgame unlocks at `alive <= 3 || next_r <= 40.0`.
fn engageable(th: &ThreatInfo, threats: &[ThreatInfo], obs: &Obs, genome: &ApexGenome, is_ffa: bool) -> bool {
    if th.is_ghost || !(th.visible_now || obs.t - th.last_seen_t <= TARGET_FRESH_S) {
        return false;
    }
    let next_r = obs.zone.next_radius;
    let endgame = obs.alive <= 3 || (next_r > 0.0 && next_r <= 40.0);
    if endgame {
        return true;
    }
    let me = obs.me.pos;
    let bearing_to_us = (me[1] - th.pos[1]).atan2(me[0] - th.pos[0]);
    let aim_diff = wrap_pi(th.aim[0] - bearing_to_us).abs();
    let retaliate = obs.t - th.last_hit_us_t < 4.0;
    let lone = !genome.lone_only
        || !threats
            .iter()
            .any(|o| o.id != th.id && !o.is_ghost && obs.t - o.last_seen_t < 5.0 && (dist2d(o.pos, th.pos) < 110.0 || o.dist < 110.0));
    let stealth_ffa = is_ffa || genome.quiet_walk;
    if !stealth_ffa {
        let aimed = th.charging && aim_diff < 0.25 && th.dist < 125.0;
        let weak = th.hp < 350.0 && th.dist < 130.0;
        return (th.dist <= genome.engage_radius && lone) || retaliate || th.dist < 50.0 || aimed || weak;
    }

    let hp_frac = if obs.me.max_hp > 0.0 { obs.me.hp / obs.me.max_hp } else { 1.0 };
    let unarmed = obs.me.weapons[1].is_none();

    // 1. Self-defense / retaliation (never blocked by unarmed).
    let self_defense = th.dist < 60.0 || (retaliate && th.los) || (th.charging && aim_diff < 0.25 && th.dist < 120.0);
    // 2. Weak prey.
    let weak_prey = hp_frac > 0.45 && th.hp < 350.0 && th.dist < 130.0 && th.los;
    // 3. Armed clean engagement.
    let clean_engage = if !unarmed && hp_frac > 0.55 {
        if genome.lone_only {
            (th.dist <= genome.engage_radius && lone) || (lone && th.los && th.dist < 130.0 && hp_frac > 0.75 && th.hp_frac() < 0.55 && aim_diff > 0.6)
        } else {
            let others_within_160m = threats
                .iter()
                .filter(|o| o.id != th.id && !o.is_ghost && obs.t - o.last_seen_t < 5.0 && o.dist < 160.0)
                .count();
            (th.dist <= genome.engage_radius && others_within_160m == 0) || th.dist <= 75.0
        }
    } else {
        false
    };

    self_defense || weak_prey || clean_engage
}

/// Whether to retreat this tick; updates the latch and pulse state in `s`.
fn wants_retreat(
    s: &mut ApexState,
    genome: &ApexGenome,
    obs: &Obs,
    threats: &[ThreatInfo],
    target: Option<&ThreatInfo>,
    hp_frac: f64,
    recently_hit: bool,
) -> bool {
    let t = obs.t;
    let me = &obs.me;
    let near = |r: f64| {
        threats
            .iter()
            .filter(|th| !th.is_ghost && th.dist < r && (th.visible_now || t - th.last_seen_t < RECENT_HIT_S))
            .count()
    };
    let (near_120, near_160) = (near(120.0), near(160.0));
    let near_110 = near(110.0);
    let have_laser = me.weapons.iter().flatten().any(|w| w.kind() == proto::WeaponKind::Laser);
    let unarmed = me.weapons[1].is_none();
    let unarmed_outgunned = unarmed
        && ((near_110 >= 2 && hp_frac < 0.70)
            || (!have_laser
                && hp_frac < 0.80
                && threats.iter().any(|th| {
                    !th.is_ghost && th.visible_now && th.los && th.dist > 55.0 && th.dist < 160.0 && th.weapons.contains(&Some(proto::WeaponKind::Laser))
                })));
    let outgunned = unarmed_outgunned
        || (!unarmed
            && !have_laser
            && hp_frac < 0.85
            && threats.iter().any(|th| {
                !th.is_ghost && th.visible_now && th.los && th.dist > 55.0 && th.dist < 160.0 && th.weapons.contains(&Some(proto::WeaponKind::Laser))
            }));
    let crowded = (near_120 >= 3 && hp_frac < 0.85) || outgunned;
    match genome.retreat_mode {
        RetreatMode::Latched | RetreatMode::TtlLastStand => {
            // A genome switch may leave the other mode's state behind.
            s.retreat_pulse_until = f64::NEG_INFINITY;
            let enter_hp = if near_120 >= 2 { genome.multi_retreat_hp } else { genome.retreat_hp };
            let pressured = near_120 > 0 || recently_hit;
            if (hp_frac < enter_hp && pressured) || crowded {
                s.retreat_latched = true;
            }
            let calm = t - s.mem.last_damage_t() > RETREAT_CALM_EXIT_S && near_120 == 0 && hp_frac > 0.55;
            if (hp_frac >= genome.retreat_exit_hp && !crowded) || calm {
                s.retreat_latched = false;
            }
            let mut r = s.retreat_latched;
            // Finish a nearly dead target rather than turn our back on it.
            let finish = target.is_some_and(|tg| tg.dist < 120.0 && (tg.hp < 200.0 || tg.hp < me.hp * 0.50)) && near_120 <= 2 && hp_frac > 0.20 && !outgunned;
            if r && finish {
                r = false;
            }
            if r && genome.retreat_mode == RetreatMode::TtlLastStand && ttl_last_stand(obs, threats, target, hp_frac) {
                r = false;
            }
            r
        }
        RetreatMode::Pulse15s => {
            s.retreat_latched = false;
            // The lose rule: outnumbered while hurt, crowded, or losing the duel.
            let mut losing = (near_160 >= 2 && hp_frac < genome.multi_retreat_hp) || (near_160 >= 3 && hp_frac < 0.80) || crowded;
            if let Some(tg) = target {
                let tf = tg.hp_frac();
                if hp_frac < genome.retreat_hp && tf > hp_frac + 0.10 {
                    losing = true;
                }
                if tf < 0.25 && hp_frac > 0.20 && near_160 <= 2 && !outgunned {
                    losing = false;
                }
            }
            if losing {
                s.retreat_pulse_until = t + 1.5;
            }
            t < s.retreat_pulse_until
        }
    }
}

/// `TtlLastStand`: stand and finish the target when it dies well before we do.
/// Never below 25% HP; NPCs within 75 m add 25 DPS each to what we take.
fn ttl_last_stand(obs: &Obs, threats: &[ThreatInfo], target: Option<&ThreatInfo>, hp_frac: f64) -> bool {
    if hp_frac < 0.25 {
        return false;
    }
    let Some(tg) = target.filter(|tg| tg.los && tg.visible_now) else {
        return false;
    };
    let pos = obs.me.pos;
    let npc_dps = 25.0 * obs.mechs.iter().filter(|m| !m.is_mech() && dist2d(m.pos, pos) < 75.0).count() as f64;
    let incoming: f64 = threats.iter().filter(|th| !th.is_ghost && th.is_aware).map(|th| th.threat_dps).sum::<f64>() + npc_dps;
    let ttl = if incoming > 0.0 { obs.me.hp / incoming } else { f64::INFINITY };
    let kill_t = if tg.threat_dps > 0.0 { tg.hp / tg.threat_dps } else { f64::INFINITY };
    kill_t < 0.9 * ttl
}

/// A reachable drop worth heading to under `contest`, nearest first.
fn pick_drop(obs: &Obs, nav: &ApexNav, threats: &[ThreatInfo], contest: DropContest, hp_frac: f64, max_r: f64) -> Option<NavGoal> {
    let me = obs.me.pos;
    let (zc, zr) = (obs.zone.center, obs.zone.radius);
    let max_r = if contest == DropContest::HealOnly95 { max_r.min(160.0) } else { max_r };
    let walk = obs.me.stats.walk_speed.max(1.0);
    obs.drops
        .iter()
        .filter(|d| d.holder.is_none_or(|h| h == obs.me.id))
        .filter(|d| dist2d(me, d.pos) <= max_r)
        .filter(|d| zr <= 0.0 || dist_xy(zc, xy(d.pos)) < zr)
        .filter(|d| nav.is_main_xy(d.pos[0], d.pos[1]) || nav.snap_to_main(d.pos[0], d.pos[1], d.radius.max(2.0)).is_some())
        .filter(|d| match contest {
            DropContest::Off => false,
            DropContest::HealOnly95 => {
                let eta = dist2d(me, d.pos) / walk;
                hp_frac < 0.95 && d.lands_in <= eta + 6.0 && !threats.iter().any(|t| !t.is_ghost && dist2d(t.pos, d.pos) < 70.0)
            }
            DropContest::AlwaysUncontested => !threats.iter().any(|t| !t.is_ghost && dist2d(t.pos, d.pos) < d.radius + 25.0),
        })
        .min_by(|a, b| dist2d(me, a.pos).total_cmp(&dist2d(me, b.pos)).then(a.id.cmp(&b.id)))
        .map(|d| NavGoal {
            pos: d.pos,
            radius: (d.radius * 0.6).max(1.0),
            exact: false,
        })
}

/// A goal-free direct move, for maneuvers the planner does not model.
fn direct_move(dir: [f64; 2], boost: bool, jump: bool) -> MoveDecision {
    MoveDecision { move_dir: dir, boost, jump }
}

/// Fills each live threat's `target_score`; lower is a better target. The
/// current target gets a stickiness discount.
fn score_threats(threats: &mut [ThreatInfo], t: f64, target_id: Option<u32>) {
    for th in threats.iter_mut().filter(|th| !th.is_ghost) {
        let bonus = |cond: bool, v: f64| if cond { v } else { 0.0 };
        th.target_score = th.dist + 0.12 * th.hp
            - bonus(th.los, 80.0)
            - bonus(th.visible_now, 40.0)
            - bonus(t - th.last_hit_us_t < 4.0, 70.0)
            - bonus(th.charging, 35.0)
            - bonus(th.hp_frac() < 0.35, 55.0)
            - bonus(Some(th.id) == target_id, 50.0)
            + bonus(!th.los, 180.0);
    }
}

/// The engageable threat with the lowest `target_score`, lower id on ties.
fn pick_target(threats: &[ThreatInfo], obs: &Obs, genome: &ApexGenome, is_ffa: bool) -> Option<ThreatInfo> {
    threats
        .iter()
        .filter(|th| engageable(th, threats, obs, genome, is_ffa))
        .min_by(|a, b| a.target_score.total_cmp(&b.target_score).then(a.id.cmp(&b.id)))
        .copied()
}

/// Drops the current escape and holds off looking for another for a while.
fn abandon_escape(s: &mut ApexState, t: f64) {
    s.escape = None;
    s.escape_check_t = t + ESCAPE_COOLDOWN_S;
}

/// Starts the give-up clock for a new item, blacklisting one chased too long.
fn track_item(s: &mut ApexState, id: u32, t: f64) {
    if s.item_id != Some(id) {
        s.item_id = Some(id);
        s.item_start_t = t;
    } else if t - s.item_start_t > ITEM_GIVE_UP_S {
        s.mem.blacklist_item(id, t + 30.0);
        s.item_id = None;
    }
}

/// An NPC to shoot back at: the one that hit us in the last few seconds when
/// still in view, else the nearest visible NPC in range.
fn npc_return_fire(s: &ApexState, obs: &Obs) -> Option<ItemGoal> {
    let pos = obs.me.pos;
    let npcs = || obs.mechs.iter().filter(|m| !m.is_mech());
    let attacker = s
        .last_npc_hit_id
        .filter(|_| obs.t - s.last_npc_hit_t <= RECENT_HIT_S)
        .and_then(|id| npcs().find(|m| m.id == id));
    let nearest = || {
        npcs()
            .filter(|m| dist2d(pos, m.pos) <= NPC_RETURN_FIRE_M)
            .min_by(|a, b| dist2d(pos, a.pos).total_cmp(&dist2d(pos, b.pos)).then(a.id.cmp(&b.id)))
    };
    attacker.or_else(nearest).map(|npc| ItemGoal {
        kind: ItemKind::Npc,
        id: npc.id,
        pos: npc.pos,
        score: 1.0,
        stand_off: 45.0,
    })
}

/// One decision. Free so `act` can wrap it in `catch_unwind`.
fn step(ffa_genome: &ApexGenome, practice_genome: &ApexGenome, s: &mut ApexState, obs: &Obs) -> Action {
    let t = obs.t;
    let nav = s.nav.clone();
    let world = nav.world();
    s.mem.update(obs, world, &mut s.rng);
    let is_ffa = s.mem.is_ffa();
    // 1v1 duels play practice from t=0; 16-mech lobbies play stealth FFA from t=0 until practice is confirmed.
    let genome = if !ffa_genome.ffa_switch {
        ffa_genome
    } else if s.mem.start_alive().is_some_and(|n| n <= 2) {
        practice_genome
    } else if is_ffa || t < OPENING_S {
        ffa_genome
    } else {
        practice_genome
    };

    let me = &obs.me;
    let pos = me.pos;
    let pos2 = xy(pos);
    s.visited_cells.insert(((pos[0] / 40.0).floor() as i32, (pos[1] / 40.0).floor() as i32));
    let hp_frac = if me.max_hp > 0.0 { me.hp / me.max_hp } else { 1.0 };
    let recently_hit = t - s.mem.last_damage_t() < RECENT_HIT_S;

    let mut threats = s.mem.threats(obs, world);
    threats.sort_by_key(|th| th.id);
    score_threats(&mut threats, t, s.target_id);
    let (zone_must, zone_should, safe) = economy::zone_assessment(obs, &nav);

    // ---- fight target (needed by retreat's last-stand rule as well)
    let target = pick_target(&threats, obs, genome, is_ffa);

    // ---- escape
    if s.escape.is_none() && t >= s.escape_check_t {
        s.escape_check_t = t + ESCAPE_CHECK_S;
        if let Some(plan) = nav.find_escape(pos) {
            s.escape = Some(plan);
            s.escape_start_t = t;
            s.escape_launched_t = None;
            s.escape_best_d = f64::INFINITY;
            s.escape_progress_t = t;
        }
    }
    if s.escape.is_some() && t - s.escape_start_t > ESCAPE_TIMEOUT_S {
        abandon_escape(s, t);
    }

    let mut mode;
    let mut goal: Option<NavGoal> = None;
    let mut farm_target: Option<ItemGoal> = None;
    let mut forced_move: Option<MoveDecision> = None;
    let mut fight_target: Option<ThreatInfo> = None;

    if let Some(plan) = s.escape {
        mode = Mode::Escape;
        let need = 70.0_f64.min(0.7 * me.max_energy);
        match s.escape_launched_t {
            None => {
                let d = dist_xy(pos2, plan.launch_xy);
                if d > 2.0 && me.grounded {
                    if d < s.escape_best_d - 0.5 {
                        s.escape_best_d = d;
                        s.escape_progress_t = t;
                    }
                    let area = nav.ground_area_of(pos);
                    let mut dir = if d > 4.0 && area.is_some() && area == nav.area_of(plan.launch_xy[0], plan.launch_xy[1]) {
                        let launch = NavGoal {
                            pos: [plan.launch_xy[0], plan.launch_xy[1], nav.ground_at(plan.launch_xy[0], plan.launch_xy[1])],
                            radius: 2.0,
                            exact: true,
                        };
                        unit_to(pos2, xy(follow_path(s, &nav, pos2, launch, t).pos))
                    } else {
                        unit_to(pos2, plan.launch_xy)
                    };
                    let k = (d / 4.0).min(1.0);
                    dir = [dir[0] * k, dir[1] * k];
                    forced_move = Some(direct_move(dir, false, false));
                    if t - s.escape_progress_t > ESCAPE_STALL_S {
                        abandon_escape(s, t);
                    }
                } else if me.energy >= need && me.grounded {
                    s.escape_launched_t = Some(t);
                    forced_move = Some(direct_move(unit_to(pos2, plan.land_xy), true, true));
                } else {
                    forced_move = Some(direct_move([0.0, 0.0], false, false));
                }
            }
            Some(lt) => {
                if me.grounded && t - lt > 0.4 {
                    if nav.is_main_xy(pos[0], pos[1]) || (!nav.is_main_xy(plan.land_xy[0], plan.land_xy[1]) && dist_xy(pos2, plan.land_xy) <= 6.0) {
                        s.escape = None;
                        s.escape_check_t = t + 0.1;
                    } else {
                        abandon_escape(s, t);
                    }
                }
                forced_move = Some(direct_move(unit_to(pos2, plan.land_xy), !me.grounded || t - lt <= 0.4, false));
            }
        }
        fight_target = target;
    } else {
        mode = Mode::Explore;
    }

    // ---- retreat bookkeeping (runs every tick so latches stay consistent)
    let wants_retreat = wants_retreat(s, genome, obs, &threats, target.as_ref(), hp_frac, recently_hit);

    if mode != Mode::Escape {
        if zone_must {
            mode = Mode::ZoneMust;
            goal = Some(NavGoal {
                pos: safe,
                radius: 6.0,
                exact: false,
            });
            fight_target = target;
        } else if wants_retreat {
            if t - s.cover_t >= COVER_CACHE_S {
                s.cover_t = t;
                let trees = if hp_frac < 0.65 || me.weapons[1].is_none() {
                    t >= 15.0
                } else {
                    hp_frac < 0.35 && recently_hit && t >= 40.0
                };
                s.cover = nav
                    .find_cover(pos, &threats, &obs.zone, trees, 80.0)
                    .or_else(|| nav.find_commit_cover(pos, &threats, &obs.zone, trees));
            }
            match s.cover {
                Some(c) => {
                    mode = Mode::Retreat;
                    goal = Some(NavGoal {
                        pos: c,
                        radius: 1.5,
                        exact: true,
                    });
                    fight_target = target;
                }
                None if genome.commit_no_cover => {}
                None => {
                    // No cover: back away from the threat centroid, inside the safe circle.
                    let live: Vec<&ThreatInfo> = threats.iter().filter(|th| !th.is_ghost).collect();
                    if !live.is_empty() {
                        let n = live.len() as f64;
                        let c = [
                            live.iter().map(|th| th.pos[0]).sum::<f64>() / n,
                            live.iter().map(|th| th.pos[1]).sum::<f64>() / n,
                        ];
                        let away = unit_to(c, pos2);
                        let (sc, sr) = safe_circle(obs);
                        let mut p = [pos2[0] + away[0] * 40.0, pos2[1] + away[1] * 40.0];
                        if dist_xy(p, sc) > sr * 0.9 {
                            let k = sr * 0.9 / dist_xy(p, sc).max(1e-6);
                            p = [sc[0] + (p[0] - sc[0]) * k, sc[1] + (p[1] - sc[1]) * k];
                        }
                        let p = nav.snap_to_main(p[0], p[1], 15.0).unwrap_or(p);
                        mode = Mode::Retreat;
                        goal = Some(NavGoal {
                            pos: [p[0], p[1], nav.ground_at(p[0], p[1])],
                            radius: 3.0,
                            exact: false,
                        });
                        fight_target = target;
                    }
                }
            }
        }
    }

    let undecided = mode == Mode::Explore && goal.is_none() && forced_move.is_none();
    let unarmed = me.weapons[1].is_none();
    if undecided {
        // Gun-only: a reachable weapon beats an optional fight.
        let weapon_rush = || {
            let pressed = threats
                .iter()
                .any(|th| t - th.last_hit_us_t < RECENT_HIT_S || (!th.is_ghost && (th.dist < RUSH_BLOCK_M || th.hp < RUSH_WEAK_HP)));
            if pressed {
                return None;
            }
            economy::choose_loot_or_farm(obs, &nav, &s.mem, genome, false, &threats).filter(|it| it.kind == ItemKind::Weapon)
        };
        // Outside stealth/FFA, an unarmed bot rushes a weapon ahead of an optional fight.
        let rush = if unarmed && target.is_some() && !(is_ffa || genome.quiet_walk) {
            weapon_rush()
        } else {
            None
        };
        if let Some(g) = pick_drop(obs, &nav, &threats, genome.drop_contest, hp_frac, 350.0) {
            mode = Mode::Drop;
            goal = Some(g);
            fight_target = target;
        } else if let Some(item) = rush {
            mode = Mode::Loot;
            goal = Some(NavGoal {
                pos: item.pos,
                radius: 1.0,
                exact: true,
            });
            track_item(s, item.id, t);
            fight_target = target.filter(|tg| tg.los && tg.dist < RUSH_SHOOT_M);
        } else if let Some(tg) = target {
            fight_target = Some(tg);
            let detour =
                economy::choose_loot_or_farm(obs, &nav, &s.mem, genome, true, &threats).filter(|it| matches!(it.kind, ItemKind::Cube | ItemKind::Weapon));
            if let Some(item) = detour {
                mode = if item.kind == ItemKind::Cube && t - s.mem.last_kill_t() <= 4.0 {
                    Mode::KillCube
                } else {
                    Mode::Loot
                };
                goal = Some(NavGoal {
                    pos: item.pos,
                    radius: 1.0,
                    exact: true,
                });
                track_item(s, item.id, t);
            } else {
                mode = Mode::Fight;
                goal = Some(NavGoal {
                    pos: tg.pos,
                    radius: if tg.los { (0.6 * genome.gun_range).max(10.0) } else { 3.0 },
                    exact: false,
                });
            }
        } else {
            let in_fight = recently_hit || threats.iter().any(|th| !th.is_ghost && th.visible_now && th.dist < genome.engage_radius);
            let calm_item = economy::choose_loot_or_farm(obs, &nav, &s.mem, genome, in_fight, &threats);
            let close_los_threat = threats.iter().any(|th| !th.is_ghost && th.visible_now && th.los && th.dist < 55.0);
            let is_laser_item = |it: &ItemGoal| {
                it.kind == ItemKind::Weapon
                    && (obs.pickups.iter().any(|p| p.id == it.id && p.kind == proto::WeaponKind::Laser)
                        || s.mem
                            .items()
                            .iter()
                            .any(|m| m.id == it.id && m.kind == ItemKind::Weapon && m.weapon_kind == Some(proto::WeaponKind::Laser)))
            };
            let has_laser = me.weapons.iter().flatten().any(|w| w.kind() == proto::WeaponKind::Laser);
            let urgent_grab = calm_item.filter(|it| {
                !close_los_threat
                    && ((!has_laser && is_laser_item(it) && dist2d(pos, it.pos) < 65.0)
                        || (it.kind == ItemKind::Cube && hp_frac > 0.70 && dist2d(pos, it.pos) < 30.0))
            });
            let endgame = obs.alive <= 3 || (obs.zone.next_radius > 0.0 && obs.zone.next_radius <= 40.0);
            let ffa_threat_close = (is_ffa || genome.quiet_walk) && !endgame && threats.iter().any(|th| !th.is_ghost && th.visible_now && th.dist < 145.0);
            if let Some(item) = urgent_grab {
                mode = if item.kind == ItemKind::Cube && t - s.mem.last_kill_t() <= 4.0 {
                    Mode::KillCube
                } else {
                    Mode::Loot
                };
                goal = Some(NavGoal {
                    pos: item.pos,
                    radius: item.stand_off.max(1.0),
                    exact: item.stand_off <= 0.0,
                });
                track_item(s, item.id, t);
            } else if let Some(c) = ffa_threat_close.then(|| avoid_cover(s, obs, &nav, &threats)).flatten() {
                mode = Mode::Home;
                goal = Some(NavGoal {
                    pos: c,
                    radius: 1.5,
                    exact: true,
                });
            } else if let Some(item) = calm_item {
                mode = match item.kind {
                    ItemKind::Cube if t - s.mem.last_kill_t() <= 4.0 => Mode::KillCube,
                    ItemKind::Cube | ItemKind::Weapon => Mode::Loot,
                    ItemKind::Crate | ItemKind::Npc => Mode::Farm,
                    ItemKind::Drop => Mode::Drop,
                };
                if mode == Mode::Farm {
                    farm_target = Some(item);
                }
                goal = Some(NavGoal {
                    pos: item.pos,
                    radius: item.stand_off.max(1.0),
                    exact: item.stand_off <= 0.0,
                });
                track_item(s, item.id, t);
            } else if zone_should {
                mode = Mode::ZoneShould;
                goal = Some(NavGoal {
                    pos: safe,
                    radius: 10.0,
                    exact: false,
                });
            } else {
                let stealth_idle = is_ffa || genome.quiet_walk;
                let unarmed_search = stealth_idle
                    && !has_laser
                    && t < 45.0
                    && obs.alive > 5
                    && hp_frac > 0.75
                    && !threats.iter().any(|th| !th.is_ghost && th.visible_now && th.los && th.dist < 75.0);
                if let Some(c) = stealth_idle.then(|| avoid_cover(s, obs, &nav, &threats)).flatten() {
                    mode = Mode::Home;
                    goal = Some(NavGoal {
                        pos: c,
                        radius: 1.5,
                        exact: true,
                    });
                } else if let Some(sp) = unarmed_search.then(|| search_goal(s, obs, &nav, &threats)).flatten() {
                    mode = Mode::Explore;
                    goal = Some(sp);
                } else if let Some(h) = stealth_idle.then(|| home_goal(s, obs, &nav, genome, &threats)).flatten() {
                    mode = Mode::Home;
                    goal = Some(h);
                } else {
                    mode = Mode::Explore;
                    goal = explore_goal(s, obs, &nav);
                }
            }
        }
    }
    if !matches!(mode, Mode::Loot | Mode::KillCube | Mode::Farm) {
        s.item_id = None;
    }
    for ev in &obs.events {
        if let GameEvent::Damaged {
            weapon: DamageKind::Npc,
            by: Some(id),
            ..
        } = ev
        {
            s.last_npc_hit_id = Some(*id);
            s.last_npc_hit_t = t;
        }
    }
    if fight_target.is_none() && farm_target.is_none() {
        farm_target = npc_return_fire(s, obs);
    }
    s.target_id = fight_target.map(|tg| tg.id);

    // ---- path following
    let nav_goal = goal.map(|g| follow_path(s, &nav, pos2, g, t));

    // ---- unstuck (never while escaping, holding at home, or capturing a drop)
    let base_mode = mode;
    let in_non_main = !nav.is_main_xy(pos[0], pos[1]);
    let should_make_progress = match base_mode {
        Mode::Escape | Mode::Drop => false,
        Mode::Home => goal.is_some_and(|g| dist2d(pos, g.pos) > g.radius + 1.0),
        Mode::Fight => me.grounded || t < s.unstuck_until,
        _ => goal.is_some_and(|g| (me.grounded || t < s.unstuck_until || in_non_main) && dist2d(pos, g.pos) > g.radius.max(1.0)),
    };
    if should_make_progress {
        if dist_xy(pos2, s.stuck_ref_pos) >= STUCK_PROGRESS_M {
            s.stuck_ref_pos = pos2;
            s.stuck_ref_t = t;
        } else if t - s.stuck_ref_t > STUCK_AFTER_S && t >= s.unstuck_until {
            let fallback_wp = fight_target.map_or(pos, |tg| tg.pos);
            let want = unit_to(pos2, xy(nav_goal.map_or(fallback_wp, |w| w.pos)));
            let side = if s.rng.chance(0.5) { 1.0 } else { -1.0 };
            let norm2 = |v: [f64; 2]| {
                let m = v[0].hypot(v[1]);
                if m > 1e-6 {
                    [v[0] / m, v[1] / m]
                } else {
                    [1.0, 0.0]
                }
            };
            let cands = [
                norm2([-want[1] * side + want[0] * 0.3, want[0] * side + want[1] * 0.3]),
                norm2([want[1] * side + want[0] * 0.3, -want[0] * side + want[1] * 0.3]),
                norm2([-want[1] * side, want[0] * side]),
                norm2([want[1] * side, -want[0] * side]),
                norm2([-want[0], -want[1]]),
            ];
            let best_dir = cands
                .into_iter()
                .filter(|&d| !nav.step_blocked_ahead(pos, d, 2.5))
                .max_by(|&a, &b| {
                    nav.clearance_at(pos2[0] + a[0] * 4.0, pos2[1] + a[1] * 4.0)
                        .total_cmp(&nav.clearance_at(pos2[0] + b[0] * 4.0, pos2[1] + b[1] * 4.0))
                })
                .unwrap_or(cands[0]);
            s.unstuck_dir = best_dir;
            s.unstuck_jump_until = t + 0.85;
            s.unstuck_until = t + UNSTUCK_S;
            s.stuck_ref_pos = pos2;
            s.stuck_ref_t = t + UNSTUCK_S;
            s.path_goal = None;
        }
        if t < s.unstuck_until {
            mode = Mode::Unstuck;
            let jet = s.unstuck_jump_until > t && obs.me.energy > 0.15;
            let dir = if in_non_main && !me.grounded {
                let want = nav_goal.map_or(s.unstuck_dir, |w| unit_to(pos2, xy(w.pos)));
                if want != [0.0, 0.0] && !nav.step_blocked_ahead(pos, want, 2.0) {
                    want
                } else {
                    s.unstuck_dir
                }
            } else {
                s.unstuck_dir
            };
            forced_move = Some(direct_move(dir, jet, jet));
        }
    } else {
        s.stuck_ref_pos = pos2;
        s.stuck_ref_t = t;
        s.unstuck_until = f64::NEG_INFINITY;
    }
    s.mode = mode;

    // ---- movement
    let has_los = fight_target.is_some_and(|tg| tg.los);
    let chasing_laser = mode == Mode::Loot
        && s.item_id.is_some_and(|id| {
            obs.pickups.iter().any(|p| p.id == id && p.kind == proto::WeaponKind::Laser)
                || s.mem
                    .items()
                    .iter()
                    .any(|it| it.id == id && it.kind == ItemKind::Weapon && it.weapon_kind == Some(proto::WeaponKind::Laser))
        });
    let mut mv = match forced_move {
        Some(m) => m,
        None => s.planner.plan(
            obs,
            &nav,
            &s.mem,
            genome,
            mode,
            nav_goal,
            fight_target.as_ref(),
            &threats,
            has_los,
            laser_ready(obs),
            chasing_laser,
            &mut s.rng,
        ),
    };
    if mode == Mode::Retreat {
        let m = mv.move_dir[0].hypot(mv.move_dir[1]);
        if m > 0.1 && nav.step_blocked_ahead(pos, [mv.move_dir[0] / m, mv.move_dir[1] / m], 2.0) && t >= s.retreat_side_until {
            s.retreat_side = if s.rng.chance(0.5) { 1.0 } else { -1.0 };
            s.retreat_side_until = t + UNSTUCK_S;
        }
        if t < s.retreat_side_until {
            let sd = s.retreat_side;
            mv.move_dir = [-mv.move_dir[1] * sd, mv.move_dir[0] * sd];
        }
    }

    // ---- aim and economy
    let ad = s.aim.decide(
        obs,
        world,
        &s.mem,
        genome,
        mode,
        fight_target.as_ref(),
        farm_target.as_ref(),
        &threats,
        mv.move_dir,
        &mut s.rng,
    );
    let (take, drop) = economy::pick_weapon_action(obs, genome);

    let mut action = Action::new(obs.tick);
    action.move_dir = mv.move_dir;
    action.boost = mv.boost;
    action.jump = mv.jump;
    action.aim = ad.aim;
    action.fire = ad.fire;
    action.reload = ad.reload;
    action.pick = economy::pick_upgrade(obs, genome);
    action.take = take;
    action.drop = drop;
    action
}

/// Cover from visible threats with line of sight within `AVOID_LOS_M`.
/// Keeps the cached cover spot as long as it stays inside the zone and hidden
/// from every nearby threat in view, so we walk to it once and settle.
fn avoid_cover(s: &mut ApexState, obs: &Obs, nav: &ApexNav, threats: &[ThreatInfo]) -> Option<[f64; 3]> {
    let t = obs.t;
    let seer_near = threats.iter().any(|th| !th.is_ghost && th.visible_now && th.los && th.dist < AVOID_LOS_M);
    if !seer_near {
        return None;
    }
    let z = &obs.zone;
    let keep_cover = s.cover.is_some_and(|c| {
        let in_z = z.radius <= 0.0 || dist_xy(xy(c), z.center) <= z.radius - 3.0;
        let body = [c[0], c[1], c[2] + 0.5 * obs.me.stats.height];
        let eye_h = obs.me.stats.eye_height;
        in_z && threats
            .iter()
            .filter(|th| !th.is_ghost && th.visible_now && th.dist < AVOID_LOS_M)
            .all(|th| !nav.world().los([th.pos[0], th.pos[1], th.pos[2] + eye_h], body))
    });
    if !keep_cover {
        s.cover_t = t;
        s.cover = nav.find_cover(obs.me.pos, threats, z, false, AVOID_COVER_R);
    }
    s.cover
}

/// Replans when stale, then returns the string-pulled waypoint as a nav goal.
fn follow_path(s: &mut ApexState, nav: &ApexNav, pos2: [f64; 2], g: NavGoal, t: f64) -> NavGoal {
    let g2 = xy(g.pos);
    let stale = s.path_goal.is_none_or(|pg| dist_xy(pg, g2) > REPLAN_MOVE_M) || t - s.path_t >= REPLAN_EVERY_S;
    if stale {
        s.path = nav.find_path(pos2, g2).unwrap_or_default();
        s.path_idx = 0;
        s.path_goal = Some(g2);
        s.path_t = t;
    }
    if s.path.len() < 2 {
        return g;
    }
    let (idx, wp) = nav.string_pull_waypoint(pos2, &s.path, s.path_idx);
    s.path_idx = idx;
    if idx + 1 >= s.path.len() || dist_xy(wp, g2) < 0.5 || dist_xy(pos2, g2) <= g.radius {
        return g;
    }
    NavGoal {
        pos: [wp[0], wp[1], nav.ground_at(wp[0], wp[1])],
        radius: 1.5,
        exact: false,
    }
}

/// A low-exposure spot in the safe disc. The held spot is kept until it
/// leaves the safe disc/zone, or until a re-pick finds one meaningfully lower
/// cost (or the held one is poorly covered).
fn home_goal(s: &mut ApexState, obs: &Obs, nav: &ApexNav, genome: &ApexGenome, threats: &[ThreatInfo]) -> Option<NavGoal> {
    let t = obs.t;
    let pos = obs.me.pos;
    let (sc, sr) = safe_circle(obs);
    let valid = |p: [f64; 3]| {
        let p2 = xy(p);
        let in_next_zone = obs.zone.next_radius <= 0.0 || dist_xy(p2, obs.zone.next_center) <= obs.zone.next_radius - 3.0;
        let in_cur_zone = obs.zone.radius <= 0.0 || dist_xy(p2, obs.zone.center) <= obs.zone.radius - 3.0;
        let threat_near_home = threats.iter().any(|th| !th.is_ghost && t - th.last_seen_t < 5.0 && dist2d(th.pos, p) < 55.0);
        in_next_zone && in_cur_zone && !threat_near_home
    };
    let keep = s.home.filter(|&h| valid(h));
    if keep.is_none() || t - s.home_t >= HOME_REPICK_S {
        let fresh = nav.sample_home_spot(
            pos,
            &obs.zone,
            genome.home_ring_frac,
            genome.exposure_weight,
            genome.crowd_pen,
            threats,
            &obs.drops,
            &mut s.rng,
        );
        s.home_t = t;
        s.home = match (keep, fresh) {
            (Some(old), Some(new)) => {
                // Hysteresis: a poorly covered spot always yields; otherwise move only
                // once settled, to a spot clearly elsewhere and meaningfully lower cost.
                let old_exp = nav.exposure_at(old[0], old[1]);
                let poor = old_exp > HOME_POOR_EXPOSURE;
                let old_cost = nav.home_spot_cost(pos, xy(old), &obs.zone, genome.exposure_weight, genome.crowd_pen, threats, &obs.drops);
                let new_cost = nav.home_spot_cost(pos, xy(new), &obs.zone, genome.exposure_weight, genome.crowd_pen, threats, &obs.drops);
                let better = old_cost - new_cost >= HOME_MIN_COST_GAIN;
                let settled = dist2d(pos, old) <= 5.0;
                if poor || (settled && better && dist2d(old, new) >= 6.0) {
                    Some(new)
                } else {
                    Some(old)
                }
            }
            (None, Some(new)) => Some(new),
            (old, None) => old,
        };
        if s.home.is_none() {
            // Mid-ring settle: 0.55 of the next radius, on our side of the center.
            let dir = unit_to(sc, xy(pos));
            let dir = if dir == [0.0, 0.0] { [1.0, 0.0] } else { dir };
            let p = [sc[0] + dir[0] * 0.55 * sr, sc[1] + dir[1] * 0.55 * sr];
            s.home = nav.snap_to_main(p[0], p[1], 20.0).map(|q| [q[0], q[1], nav.ground_at(q[0], q[1])]);
        }
    }
    s.home.map(|h| NavGoal {
        pos: h,
        radius: 3.5,
        exact: false,
    })
}

/// Early unarmed laser search (`benchmaster` `searchPoint`): picks a safe point
/// 90–140 m away inside the next zone, preferring unvisited cells away from
/// known threats, so we actively sweep for a ground laser instead of idling in Home.
fn search_goal(s: &mut ApexState, obs: &Obs, nav: &ApexNav, threats: &[ThreatInfo]) -> Option<NavGoal> {
    let t = obs.t;
    let pos = obs.me.pos;
    let (sc, sr) = safe_circle(obs);
    let valid = |pt: [f64; 3]| {
        let p2 = xy(pt);
        let in_next = obs.zone.next_radius <= 0.0 || dist_xy(p2, sc) <= (sr - 5.0).max(5.0);
        let in_cur = obs.zone.radius <= 0.0 || dist_xy(p2, obs.zone.center) <= obs.zone.radius - 5.0;
        let safe_from_threats = !threats.iter().any(|th| !th.is_ghost && t - th.last_seen_t < 5.0 && dist2d(th.pos, pt) < 75.0);
        in_next && in_cur && safe_from_threats
    };
    let arrived = s.search_pt.is_some_and(|pt| dist2d(pos, pt) < 8.0);
    if s.search_pt.is_none_or(|pt| !valid(pt)) || arrived || t - s.search_t >= 10.0 {
        s.search_t = t;
        let mut best: Option<(f64, [f64; 3])> = None;
        for k in 0..16 {
            let a = f64::from(k) * (std::f64::consts::TAU / 16.0) + s.rng.range(-0.15, 0.15);
            let r = s.rng.range(90.0, 140.0);
            let raw = [pos[0] + r * a.cos(), pos[1] + r * a.sin()];
            let Some(q) = nav.snap_to_main(raw[0], raw[1], 12.0).filter(|q| nav.is_main_xy(q[0], q[1])) else {
                continue;
            };
            let pt = [q[0], q[1], nav.ground_at(q[0], q[1])];
            if !valid(pt) || dist2d(pos, pt) < 60.0 {
                continue;
            }
            let cell = ((q[0] / 40.0).floor() as i32, (q[1] / 40.0).floor() as i32);
            let unvisited = if s.visited_cells.contains(&cell) { 0.0 } else { 100.0 };
            let threat_pen: f64 = threats
                .iter()
                .filter(|th| !th.is_ghost && t - th.last_seen_t < 15.0)
                .map(|th| 2.0 * (160.0 - dist2d(pt, th.pos)).max(0.0))
                .sum();
            let score = unvisited - threat_pen + 0.1 * dist2d(pos, pt).min(140.0);
            if best.is_none_or(|(bs, _)| score > bs) {
                best = Some((score, pt));
            }
        }
        s.search_pt = best.map(|(_, pt)| pt);
    }
    s.search_pt
        .map(|pt| NavGoal {
            pos: pt,
            radius: 5.0,
            exact: false,
        })
        .or_else(|| explore_goal(s, obs, nav))
}

/// A random point inside the safe circle, on our own walkable area when we
/// stand on the ground, else on the main floor.
fn explore_goal(s: &mut ApexState, obs: &Obs, nav: &ApexNav) -> Option<NavGoal> {
    let t = obs.t;
    let pos = obs.me.pos;
    let arrived = s.explore.is_some_and(|e| dist2d(pos, e) < 4.0);
    if s.explore.is_none() || arrived || t - s.explore_t >= EXPLORE_REPICK_S {
        let (sc, sr) = safe_circle(obs);
        let area = nav.ground_area_of(pos);
        s.explore_t = t;
        s.explore = (0..8)
            .filter_map(|_| {
                let a = s.rng.range(-std::f64::consts::PI, std::f64::consts::PI);
                let r = sr * (0.20 + 0.58 * s.rng.next_f64().sqrt());
                let p = [sc[0] + a.cos() * r, sc[1] + a.sin() * r];
                let snapped = match area {
                    Some(a) => nav.snap_to_area(p[0], p[1], 25.0, a).or_else(|| nav.snap_to_main(p[0], p[1], 10.0)),
                    None => nav.snap_to_main(p[0], p[1], 10.0),
                };
                snapped.map(|q| [q[0], q[1], nav.ground_at(q[0], q[1])])
            })
            .max_by(|a, b| dist2d(pos, *a).total_cmp(&dist2d(pos, *b)));
    }
    s.explore.map(|e| NavGoal {
        pos: e,
        radius: 3.0,
        exact: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn act_before_hello_is_idle() {
        let mut apex = Apex::default();
        let obs = Obs { tick: 7, ..Obs::default() };
        assert_eq!(apex.act(&obs), Action::new(7));
    }

    #[test]
    fn apex_holds_practice_and_ffa_genomes() {
        let g = ApexGenome {
            engage_radius: 60.0,
            lone_only: true,
            quiet_walk: true,
            ..ApexGenome::default()
        };
        let apex = Apex::new(g.clone());
        assert_eq!(apex.ffa_genome, g);
        assert_eq!(apex.practice_genome, ApexGenome::practice_default());
        assert!(!apex.practice_genome.ffa_switch);
    }

    fn threat(id: u32, pos: [f64; 3], hp: f64, los: bool) -> ThreatInfo {
        ThreatInfo {
            id,
            pos,
            vel: [0.0; 3],
            aim: [0.0; 2],
            hp,
            max_hp: 1400.0,
            cubes: 0,
            weapons: [None, None],
            charging: false,
            dist: dist_xy([0.0, 0.0], [pos[0], pos[1]]),
            los,
            visible_now: los,
            last_seen_t: 30.0,
            last_hit_us_t: f64::NEG_INFINITY,
            last_beam_t: f64::NEG_INFINITY,
            is_ghost: false,
            is_aware: false,
            threat_dps: 100.0,
            target_score: 0.0,
        }
    }

    fn synthetic_state(world: world::World) -> ApexState {
        let cfg = Config::default();
        let nav = Arc::new(ApexNav::build(Arc::new(world), &cfg.mech, &cfg.vision));
        ApexState::new(nav, 1, 0, &cfg)
    }

    fn me_at(t: f64, pos: [f64; 3]) -> Obs {
        let mut obs = Obs {
            tick: (t * 20.0).round() as u32,
            t,
            alive: 10,
            ..Obs::default()
        };
        obs.me.pos = pos;
        obs.me.hp = 1400.0;
        obs.me.max_hp = 1400.0;
        obs.me.energy = 100.0;
        obs.me.max_energy = 100.0;
        obs.me.grounded = true;
        obs.me.stats = proto::MechStats::default();
        obs.me.weapons = [
            Some(WeaponObs::Gun {
                cooldown: 0.0,
                stats: proto::GunStats::default(),
                ammo: 12.0,
                reload: 0.0,
            }),
            None,
        ];
        obs
    }

    #[test]
    fn close_visible_weak_threat_outranks_far_occluded_one() {
        let far = threat(1, [120.0, 0.0, 0.0], 1400.0, false);
        let near = threat(2, [40.0, 0.0, 0.0], 300.0, true);
        let mut threats = vec![far, near];
        score_threats(&mut threats, 30.0, None);
        assert!(threats[1].target_score < threats[0].target_score);
        let mut obs = me_at(30.0, [0.0; 3]);
        obs.alive = 3; // both engageable
        assert_eq!(pick_target(&threats, &obs, &ApexGenome::default(), false).map(|th| th.id), Some(2));
        // The current target keeps a 50-point edge.
        let mut tied = vec![threat(1, [50.0, 0.0, 0.0], 800.0, true), threat(2, [50.0, 0.0, 0.0], 800.0, true)];
        score_threats(&mut tied, 30.0, Some(2));
        assert!((tied[0].target_score - tied[1].target_score - 50.0).abs() < 1e-9);
    }

    #[test]
    fn attacking_npc_draws_return_fire_without_a_mech_target() {
        let mut s = synthetic_state(world::World::flat(200.0));
        let mut obs = me_at(30.0, [0.0; 3]);
        obs.mechs.push(proto::ActorObs {
            id: 50,
            kind: proto::ActorKind::Npc,
            pos: [40.0, 0.0, 0.0],
            vel: [0.0; 3],
            aim: [std::f64::consts::PI, 0.0],
            hp: None,
            max_hp: None,
            cubes: None,
            weapons: None,
            charging: None,
        });
        obs.events.push(GameEvent::Damaged {
            amount: 10.0,
            weapon: DamageKind::Npc,
            bearing: Some(0.0),
            by: Some(50),
            rear: false,
        });
        let g = ApexGenome::default();
        let fired = (0..20).any(|k| {
            let mut o = obs.clone();
            o.t = 30.0 + k as f64 * 0.05;
            o.tick = 600 + k;
            if k > 0 {
                o.events.clear();
            }
            step(&g, &g, &mut s, &o).fire[0]
        });
        assert_eq!(s.last_npc_hit_id, Some(50));
        let goal = npc_return_fire(&s, &obs).expect("the attacker is a farm target");
        assert_eq!((goal.kind, goal.id), (ItemKind::Npc, 50));
        assert!(fired, "the gun shoots back at the NPC");
    }

    #[test]
    fn stalled_escape_is_abandoned_then_cools_down() {
        let mut w = world::World::flat(100.0);
        w.prisms
            .push(world::Prism::rect(world::PrismKind::Building, [-10.0, -10.0], [10.0, 10.0], 0.0, 12.0));
        let mut s = synthetic_state(w);
        let g = ApexGenome::default();
        let roof = [0.0, 0.0, 12.0];
        step(&g, &g, &mut s, &me_at(10.0, roof));
        assert!(s.escape.is_some(), "an escape off the roof is planned");
        assert_eq!(s.mode, Mode::Escape);
        // Pinned in place: no progress toward the launch spot.
        let mut t = 10.0;
        while s.escape.is_some() && t < 14.0 {
            t += 0.05;
            step(&g, &g, &mut s, &me_at(t, roof));
        }
        assert!(s.escape.is_none(), "abandoned when stalled");
        assert!(
            t - 10.0 > ESCAPE_STALL_S && t - 10.0 < ESCAPE_STALL_S + 0.2,
            "abandoned after {:.2} s",
            t - 10.0
        );
        let quit_t = t;
        for k in 1..40 {
            let tt = quit_t + 0.25 * f64::from(k);
            step(&g, &g, &mut s, &me_at(tt, roof));
            assert!(s.escape.is_none(), "no re-check during the cooldown (t={tt:.2})");
        }
        step(&g, &g, &mut s, &me_at(quit_t + ESCAPE_COOLDOWN_S + 0.05, roof));
        assert!(s.escape.is_some(), "re-checked after the cooldown");
    }

    /// A wide, far-from-shrinking zone around the origin.
    fn in_calm_zone(obs: &mut Obs) {
        obs.me.in_zone = true;
        obs.zone = proto::ZoneObs {
            center: [0.0, 0.0],
            radius: 250.0,
            next_center: [0.0, 0.0],
            next_radius: 250.0,
            shrink_start: 500.0,
            shrink_end: 600.0,
            dps: 0.0,
        };
    }

    fn enemy_mech(id: u32, pos: [f64; 3]) -> proto::ActorObs {
        proto::ActorObs {
            id,
            kind: proto::ActorKind::Mech,
            pos,
            vel: [0.0; 3],
            // Aimed away from us.
            aim: [0.0, 0.0],
            hp: Some(1400.0),
            max_hp: Some(1400.0),
            cubes: Some(0),
            weapons: Some(vec![proto::WeaponKind::Gun]),
            charging: Some(false),
        }
    }

    #[test]
    fn ffa_switch_plays_opening_then_practice_until_ffa() {
        let g = ApexGenome {
            ffa_switch: true,
            ..ApexGenome::ffa_default()
        };
        // Five decisions with a full-HP gun mech 100 m away; `alive` below the
        // 16-mech roster with a harmless zone latches FFA on the third.
        let engaged = |apex: &Apex, t0: f64, alive: u32| {
            let mut s = synthetic_state(world::World::flat(300.0));
            for k in 0..5 {
                let mut obs = me_at(t0 + 0.05 * f64::from(k), [0.0; 3]);
                obs.alive = alive;
                in_calm_zone(&mut obs);
                obs.mechs.push(enemy_mech(5, [100.0, 0.0, 0.0]));
                step(&apex.ffa_genome, &apex.practice_genome, &mut s, &obs);
            }
            (s.target_id == Some(5), s.mem.is_ffa())
        };
        let apex = Apex::new(g.clone());
        assert_eq!(engaged(&apex, 10.0, 16), (false, false), "ffa genome before 45 s: 100 m outside 75 m");
        assert_eq!(engaged(&apex, 40.0, 16), (false, false), "ffa genome at 40 s before 45 s opening ends");
        assert_eq!(
            engaged(&apex, 50.0, 16),
            (true, false),
            "practice genome after 45 s before FFA: 100 m inside 190 m"
        );
        assert_eq!(engaged(&apex, 50.0, 10), (false, true), "FFA genome once FFA latches: 100 m outside 75 m");
        let off = Apex::new(ApexGenome { ffa_switch: false, ..g });
        assert_eq!(engaged(&off, 50.0, 16), (false, false), "without ffa_switch the FFA genome always plays");
    }

    #[test]
    fn duel_start_alive_two_plays_practice_at_once() {
        let apex = Apex::new(ApexGenome::ffa_default());
        let mut s = synthetic_state(world::World::flat(300.0));
        // Capture start_alive = 2 at t <= 1.0 s and verify practice_genome is active at t = 1.0 s.
        for t in [0.05, 1.0] {
            let mut obs = me_at(t, [0.0; 3]);
            obs.alive = 2;
            in_calm_zone(&mut obs);
            obs.mechs.push(enemy_mech(5, [100.0, 0.0, 0.0]));
            step(&apex.ffa_genome, &apex.practice_genome, &mut s, &obs);
        }
        assert_eq!(s.mem.start_alive(), Some(2));
        assert_eq!(s.target_id, Some(5), "1v1 duel engages at 100 m using practice_genome at t = 1.0 s");
    }

    #[test]
    fn latched_retreat_finishes_weak_targets_and_exits_when_calm() {
        let world = world::World::flat(300.0);
        let g = ApexGenome {
            retreat_mode: RetreatMode::Latched,
            retreat_hp: 0.40,
            ..ApexGenome::default()
        };
        let mut obs = me_at(30.0, [0.0; 3]);
        obs.me.hp = 0.3 * 1400.0;
        obs.me.weapons[1] = Some(WeaponObs::Laser {
            cooldown: 0.0,
            charge: 0.0,
            stats: proto::LaserStats::default(),
        });
        let attacker = threat(1, [80.0, 0.0, 0.0], 1400.0, true);
        let mut s = synthetic_state(world.clone());
        assert!(
            wants_retreat(&mut s, &g, &obs, &[attacker], Some(&attacker), 0.3, true),
            "hurt with a threat near"
        );
        // Calm: no damage for 7 s and nothing within 120 m ends the latch below exit HP.
        let calm = me_at(31.0, [0.0; 3]);
        assert!(!wants_retreat(&mut s, &g, &calm, &[], None, 0.6, false));
        assert!(!s.retreat_latched);
        // A nearly dead target at 80 m is finished instead of fled, latch kept.
        let weak = threat(2, [80.0, 0.0, 0.0], 150.0, true);
        let mut s = synthetic_state(world);
        assert!(!wants_retreat(&mut s, &g, &obs, &[weak], Some(&weak), 0.3, true));
        assert!(s.retreat_latched);
    }

    #[test]
    fn pulse_retreat_follows_the_lose_rule() {
        let world = world::World::flat(300.0);
        let g = ApexGenome {
            retreat_mode: RetreatMode::Pulse15s,
            retreat_hp: 0.35,
            multi_retreat_hp: 0.55,
            ..ApexGenome::default()
        };
        let mut obs = me_at(30.0, [0.0; 3]);
        obs.me.weapons[1] = Some(WeaponObs::Laser {
            cooldown: 0.0,
            charge: 0.0,
            stats: proto::LaserStats::default(),
        });
        // Losing the duel: we are at 30%, the target at 90%.
        let strong = threat(1, [60.0, 0.0, 0.0], 0.9 * 1400.0, true);
        let mut s = synthetic_state(world.clone());
        assert!(wants_retreat(&mut s, &g, &obs, &[strong], Some(&strong), 0.3, true));
        assert!(s.retreat_pulse_until > 30.0 && s.retreat_pulse_until <= 31.5);
        // A target under 25% is finished.
        let weak = threat(1, [60.0, 0.0, 0.0], 0.2 * 1400.0, true);
        let mut s = synthetic_state(world.clone());
        assert!(!wants_retreat(&mut s, &g, &obs, &[weak], Some(&weak), 0.3, true));
        // Two mechs within 160 m below `multi_retreat_hp`.
        let pair = [threat(2, [140.0, 0.0, 0.0], 1400.0, true), threat(3, [0.0, 150.0, 0.0], 1400.0, true)];
        let mut s = synthetic_state(world);
        assert!(wants_retreat(&mut s, &g, &obs, &pair, None, 0.5, false));
    }

    #[test]
    fn ffa_vulture_engages_hurt_mech_looking_away() {
        let mut obs = me_at(30.0, [0.0; 3]);
        obs.me.weapons[1] = Some(WeaponObs::Lightning {
            cooldown: 0.0,
            stats: proto::LightningStats::default(),
        });
        let g = ApexGenome {
            engage_radius: 60.0,
            quiet_walk: false,
            ..ApexGenome::default()
        };
        // 100 m away at 35% HP (490 HP), aimed away from us (aim 0 points away along +x).
        let hurt = threat(1, [100.0, 0.0, 0.0], 0.35 * 1400.0, true);
        assert!(!engageable(&hurt, &[hurt], &obs, &g, false), "not outside stealth/FFA");
        assert!(engageable(&hurt, &[hurt], &obs, &g, true));
        // Also requires lone, dist < 120, hp_frac < 0.40, and our hp_frac > 0.80.
        let bystander = threat(2, [-50.0, 0.0, 0.0], 1400.0, true);
        assert!(!engageable(&hurt, &[hurt, bystander], &obs, &g, true));
        let far_hurt = threat(1, [130.0, 0.0, 0.0], 0.35 * 1400.0, true);
        assert!(!engageable(&far_hurt, &[far_hurt], &obs, &g, true));
        // Not when it looks at us.
        let mut watching = hurt;
        watching.aim = [std::f64::consts::PI, 0.0];
        assert!(!engageable(&watching, &[watching], &obs, &g, true));
        // Not when we are hurt ourselves.
        let mut low = obs.clone();
        low.me.hp = 0.75 * 1400.0;
        assert!(!engageable(&hurt, &[hurt], &low, &g, true));
    }

    #[test]
    fn unarmed_weapon_rush_preempts_optional_fight() {
        let g = ApexGenome {
            engage_radius: 150.0,
            quiet_walk: false,
            ..ApexGenome::default()
        };
        let mut s = synthetic_state(world::World::flat(300.0));
        s.stuck_ref_t = 30.0;
        let mut obs = me_at(30.0, [0.0; 3]);
        in_calm_zone(&mut obs);
        obs.pickups.push(proto::PickupObs {
            id: 40,
            kind: proto::WeaponKind::Laser,
            pos: [0.0, 40.0, 0.0],
        });
        obs.mechs.push(enemy_mech(5, [70.0, 0.0, 0.0]));
        step(&g, &g, &mut s, &obs);
        assert_eq!(s.mode, Mode::Loot);
        assert_eq!(s.item_id, Some(40), "heads for the weapon");
        assert_eq!(s.target_id, None, "the 70 m target is dropped while rushing");
    }

    #[test]
    fn unarmed_stealth_lobby_searches_early_then_holds_home() {
        let apex = Apex::new(ApexGenome::ffa_default());
        let mut s = synthetic_state(world::World::flat(300.0));
        for (t, pos) in [(0.05, [0.0, 0.0, 0.0]), (5.0, [25.0, 0.0, 0.0])] {
            let mut obs = me_at(t, pos);
            obs.alive = 16;
            in_calm_zone(&mut obs);
            step(&apex.ffa_genome, &apex.practice_genome, &mut s, &obs);
        }
        assert_eq!(s.mode, Mode::Explore, "unarmed early lobby sweeps for a laser");
        // Once armed with a laser (or after t=45s), it settles in Home.
        let mut armed = me_at(6.0, [25.0, 0.0, 0.0]);
        armed.alive = 16;
        in_calm_zone(&mut armed);
        armed.me.weapons[1] = Some(WeaponObs::Laser {
            cooldown: 0.0,
            charge: 0.0,
            stats: proto::LaserStats::default(),
        });
        step(&apex.ffa_genome, &apex.practice_genome, &mut s, &armed);
        assert_eq!(s.mode, Mode::Home);
    }

    fn npc(id: u32, pos: [f64; 3]) -> proto::ActorObs {
        proto::ActorObs {
            id,
            kind: proto::ActorKind::Npc,
            pos,
            vel: [0.0; 3],
            aim: [0.0, 0.0],
            hp: None,
            max_hp: None,
            cubes: None,
            weapons: None,
            charging: None,
        }
    }

    #[test]
    fn ttl_last_stand_respects_low_hp_and_npc_dps() {
        let mut obs = me_at(30.0, [0.0; 3]);
        obs.me.hp = 700.0;
        // Target dies in 2 s; one aware 100 DPS attacker gives us a 7 s TTL.
        let mut tg = threat(1, [40.0, 0.0, 0.0], 200.0, true);
        tg.is_aware = true;
        let threats = vec![tg];
        assert!(ttl_last_stand(&obs, &threats, Some(&tg), 0.5));
        // Below 25% HP it never stands.
        assert!(!ttl_last_stand(&obs, &threats, Some(&tg), 0.2));
        // Ten NPCs within 75 m add 250 DPS: TTL 2 s, no longer worth standing.
        for k in 0..10 {
            obs.mechs.push(npc(100 + k, [30.0, f64::from(k), 0.0]));
        }
        assert!(!ttl_last_stand(&obs, &threats, Some(&tg), 0.5));
        // NPCs beyond 75 m do not count.
        for m in &mut obs.mechs {
            m.pos[0] = 90.0;
        }
        assert!(ttl_last_stand(&obs, &threats, Some(&tg), 0.5));
    }

    #[test]
    fn engageable_ignores_glances_and_crowds() {
        let obs = me_at(30.0, [0.0; 3]);
        let g = ApexGenome {
            engage_radius: 60.0,
            lone_only: false,
            quiet_walk: false,
            ..ApexGenome::default()
        };
        // Aimed at us from 140 m: a glance unless charging (and even then too far).
        let mut th = threat(1, [140.0, 0.0, 0.0], 1400.0, true);
        th.aim = [std::f64::consts::PI, 0.0];
        assert!(!engageable(&th, &[th], &obs, &g, false));
        th.charging = true;
        assert!(!engageable(&th, &[th], &obs, &g, false));
        // Charging and aimed at 100 m engages.
        let mut close = threat(1, [100.0, 0.0, 0.0], 1400.0, true);
        close.aim = [std::f64::consts::PI, 0.0];
        close.charging = true;
        assert!(engageable(&close, &[close], &obs, &g, false));
        close.charging = false;
        assert!(!engageable(&close, &[close], &obs, &g, false));

        // lone_only: a target inside engage_radius with no company is engageable.
        let g = ApexGenome {
            engage_radius: 100.0,
            lone_only: true,
            quiet_walk: false,
            ..ApexGenome::default()
        };
        let tg = threat(1, [90.0, 0.0, 0.0], 1400.0, true);
        assert!(engageable(&tg, &[tg], &obs, &g, false));
        // Another mech 100 m from the target (190 m from us) crowds it.
        let near_th = threat(2, [190.0, 0.0, 0.0], 1400.0, true);
        assert!(!engageable(&tg, &[tg, near_th], &obs, &g, false));
        // Another mech 100 m from us (but 135 m from the target) crowds it too.
        let near_me = threat(3, [0.0, 100.0, 0.0], 1400.0, true);
        assert!(!engageable(&tg, &[tg, near_me], &obs, &g, false));
        // Stale (> 5 s) threat no longer crowds.
        let mut stale_me = near_me;
        stale_me.last_seen_t = 24.5;
        assert!(engageable(&tg, &[tg, stale_me], &obs, &g, false));
        // Far from both, still lone.
        let far = threat(4, [-150.0, 0.0, 0.0], 1400.0, true);
        assert!(engageable(&tg, &[tg, far], &obs, &g, false));
        // Ghosts never crowd.
        let mut ghost = threat(5, [0.0, 50.0, 0.0], 1400.0, false);
        ghost.is_ghost = true;
        assert!(engageable(&tg, &[tg, ghost], &obs, &g, false));
    }

    #[test]
    fn wants_retreat_on_outgunned_and_crowds() {
        let world = world::World::flat(300.0);
        for mode in [RetreatMode::Latched, RetreatMode::TtlLastStand, RetreatMode::Pulse15s] {
            let g = ApexGenome {
                retreat_mode: mode,
                retreat_hp: 0.3,
                multi_retreat_hp: 0.4,
                ..ApexGenome::default()
            };
            let mut obs = me_at(30.0, [0.0; 3]);
            obs.me.hp = 0.75 * 1400.0;
            // An enemy laser at 105 m (> 55 m) with LOS while we are unarmed and < 80% HP triggers outgunned.
            let mut laser = threat(1, [105.0, 0.0, 0.0], 1400.0, true);
            laser.weapons = [Some(proto::WeaponKind::Laser), None];
            let mut s = synthetic_state(world.clone());
            assert!(wants_retreat(&mut s, &g, &obs, &[laser], None, 0.75, false), "{mode:?} outgunned");
            // Not when HP >= 80% (even vs laser at 80 m).
            let mut s = synthetic_state(world.clone());
            assert!(!wants_retreat(&mut s, &g, &obs, &[laser], None, 0.85, false), "{mode:?} healthy");
            let mut close_laser = threat(1, [80.0, 0.0, 0.0], 1400.0, true);
            close_laser.weapons = [Some(proto::WeaponKind::Laser), None];
            let mut s = synthetic_state(world.clone());
            assert!(
                !wants_retreat(&mut s, &g, &obs, &[close_laser], None, 0.95, false),
                "{mode:?} unarmed at 95% HP does not panic-retreat"
            );
            // Unarmed at <= 55 m vs laser does not trigger the > 55 m outgunned rule.
            let mut brawl_laser = threat(1, [50.0, 0.0, 0.0], 1400.0, true);
            brawl_laser.weapons = [Some(proto::WeaponKind::Laser), None];
            let mut s = synthetic_state(world.clone());
            assert!(!wants_retreat(&mut s, &g, &obs, &[brawl_laser], None, 0.75, false), "{mode:?} <= 55 m");
            // Not when we hold a laser ourselves.
            let mut armed = obs.clone();
            armed.me.weapons[1] = Some(WeaponObs::Laser {
                cooldown: 0.0,
                charge: 0.0,
                stats: proto::LaserStats::default(),
            });
            let mut s = synthetic_state(world.clone());
            assert!(!wants_retreat(&mut s, &g, &armed, &[laser], None, 0.75, false), "{mode:?} armed");
            // Not without LOS.
            let mut hidden = laser;
            hidden.los = false;
            let mut s = synthetic_state(world.clone());
            assert!(!wants_retreat(&mut s, &g, &obs, &[hidden], None, 0.75, false), "{mode:?} no los");

            // Three gun mechs within 120 m at 80% HP.
            let crowd = [
                threat(2, [50.0, 0.0, 0.0], 1400.0, true),
                threat(3, [0.0, 60.0, 0.0], 1400.0, true),
                threat(4, [-70.0, 0.0, 0.0], 1400.0, true),
            ];
            let mut s = synthetic_state(world.clone());
            assert!(wants_retreat(&mut s, &g, &obs, &crowd, None, 0.8, false), "{mode:?} crowd");
            // Two threats within 110 m while unarmed: triggers at < 70% HP, not at 80%.
            let two = [threat(2, [100.0, 0.0, 0.0], 1400.0, true), threat(3, [0.0, 100.0, 0.0], 1400.0, true)];
            let mut s = synthetic_state(world.clone());
            assert!(!wants_retreat(&mut s, &g, &obs, &two, None, 0.8, false), "{mode:?} two at 80%");
            let mut s = synthetic_state(world.clone());
            assert!(wants_retreat(&mut s, &g, &obs, &two, None, 0.65, false), "{mode:?} two unarmed <70%");
        }
    }

    #[test]
    fn unstuck_sustains_jump_across_multiple_ticks_in_non_main_component() {
        // Large enclosed basin (>= 400 cells) where find_escape stays None and ZoneMust moves toward safe circle.
        let mut w = world::World::flat(120.0);
        for (a, b) in [
            ([-60.0, -60.0], [60.0, -54.0]),
            ([-60.0, 54.0], [60.0, 60.0]),
            ([-60.0, -54.0], [-54.0, 54.0]),
            ([54.0, -54.0], [60.0, 54.0]),
            ([24.0, -20.0], [28.0, 20.0]),
        ] {
            w.prisms.push(world::Prism::rect(world::PrismKind::Building, a, b, 0.0, 7.0));
        }
        let mut s = synthetic_state(w);
        let g = ApexGenome::default();
        // Stand inside the enclosed basin > 28 m from main, pinned against an interior wall toward the zone.
        let mut jumps = Vec::new();
        for k in 0..25 {
            let t = 10.0 + f64::from(k) * 0.1;
            let mut obs = me_at(t, [20.0, 0.0, 0.0]);
            obs.me.in_zone = false;
            obs.me.grounded = k < 17; // airborne once jump begins
            obs.zone = proto::ZoneObs {
                center: [90.0, 0.0],
                radius: 20.0,
                next_center: [90.0, 0.0],
                next_radius: 20.0,
                shrink_start: 5.0,
                shrink_end: 20.0,
                dps: 10.0,
            };
            let act = step(&g, &g, &mut s, &obs);
            jumps.push((t, s.mode, act.jump));
        }
        let unstuck_jumps = jumps.iter().filter(|&&(_, m, j)| m == Mode::Unstuck && j).count();
        assert!(unstuck_jumps >= 5, "sustains jump across multiple ticks, got {jumps:?}");
    }

    #[test]
    fn stealth_ffa_unarmed_retaliation_and_clean_engage() {
        let g = ApexGenome::default();
        let mut obs_unarmed = me_at(30.0, [0.0; 3]);
        obs_unarmed.alive = 10;
        // Unarmed retaliation / self-defense at 55 m is allowed in stealth FFA.
        let mut foe_55 = threat(1, [55.0, 0.0, 0.0], 1400.0, true);
        foe_55.last_hit_us_t = 29.0;
        assert!(engageable(&foe_55, &[foe_55], &obs_unarmed, &g, true), "unarmed retaliation allowed at 55 m");
        // Even at 80 m when actively hitting us with LOS.
        let mut attacker_80 = threat(1, [80.0, 0.0, 0.0], 1400.0, true);
        attacker_80.last_hit_us_t = 29.0;
        assert!(engageable(&attacker_80, &[attacker_80], &obs_unarmed, &g, true));
        // Passive full-HP foe at 80 m is skipped while unarmed.
        let passive_80 = threat(1, [80.0, 0.0, 0.0], 1400.0, true);
        assert!(!engageable(&passive_80, &[passive_80], &obs_unarmed, &g, true));

        // Armed with Laser, healthy: crowded full-HP foe at 135 m is NOT engaged under lone_only,
        // while weak prey < 350 HP at < 130 m with LOS IS engaged.
        let mut obs_laser = obs_unarmed.clone();
        obs_laser.me.weapons[1] = Some(WeaponObs::Laser {
            cooldown: 0.0,
            charge: 0.0,
            stats: proto::LaserStats::default(),
        });
        let bystander = threat(2, [-40.0, 0.0, 0.0], 1400.0, true);
        let weak_120 = threat(1, [120.0, 0.0, 0.0], 300.0, true);
        assert!(engageable(&weak_120, &[weak_120, bystander], &obs_laser, &g, true));
        let busy_135 = threat(1, [135.0, 0.0, 0.0], 1400.0, true);
        assert!(!engageable(&busy_135, &[busy_135, bystander], &obs_laser, &g, true));

        // Endgame unlocks at alive <= 3 or next_radius <= 40.0 (not alive = 5).
        let mut alive5 = obs_laser.clone();
        alive5.alive = 5;
        assert!(!engageable(&busy_135, &[busy_135, bystander], &alive5, &g, true));
        let mut alive3 = obs_laser.clone();
        alive3.alive = 3;
        assert!(engageable(&busy_135, &[busy_135, bystander], &alive3, &g, true));
        let mut small_zone = obs_laser;
        small_zone.zone.next_radius = 40.0;
        assert!(engageable(&busy_135, &[busy_135, bystander], &small_zone, &g, true));
    }

    #[test]
    fn avoid_cover_holds_without_4s_reset_while_hidden() {
        let mut w = world::World::flat(200.0);
        w.prisms
            .push(world::Prism::rect(world::PrismKind::Building, [-6.0, -25.0], [0.0, 25.0], 0.0, 10.0));
        let nav = ApexNav::build(Arc::new(w.clone()), &proto::MechStats::default(), &proto::VisionCfg::default());
        let mut s = synthetic_state(w);
        let mut obs = me_at(10.0, [4.0, 28.0, 0.0]);
        in_calm_zone(&mut obs);
        let foe = threat(1, [40.0, 0.0, 0.0], 1400.0, true);
        let first = avoid_cover(&mut s, &obs, &nav, &[foe]).expect("cover found behind wall");
        assert!((s.cover_t - 10.0).abs() < 1e-9);

        // More than 4.0 s later, while cached cover is still in zone and hidden from foe, keep it without reset.
        obs.t = 16.5;
        let second = avoid_cover(&mut s, &obs, &nav, &[foe]).expect("cached cover retained");
        assert_eq!(second, first);
        assert!((s.cover_t - 10.0).abs() < 1e-9, "avoid_cover must not reset every 4s while hidden");

        // Once a threat moves to get LOS to the cached cover, avoid_cover re-picks.
        obs.t = 17.0;
        let flanking = threat(1, [-40.0, 0.0, 0.0], 1400.0, true);
        let _ = avoid_cover(&mut s, &obs, &nav, &[flanking]);
        assert!((s.cover_t - 17.0).abs() < 1e-9);
    }
}
