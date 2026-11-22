//! [`BotMemory`]: everything the apex bot remembers between decisions.
//!
//! An observation only shows what is in view right now. Memory keeps enemy
//! mechs (with a short velocity history for lead), NPCs, items, sounds, the
//! direction of hits from unseen attackers, a gun-lead bandit, and a latched
//! free-for-all detector. Maps are `BTreeMap`s so iteration order, and with it
//! every decision, is deterministic.

use std::collections::{BTreeMap, VecDeque};
use std::f64::consts::PI;

use proto::{
    ActorKind, ActorObs, BeamKind, BeamObs, Config, DamageKind, GameEvent, Obs, SoundDist, SoundKind, TakeFailReason, VisionCfg, WeaponKind, WeaponObs,
    WeaponsCfg,
};
use world::World;

use super::types::{clamp, dist2d, dist3d, wrap_pi, ItemKind, Rng64, ThreatInfo};

/// Mech ids are below this; NPCs, crates, cubes, and pickups are above.
const MECH_ID_LIMIT: u32 = 16;
/// Seconds an unseen enemy mech is remembered.
const MECH_TRACK_TTL: f64 = 18.0;
/// Seconds an unseen NPC track is remembered.
const NPC_TRACK_TTL: f64 = 60.0;
/// Seconds of velocity history kept per enemy mech.
const VEL_WINDOW: f64 = 0.6;
/// Youngest sample usable as the far end of the acceleration difference.
const ACCEL_MIN_AGE: f64 = 0.2;
/// Damping applied to the measured acceleration.
const ACCEL_DAMP: f64 = 0.5;
/// Horizontal acceleration cap while airborne / grounded, m/s².
const ACCEL_CAP_AIR: f64 = 12.0;
const ACCEL_CAP_GROUND: f64 = 58.0;
/// Vertical acceleration bounds, m/s².
const ACCEL_Z_MIN: f64 = -22.0;
const ACCEL_Z_MAX: f64 = 34.0;
/// Vertical speed above which a mech counts as airborne, m/s.
const AIRBORNE_VZ: f64 = 1.0;
/// Seconds a hit from an unseen attacker stays a ghost threat.
const GHOST_TTL: f64 = 3.5;
/// Meters along the hit bearing at which the ghost is placed.
const GHOST_RANGE: f64 = 110.0;
/// A visible mech within this many radians of a hit bearing explains it.
const GHOST_EXPLAIN_RAD: f64 = 0.35;
/// Id reported for the ghost threat; never a real actor.
const GHOST_ID: u32 = u32::MAX;
/// Unseen items within this range with clear sight get deleted, meters.
const ITEM_DELETE_R: f64 = 35.0;
/// Consecutive clear-sight misses (decisions) before an item is deleted.
const ITEM_DELETE_MISSES: u32 = 2;
/// Unseen lifetimes per item kind, seconds. Crates are static and never expire.
const CUBE_TTL: f64 = 40.0;
const WEAPON_TTL: f64 = 60.0;
const NPC_ITEM_TTL: f64 = 30.0;
const DROP_TTL: f64 = 60.0;
/// Standing within this range of a cube or weapon this long means we cannot
/// reach it: blacklist it for `CHASE_BLACKLIST_S`.
const CHASE_R: f64 = 10.0;
const CHASE_LIMIT_S: f64 = 8.0;
const CHASE_BLACKLIST_S: f64 = 25.0;
/// Seconds a heard sound is remembered.
const SOUND_TTL: f64 = 6.0;
/// Consecutive positive decisions before free-for-all latches.
const FFA_LATCH_TICKS: u32 = 3;
/// Third-party beam hits needed, and their minimum spacing in seconds.
const FFA_BEAM_HITS: usize = 2;
const FFA_BEAM_SPACING: f64 = 1.0;
/// Foreign-death thresholds: `(deaths, highest zone dps)`. Deaths under a
/// harmless zone are someone else's kills.
const FFA_DEATHS_CALM: (u32, f64) = (2, 35.0);
const FFA_DEATHS_ANY: (u32, f64) = (3, 100.0);
/// A third-party beam ends farther than this from us, meters.
const FFA_BEAM_FAR_FROM_US: f64 = 20.0;
/// A third-party beam ends farther than this from any NPC or crate, meters.
const FFA_BEAM_FAR_FROM_PVE: f64 = 12.0;
/// A beam end within this horizontal distance of a mech hits it, meters.
const FFA_BEAM_HIT_R: f64 = 4.5;
/// A beam end this far below the feet or above the head still hits, meters.
const FFA_BEAM_HIT_Z: f64 = 2.5;
/// The struck mech must have been seen within this many seconds of the beam.
const FFA_BEAM_SEEN_S: f64 = 1.5;
/// Margin inside the vision ranges, meters, and yaw wedge, radians, before an
/// unseen item counts as gone.
const SEE_RANGE_MARGIN: f64 = 8.0;
const SEE_FOV_MARGIN: f64 = 0.08;
/// A visible mech this close in yaw to our aim is what an ammo drop was fired at.
const FIRE_AIM_TOL: f64 = 0.3;
/// Enemy aim within this many radians of us counts as aware.
const AWARE_AIM_TOL: f64 = 0.3;
/// Seconds after hitting us that an enemy still counts as aware.
const AWARE_HIT_S: f64 = 3.0;

/// Shots per bandit arm decision.
const BANDIT_BUCKET: u32 = 12;
/// Shots each arm gets before the bandit starts exploiting.
const BANDIT_MIN_SHOTS: u32 = 12;
/// Exploration probability once every arm has its minimum shots.
const BANDIT_EXPLORE: f64 = 0.2;

/// A three-armed bandit over the gun lead multiplier `km`.
///
/// Every 12 gun shots at mechs it picks an arm: any arm with
/// fewer than 12 shots first, then 20% random exploration,
/// else the arm with the best smoothed hit rate `(hits + 1) / (shots + 4)`.
#[derive(Clone, Debug)]
pub struct HitBandit {
    /// Lead multipliers.
    pub arms: [f64; 3],
    /// Gun shots fired at mechs with each arm active.
    pub shots: [u32; 3],
    /// Gun hits on mechs with each arm active.
    pub hits: [u32; 3],
    /// Index of the arm in play.
    pub active: usize,
    /// Shots fired in the current bucket.
    pub bucket_shots: u32,
    /// `(slot, ammo)` of the gun at the last update.
    last_ammo: Option<(usize, f64)>,
}

impl Default for HitBandit {
    fn default() -> Self {
        Self {
            arms: [0.6, 1.0, 1.3],
            shots: [0; 3],
            hits: [0; 3],
            active: 1,
            bucket_shots: 0,
            last_ammo: None,
        }
    }
}

impl HitBandit {
    /// The lead multiplier in play.
    pub fn current_km(&self) -> f64 {
        self.arms[self.active]
    }

    fn record_shots(&mut self, n: u32, rng: &mut Rng64) {
        for _ in 0..n {
            self.shots[self.active] += 1;
            self.bucket_shots += 1;
            if self.bucket_shots >= BANDIT_BUCKET {
                self.bucket_shots = 0;
                self.active = self.choose(rng);
            }
        }
    }

    fn record_hit(&mut self) {
        self.hits[self.active] += 1;
    }

    fn choose(&self, rng: &mut Rng64) -> usize {
        if let Some(i) = (0..3).filter(|&i| self.shots[i] < BANDIT_MIN_SHOTS).min_by_key(|&i| (self.shots[i], i)) {
            return i;
        }
        if rng.chance(BANDIT_EXPLORE) {
            return (rng.next_u64() % 3) as usize;
        }
        // Hits can trail an arm switch, so never credit more hits than shots.
        let rate = |i: usize| (f64::from(self.hits[i].min(self.shots[i])) + 1.0) / (f64::from(self.shots[i]) + 4.0);
        let mut best = 0;
        for i in 1..3 {
            if rate(i) > rate(best) {
                best = i;
            }
        }
        best
    }

    /// Counts drops in gun ammo since the last update as shots when `at_mech`.
    /// Returns the number of shots counted.
    fn observe_ammo(&mut self, weapons: &[Option<WeaponObs>; 2], at_mech: bool, rng: &mut Rng64) -> u32 {
        let now = weapons.iter().enumerate().find_map(|(slot, w)| match w {
            Some(WeaponObs::Gun { ammo, .. }) => Some((slot, *ammo)),
            _ => None,
        });
        let mut counted = 0;
        if let (Some((slot, ammo)), Some((last_slot, last))) = (now, self.last_ammo) {
            if slot == last_slot && ammo < last {
                let n = (last - ammo).round().max(0.0) as u32;
                if at_mech && n > 0 {
                    self.record_shots(n, rng);
                    counted = n;
                }
            }
        }
        self.last_ammo = now;
        counted
    }
}

/// An item, NPC, or supply drop remembered from an earlier observation.
#[derive(Clone, Copy, Debug)]
pub struct RememberedItem {
    /// Actor or pickup id.
    pub id: u32,
    /// What it is.
    pub kind: ItemKind,
    /// The weapon, for a weapon pickup.
    pub weapon_kind: Option<WeaponKind>,
    /// Last known position, meters.
    pub pos: [f64; 3],
    /// Hit points, for crates and NPCs.
    pub hp: Option<f64>,
    /// Match time it was last seen, seconds.
    pub last_seen_t: f64,
}

/// Bookkeeping beside each remembered item.
#[derive(Clone, Copy, Debug, Default)]
struct ItemMeta {
    /// Consecutive decisions unseen while in range with clear sight.
    misses: u32,
    /// When we came within [`CHASE_R`] of it and stayed.
    near_since: Option<f64>,
}

/// What we know about one enemy mech's position and motion.
#[derive(Clone, Debug)]
struct MechTrack {
    pos: [f64; 3],
    vel: [f64; 3],
    aim: [f64; 2],
    hp: Option<f64>,
    max_hp: Option<f64>,
    cubes: u32,
    weapons: [Option<WeaponKind>; 2],
    charging: bool,
    last_seen_t: f64,
    visible_now: bool,
    /// `(t, vel)` samples from the last [`VEL_WINDOW`] seconds.
    vel_hist: VecDeque<(f64, [f64; 3])>,
    accel: [f64; 3],
}

impl MechTrack {
    fn new(pos: [f64; 3], t: f64) -> Self {
        Self {
            pos,
            vel: [0.0; 3],
            aim: [0.0; 2],
            hp: None,
            max_hp: None,
            cubes: 0,
            weapons: [None; 2],
            charging: false,
            last_seen_t: t,
            visible_now: false,
            vel_hist: VecDeque::new(),
            accel: [0.0; 3],
        }
    }
}

/// What happened between us and one enemy mech. Kept for the whole match.
#[derive(Clone, Copy, Debug)]
struct Ledger {
    last_hit_us_t: f64,
    last_beam_t: f64,
    damage_dealt_to_us: f64,
    damage_we_dealt: f64,
}

impl Default for Ledger {
    fn default() -> Self {
        Self {
            last_hit_us_t: f64::NEG_INFINITY,
            last_beam_t: f64::NEG_INFINITY,
            damage_dealt_to_us: 0.0,
            damage_we_dealt: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct NpcTrack {
    pos: [f64; 3],
    hp: Option<f64>,
    last_seen_t: f64,
}

/// Damage from an attacker we could not see.
#[derive(Clone, Copy, Debug)]
struct Ghost {
    bearing: f64,
    t: f64,
    pos: [f64; 3],
    weapon: Option<WeaponKind>,
}

#[derive(Clone, Debug)]
struct HeardSound {
    kind: SoundKind,
    bearing: f64,
    dist: SoundDist,
    /// Match time the sound was made.
    t: f64,
}

/// The apex bot's memory across decisions. Call [`BotMemory::update`] once per observation.
#[derive(Clone, Debug)]
pub struct BotMemory {
    slot: u32,
    mech_count: u32,
    mech_height: f64,
    eye_height: f64,
    muzzle_height: f64,
    mech_hp: f64,
    weapons_cfg: WeaponsCfg,
    vision: VisionCfg,
    tracks: BTreeMap<u32, MechTrack>,
    ledger: BTreeMap<u32, Ledger>,
    npcs: BTreeMap<u32, NpcTrack>,
    items: Vec<RememberedItem>,
    item_meta: BTreeMap<u32, ItemMeta>,
    blacklist: BTreeMap<u32, f64>,
    ghost: Option<Ghost>,
    sounds: Vec<HeardSound>,
    bandit: HitBandit,
    last_damage_t: f64,
    last_hostile_damage_t: f64,
    last_damage_dealt_t: f64,
    last_mech_shot_t: f64,
    last_kill_t: f64,
    last_kill_pos: Option<[f64; 3]>,
    my_kills: u32,
    third_party_beam_ts: Vec<f64>,
    start_alive: Option<u32>,
    ffa_streak: u32,
    ffa_detected: bool,
}

impl BotMemory {
    /// Empty memory for seat `slot` under rules `cfg`.
    pub fn new(slot: u32, cfg: &Config) -> Self {
        Self {
            slot,
            mech_count: cfg.match_.mechs.max(0.0) as u32,
            mech_height: cfg.mech.height,
            eye_height: cfg.mech.eye_height,
            muzzle_height: cfg.mech.muzzle_height,
            mech_hp: cfg.mech.hp,
            weapons_cfg: cfg.weapons.clone(),
            vision: cfg.vision.clone(),
            tracks: BTreeMap::new(),
            ledger: BTreeMap::new(),
            npcs: BTreeMap::new(),
            items: Vec::new(),
            item_meta: BTreeMap::new(),
            blacklist: BTreeMap::new(),
            ghost: None,
            sounds: Vec::new(),
            bandit: HitBandit::default(),
            last_damage_t: f64::NEG_INFINITY,
            last_hostile_damage_t: f64::NEG_INFINITY,
            last_damage_dealt_t: f64::NEG_INFINITY,
            last_mech_shot_t: f64::NEG_INFINITY,
            last_kill_t: f64::NEG_INFINITY,
            last_kill_pos: None,
            my_kills: 0,
            third_party_beam_ts: Vec::new(),
            start_alive: None,
            ffa_streak: 0,
            ffa_detected: false,
        }
    }

    fn is_self(&self, id: u32, obs: &Obs) -> bool {
        id == self.slot || id == obs.me.id
    }

    fn is_enemy_mech(&self, id: u32, obs: &Obs) -> bool {
        id < MECH_ID_LIMIT && !self.is_self(id, obs)
    }

    fn eye(&self, obs: &Obs) -> [f64; 3] {
        let h = if obs.me.stats.eye_height > 0.0 {
            obs.me.stats.eye_height
        } else {
            self.eye_height
        };
        let p = obs.me.pos;
        [p[0], p[1], p[2] + h]
    }

    /// Folds one observation into memory.
    pub fn update(&mut self, obs: &Obs, world: &World, rng: &mut Rng64) {
        let t = obs.t;
        self.update_actors(obs);
        self.update_beams(obs);
        self.update_events(obs);
        self.tracks.retain(|_, tr| t - tr.last_seen_t <= MECH_TRACK_TTL);
        self.npcs.retain(|_, n| t - n.last_seen_t <= NPC_TRACK_TTL);
        if self.ghost.is_some_and(|g| t - g.t > GHOST_TTL) {
            self.ghost = None;
        }
        self.update_items(obs, world);
        self.update_bandit(obs, rng);
        self.update_ffa(obs);
        self.update_sounds(obs);
    }

    fn update_actors(&mut self, obs: &Obs) {
        let t = obs.t;
        for tr in self.tracks.values_mut() {
            tr.visible_now = false;
        }
        for a in &obs.mechs {
            match a.kind {
                ActorKind::Mech if self.is_enemy_mech(a.id, obs) => self.see_mech(a, t),
                ActorKind::Mech => {}
                ActorKind::Npc => {
                    self.npcs.insert(
                        a.id,
                        NpcTrack {
                            pos: a.pos,
                            hp: a.hp,
                            last_seen_t: t,
                        },
                    );
                }
            }
        }
        for tr in self.tracks.values_mut().filter(|tr| !tr.visible_now) {
            // Stale motion would mislead lead; start over when it reappears.
            tr.vel_hist.clear();
            tr.accel = [0.0; 3];
        }
    }

    fn see_mech(&mut self, a: &ActorObs, t: f64) {
        let tr = self.tracks.entry(a.id).or_insert_with(|| MechTrack::new(a.pos, t));
        tr.pos = a.pos;
        tr.vel = a.vel;
        tr.aim = a.aim;
        tr.hp = a.hp;
        tr.max_hp = a.max_hp;
        if let Some(c) = a.cubes {
            tr.cubes = c;
        }
        if let Some(w) = &a.weapons {
            tr.weapons = [w.first().copied(), w.get(1).copied()];
        }
        tr.charging = a.charging.unwrap_or(false);
        tr.last_seen_t = t;
        tr.visible_now = true;
        tr.vel_hist.push_back((t, a.vel));
        while tr.vel_hist.front().is_some_and(|&(ts, _)| t - ts > VEL_WINDOW + 1e-9) {
            tr.vel_hist.pop_front();
        }
        tr.accel = lead2_accel(&tr.vel_hist, t, a.vel);
    }

    fn update_beams(&mut self, obs: &Obs) {
        let t = obs.t;
        for b in &obs.beams {
            if !self.is_enemy_mech(b.by, obs) {
                continue;
            }
            let bt = t - b.age;
            let led = self.ledger.entry(b.by).or_default();
            led.last_beam_t = led.last_beam_t.max(bt);
            if let Some(&muzzle) = b.path.first() {
                let origin = [muzzle[0], muzzle[1], muzzle[2] - self.muzzle_height];
                match self.tracks.get_mut(&b.by) {
                    None => {
                        self.tracks.insert(b.by, MechTrack::new(origin, bt));
                    }
                    Some(tr) if !tr.visible_now && tr.last_seen_t < bt => {
                        tr.pos = origin;
                        tr.last_seen_t = bt;
                    }
                    Some(_) => {}
                }
            }
            if self.is_third_party_beam_hit(b, bt, obs) && self.third_party_beam_ts.iter().all(|&ts| (ts - bt).abs() > FFA_BEAM_SPACING) {
                self.third_party_beam_ts.push(bt);
            }
        }
    }

    /// A laser beam that ends on another enemy mech: far from us, far from
    /// every known NPC and crate (so it was not PvE fire), and within reach of
    /// a mech other than the shooter seen around the time of the shot.
    fn is_third_party_beam_hit(&self, b: &BeamObs, bt: f64, obs: &Obs) -> bool {
        if b.kind != BeamKind::Laser && b.path.len() < 2 {
            return false;
        }
        let Some(&end) = b.path.last() else {
            return false;
        };
        if dist2d(end, obs.me.pos) <= FFA_BEAM_FAR_FROM_US {
            return false;
        }
        let near_pve = self.npcs.values().any(|n| dist2d(end, n.pos) <= FFA_BEAM_FAR_FROM_PVE)
            || obs
                .mechs
                .iter()
                .any(|a| a.kind == ActorKind::Npc && dist2d(end, a.pos) <= FFA_BEAM_FAR_FROM_PVE)
            || obs.crates.iter().any(|c| dist2d(end, c.pos) <= FFA_BEAM_FAR_FROM_PVE)
            || self
                .items
                .iter()
                .any(|it| matches!(it.kind, ItemKind::Crate | ItemKind::Npc) && dist2d(end, it.pos) <= FFA_BEAM_FAR_FROM_PVE);
        if near_pve {
            return false;
        }
        self.tracks.iter().any(|(&id, tr)| {
            id != b.by
                && (bt - tr.last_seen_t).abs() <= FFA_BEAM_SEEN_S
                && dist2d(end, tr.pos) <= FFA_BEAM_HIT_R
                && end[2] >= tr.pos[2] - FFA_BEAM_HIT_Z
                && end[2] <= tr.pos[2] + self.mech_height + FFA_BEAM_HIT_Z
        })
    }

    /// True when protocol §8 vision would show an item at `pos` to an eye at
    /// `eye`: inside the near range with a clear line, or inside the vision
    /// range, the yaw wedge, and clear sight (canopy included). Margins keep
    /// borderline cases from deleting items we merely failed to see.
    pub fn should_see_item(&self, obs: &Obs, world: &World, eye: [f64; 3], pos: [f64; 3]) -> bool {
        let stats = &obs.me.stats;
        let target = [pos[0], pos[1], pos[2] + 0.5];
        let d = dist3d(eye, target);
        let near = self.vision.near_range + stats.near_range_bonus - SEE_RANGE_MARGIN;
        if d <= near {
            return world.los(eye, target);
        }
        let range = self.vision.range + stats.vision_bonus - SEE_RANGE_MARGIN;
        if d > range {
            return false;
        }
        let half_fov = 0.5 * self.vision.fov_deg.to_radians();
        if wrap_pi(yaw_to(eye, target) - obs.me.aim[0]).abs() > half_fov - SEE_FOV_MARGIN {
            return false;
        }
        world.sight(eye, target)
    }

    fn update_events(&mut self, obs: &Obs) {
        let t = obs.t;
        for ev in &obs.events {
            match ev {
                GameEvent::Damaged {
                    amount, weapon, bearing, by, ..
                } => {
                    self.last_damage_t = t;
                    // Our own mortar splash is not an attacker.
                    if by.is_some_and(|id| self.is_self(id, obs)) {
                        continue;
                    }
                    if *weapon != DamageKind::Zone {
                        self.last_hostile_damage_t = t;
                    }
                    if let Some(id) = by.filter(|&id| self.is_enemy_mech(id, obs)) {
                        let led = self.ledger.entry(id).or_default();
                        led.last_hit_us_t = t;
                        led.damage_dealt_to_us += amount;
                    }
                    if *weapon != DamageKind::Zone && *weapon != DamageKind::Npc {
                        if let Some(b) = *bearing {
                            self.note_hit_bearing(obs, b, *by, weapon);
                        }
                    }
                }
                GameEvent::Hit { target, amount, weapon, .. } => {
                    if self.is_enemy_mech(*target, obs) {
                        self.ledger.entry(*target).or_default().damage_we_dealt += amount;
                        self.last_damage_dealt_t = t;
                        if *weapon == DamageKind::Gun {
                            self.bandit.record_hit();
                        }
                    }
                }
                GameEvent::Kill { target } => {
                    self.last_kill_t = t;
                    self.last_kill_pos = self.tracks.get(target).map(|tr| tr.pos).or_else(|| self.npcs.get(target).map(|n| n.pos));
                    if *target < MECH_ID_LIMIT {
                        self.my_kills += 1;
                    }
                    self.tracks.remove(target);
                    self.npcs.remove(target);
                    self.remove_item(*target);
                }
                GameEvent::Took { id, .. } | GameEvent::Drop { id, .. } => self.remove_item(*id),
                GameEvent::TakeFailed {
                    id,
                    reason: TakeFailReason::Gone,
                } => self.remove_item(*id),
                _ => {}
            }
        }
    }

    fn note_hit_bearing(&mut self, obs: &Obs, bearing: f64, by: Option<u32>, weapon: &DamageKind) {
        let by_visible = by.is_some_and(|id| self.tracks.get(&id).is_some_and(|tr| tr.visible_now));
        let explained = by_visible || (by.is_none() && self.visible_mech_near_bearing(obs.me.pos, bearing));
        if explained {
            return;
        }
        let p = obs.me.pos;
        self.ghost = Some(Ghost {
            bearing,
            t: obs.t,
            pos: [p[0] + GHOST_RANGE * bearing.cos(), p[1] + GHOST_RANGE * bearing.sin(), p[2]],
            weapon: weapon_of(weapon),
        });
    }

    fn visible_mech_near_bearing(&self, from: [f64; 3], bearing: f64) -> bool {
        self.tracks
            .values()
            .any(|tr| tr.visible_now && wrap_pi(yaw_to(from, tr.pos) - bearing).abs() <= GHOST_EXPLAIN_RAD)
    }

    fn remove_item(&mut self, id: u32) {
        self.items.retain(|it| it.id != id);
        self.item_meta.remove(&id);
    }

    fn upsert_item(&mut self, item: RememberedItem) {
        match self.items.iter_mut().find(|it| it.id == item.id) {
            Some(it) => *it = item,
            None => self.items.push(item),
        }
        self.item_meta.entry(item.id).or_default().misses = 0;
    }

    fn update_items(&mut self, obs: &Obs, world: &World) {
        let t = obs.t;
        let item = |id, kind, weapon_kind, pos, hp| RememberedItem {
            id,
            kind,
            weapon_kind,
            pos,
            hp,
            last_seen_t: t,
        };
        for c in &obs.cubes {
            self.upsert_item(item(c.id, ItemKind::Cube, None, c.pos, None));
        }
        for c in &obs.crates {
            self.upsert_item(item(c.id, ItemKind::Crate, None, c.pos, c.hp));
        }
        for p in &obs.pickups {
            self.upsert_item(item(p.id, ItemKind::Weapon, Some(p.kind), p.pos, None));
        }
        for d in &obs.drops {
            self.upsert_item(item(d.id, ItemKind::Drop, None, d.pos, None));
        }
        let seen_npcs: Vec<RememberedItem> = self
            .npcs
            .iter()
            .filter(|(_, n)| n.last_seen_t == t)
            .map(|(&id, n)| item(id, ItemKind::Npc, None, n.pos, n.hp))
            .collect();
        for it in seen_npcs {
            self.upsert_item(it);
        }

        // Forget what has gone stale, and what we should see but do not.
        let eye = self.eye(obs);
        let mut gone = Vec::new();
        for it in &self.items {
            if it.last_seen_t == t {
                continue;
            }
            let ttl = match it.kind {
                ItemKind::Cube => CUBE_TTL,
                ItemKind::Weapon => WEAPON_TTL,
                ItemKind::Npc => NPC_ITEM_TTL,
                ItemKind::Drop => DROP_TTL,
                ItemKind::Crate => f64::INFINITY,
            };
            if t - it.last_seen_t > ttl {
                gone.push(it.id);
                continue;
            }
            // Cubes and pickups we should see but do not are gone: taken by
            // someone, or never where we thought.
            if matches!(it.kind, ItemKind::Cube | ItemKind::Weapon) && self.should_see_item(obs, world, eye, it.pos) {
                let listed = match it.kind {
                    ItemKind::Cube => obs.cubes.iter().any(|c| c.id == it.id),
                    _ => obs.pickups.iter().any(|p| p.id == it.id),
                };
                if !listed {
                    gone.push(it.id);
                    continue;
                }
            }
            let target = [it.pos[0], it.pos[1], it.pos[2] + 0.5];
            let in_view = dist3d(eye, it.pos) <= ITEM_DELETE_R && world.los(eye, target);
            let meta = self.item_meta.entry(it.id).or_default();
            if in_view {
                meta.misses += 1;
                if meta.misses >= ITEM_DELETE_MISSES {
                    gone.push(it.id);
                }
            } else {
                meta.misses = 0;
            }
        }
        for id in gone {
            self.remove_item(id);
        }

        // Chase timers: hovering next to a pickup that never gets picked up.
        let me = obs.me.pos;
        let mut stuck = Vec::new();
        for it in &self.items {
            let meta = self.item_meta.entry(it.id).or_default();
            let chaseable = matches!(it.kind, ItemKind::Cube | ItemKind::Weapon);
            if chaseable && it.last_seen_t == t && dist2d(me, it.pos) <= CHASE_R {
                let since = *meta.near_since.get_or_insert(t);
                if t - since >= CHASE_LIMIT_S {
                    meta.near_since = None;
                    stuck.push(it.id);
                }
            } else {
                meta.near_since = None;
            }
        }
        for id in stuck {
            self.blacklist_item(id, t + CHASE_BLACKLIST_S);
        }
        self.blacklist.retain(|_, &mut until| until > t);
    }

    fn update_bandit(&mut self, obs: &Obs, rng: &mut Rng64) {
        let me = &obs.me;
        let range = me
            .weapons
            .iter()
            .find_map(|w| match w {
                Some(WeaponObs::Gun { stats, .. }) => Some(stats.range),
                _ => None,
            })
            .unwrap_or(self.weapons_cfg.gun.range);
        let at_mech = self
            .tracks
            .values()
            .any(|tr| tr.visible_now && dist3d(me.pos, tr.pos) <= range * 1.2 && wrap_pi(yaw_to(me.pos, tr.pos) - me.aim[0]).abs() <= FIRE_AIM_TOL);
        if self.bandit.observe_ammo(&me.weapons, at_mech, rng) > 0 {
            self.last_mech_shot_t = obs.t;
        }
    }

    /// Latches free-for-all after three straight decisions with a signal that
    /// a practice match cannot produce: deaths we did not cause under a zone
    /// too weak to explain them, or two spaced laser hits between other mechs.
    fn update_ffa(&mut self, obs: &Obs) {
        if self.start_alive.is_none() && obs.alive > 0 && obs.t <= 1.0 {
            self.start_alive = Some(obs.alive);
        }
        if self.ffa_detected {
            return;
        }
        let beams = self.third_party_beam_ts.len() >= FFA_BEAM_HITS;
        let dps = obs.zone.dps;
        let deaths = obs.alive != 0 && {
            let start_alive = self.start_alive.unwrap_or(self.mech_count);
            let foreign = start_alive.saturating_sub(obs.alive).saturating_sub(self.my_kills);
            (foreign >= FFA_DEATHS_CALM.0 && dps <= FFA_DEATHS_CALM.1) || (foreign >= FFA_DEATHS_ANY.0 && dps <= FFA_DEATHS_ANY.1)
        };
        if beams || deaths {
            self.ffa_streak += 1;
            if self.ffa_streak >= FFA_LATCH_TICKS {
                self.ffa_detected = true;
            }
        } else {
            self.ffa_streak = 0;
        }
    }

    fn update_sounds(&mut self, obs: &Obs) {
        let t = obs.t;
        for s in &obs.sounds {
            let st = t - s.age;
            let dup = self
                .sounds
                .iter()
                .any(|h| h.kind == s.kind && (h.t - st).abs() < 0.06 && wrap_pi(h.bearing - s.bearing).abs() < 0.05);
            if !dup {
                self.sounds.push(HeardSound {
                    kind: s.kind.clone(),
                    bearing: s.bearing,
                    dist: s.dist,
                    t: st,
                });
            }
        }
        self.sounds.retain(|h| t - h.t <= SOUND_TTL);
    }

    /// True once a free-for-all has been detected. Never resets.
    pub fn is_ffa(&self) -> bool {
        self.ffa_detected
    }

    /// Initial alive mech count captured at match start, if seen.
    pub fn start_alive(&self) -> Option<u32> {
        self.start_alive
    }

    /// The gun lead bandit.
    pub fn hit_bandit(&self) -> &HitBandit {
        &self.bandit
    }

    /// Every remembered enemy mech as a threat, plus the ghost of an unseen
    /// attacker when no visible mech explains it. `target_score` is left at 0
    /// for the caller to fill.
    pub fn threats(&self, obs: &Obs, world: &World) -> Vec<ThreatInfo> {
        let t = obs.t;
        let me = obs.me.pos;
        let eye = self.eye(obs);
        let mut out = Vec::with_capacity(self.tracks.len() + 1);
        for (&id, tr) in &self.tracks {
            if t - tr.last_seen_t > MECH_TRACK_TTL {
                continue;
            }
            let led = self.ledger.get(&id).copied().unwrap_or_default();
            let dist = dist3d(me, tr.pos);
            let body = [tr.pos[0], tr.pos[1], tr.pos[2] + 0.6 * self.mech_height];
            let aims_at_us = tr.visible_now && wrap_pi(tr.aim[0] - yaw_to(tr.pos, me)).abs() <= AWARE_AIM_TOL;
            let is_aware = aims_at_us || t - led.last_hit_us_t <= AWARE_HIT_S;
            let stale = if tr.visible_now {
                1.0
            } else {
                clamp(1.0 - (t - tr.last_seen_t) / MECH_TRACK_TTL, 0.3, 1.0)
            };
            let threat_dps = self.weapon_dps(&tr.weapons, dist) * if is_aware { 1.0 } else { 0.5 } * stale;
            out.push(ThreatInfo {
                id,
                pos: tr.pos,
                vel: tr.vel,
                aim: tr.aim,
                hp: tr.hp.unwrap_or(self.mech_hp),
                max_hp: tr.max_hp.unwrap_or(self.mech_hp),
                cubes: tr.cubes,
                weapons: tr.weapons,
                charging: tr.charging,
                dist,
                los: world.los(eye, body),
                visible_now: tr.visible_now,
                last_seen_t: tr.last_seen_t,
                last_hit_us_t: led.last_hit_us_t,
                last_beam_t: led.last_beam_t,
                is_ghost: false,
                is_aware,
                threat_dps,
                target_score: 0.0,
            });
        }
        if let Some(g) = self.ghost.filter(|g| t - g.t <= GHOST_TTL) {
            let bearing = yaw_to(me, g.pos);
            if !self.visible_mech_near_bearing(me, bearing) {
                let dist = dist3d(me, g.pos);
                let weapons = [Some(g.weapon.unwrap_or(WeaponKind::Gun)), None];
                out.push(ThreatInfo {
                    id: GHOST_ID,
                    pos: g.pos,
                    vel: [0.0; 3],
                    aim: [wrap_pi(bearing + PI), 0.0],
                    hp: self.mech_hp,
                    max_hp: self.mech_hp,
                    cubes: 0,
                    weapons,
                    charging: false,
                    dist,
                    los: false,
                    visible_now: false,
                    last_seen_t: g.t,
                    last_hit_us_t: g.t,
                    last_beam_t: f64::NEG_INFINITY,
                    is_ghost: true,
                    is_aware: true,
                    threat_dps: self.weapon_dps(&weapons, dist.min(GHOST_RANGE)),
                    target_score: 0.0,
                });
            }
        }
        out
    }

    /// Rough damage per second the loadout deals at `dist` with base stats. No
    /// known weapon means a gun, the starting loadout.
    fn weapon_dps(&self, weapons: &[Option<WeaponKind>; 2], dist: f64) -> f64 {
        let w = &self.weapons_cfg;
        let one = |k: WeaponKind| match k {
            WeaponKind::Gun => {
                let g = &w.gun;
                let cycle = g.mag * g.cooldown + g.reload;
                if dist > g.range || cycle <= 0.0 {
                    0.0
                } else {
                    g.damage * g.mag / cycle * clamp(1.2 - dist / g.range, 0.2, 0.9)
                }
            }
            WeaponKind::Laser => {
                let l = &w.laser;
                let cycle = l.charge + l.cooldown;
                if dist > l.range || cycle <= 0.0 {
                    0.0
                } else {
                    0.7 * l.damage / cycle
                }
            }
            WeaponKind::Lightning => {
                let l = &w.lightning;
                if dist > l.range || l.cooldown <= 0.0 {
                    0.0
                } else {
                    0.8 * l.damage / l.cooldown
                }
            }
            WeaponKind::Mortar => {
                let m = &w.mortar;
                let reach = if m.gravity > 0.0 { m.speed * m.speed / m.gravity } else { f64::INFINITY };
                if dist > reach || m.cooldown <= 0.0 {
                    0.0
                } else {
                    0.35 * m.damage / m.cooldown
                }
            }
        };
        if weapons.iter().all(Option::is_none) {
            return one(WeaponKind::Gun);
        }
        weapons.iter().flatten().map(|&k| one(k)).sum()
    }

    /// Match time we last took damage from any source, the zone included.
    pub fn last_damage_t(&self) -> f64 {
        self.last_damage_t
    }

    /// Match time we last hit an enemy mech.
    pub fn last_damage_dealt_t(&self) -> f64 {
        self.last_damage_dealt_t
    }

    /// Match time of our last kill (mech or NPC).
    pub fn last_kill_t(&self) -> f64 {
        self.last_kill_t
    }

    /// Where the victim of our last kill was last seen, when known.
    pub fn last_kill_pos(&self) -> Option<[f64; 3]> {
        self.last_kill_pos
    }

    /// `(bearing, age)` of the last hit from an unseen attacker, within 3.5 s.
    pub fn unseen_hit_bearing(&self, t: f64) -> Option<(f64, f64)> {
        self.ghost.filter(|g| t - g.t <= GHOST_TTL).map(|g| (g.bearing, (t - g.t).max(0.0)))
    }

    /// Remembered items, NPCs, and drops.
    pub fn items(&self) -> &[RememberedItem] {
        &self.items
    }

    /// Ignore item `id` until match time `until_t`.
    pub fn blacklist_item(&mut self, id: u32, until_t: f64) {
        self.blacklist.insert(id, until_t);
    }

    /// True while item `id` is blacklisted at time `t`.
    pub fn is_blacklisted(&self, id: u32, t: f64) -> bool {
        self.blacklist.get(&id).is_some_and(|&until| until > t)
    }

    /// Damped acceleration estimate of mech `id`, zero when unknown.
    pub fn track_accel(&self, id: u32) -> [f64; 3] {
        self.tracks.get(&id).map_or([0.0; 3], |tr| tr.accel)
    }

    /// The most telling sound heard in the last 6 s: nearer, more combat-like,
    /// and fresher wins. NPC gunfire and supply drops are ignored.
    pub fn best_sound(&self, t: f64) -> Option<(f64, SoundDist, SoundKind)> {
        let score = |h: &HeardSound| {
            let dist_w = match h.dist {
                SoundDist::Near => 3.0,
                SoundDist::Mid => 2.0,
                SoundDist::Far => 1.0,
            };
            let kind_w = match h.kind {
                SoundKind::Gun | SoundKind::Laser | SoundKind::LaserCharge | SoundKind::Lightning | SoundKind::Mortar | SoundKind::Explosion => 1.0,
                SoundKind::Step | SoundKind::Boost | SoundKind::Jet | SoundKind::Reload => 0.5,
                _ => 0.25,
            };
            dist_w + kind_w - 0.25 * (t - h.t)
        };
        self.sounds
            .iter()
            .filter(|h| t - h.t <= SOUND_TTL && !matches!(h.kind, SoundKind::NpcGun | SoundKind::Drop | SoundKind::Other(_)))
            .max_by(|a, b| score(a).total_cmp(&score(b)))
            .map(|h| (h.bearing, h.dist, h.kind.clone()))
    }

    /// True if within `window` seconds we took non-zone damage, hit a mech, or
    /// fired the gun at one.
    pub fn recently_engaged_or_damaged(&self, t: f64, window: f64) -> bool {
        t - self.last_hostile_damage_t <= window || t - self.last_damage_dealt_t <= window || t - self.last_mech_shot_t <= window
    }
}

/// World-frame yaw from `a` toward `b`.
fn yaw_to(a: [f64; 3], b: [f64; 3]) -> f64 {
    (b[1] - a[1]).atan2(b[0] - a[0])
}

fn weapon_of(kind: &DamageKind) -> Option<WeaponKind> {
    match kind {
        DamageKind::Gun => Some(WeaponKind::Gun),
        DamageKind::Mortar => Some(WeaponKind::Mortar),
        DamageKind::Laser => Some(WeaponKind::Laser),
        DamageKind::Lightning => Some(WeaponKind::Lightning),
        _ => None,
    }
}

/// `lead2` acceleration: half the velocity change against the oldest sample
/// aged 0.2..=0.6 s, horizontal magnitude capped by ground state, z bounded.
fn lead2_accel(hist: &VecDeque<(f64, [f64; 3])>, t: f64, v: [f64; 3]) -> [f64; 3] {
    let Some(&(ts, old)) = hist.iter().find(|&&(ts, _)| {
        let age = t - ts;
        (ACCEL_MIN_AGE - 1e-9..=VEL_WINDOW + 1e-9).contains(&age)
    }) else {
        return [0.0; 3];
    };
    let dt = t - ts;
    if dt <= 0.0 {
        return [0.0; 3];
    }
    let mut a = [0.0; 3];
    for i in 0..3 {
        a[i] = ACCEL_DAMP * (v[i] - old[i]) / dt;
    }
    let cap = if v[2].abs() > AIRBORNE_VZ { ACCEL_CAP_AIR } else { ACCEL_CAP_GROUND };
    let h = a[0].hypot(a[1]);
    if h > cap {
        a[0] *= cap / h;
        a[1] *= cap / h;
    }
    a[2] = clamp(a[2], ACCEL_Z_MIN, ACCEL_Z_MAX);
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::{BeamKind, DropObs, GunStats, ItemObs, MechStats, PickupObs, SelfObs, SoundObs, ZoneObs};
    use world::{Prism, PrismKind};

    const ME: u32 = 0;

    fn gun(ammo: f64) -> WeaponObs {
        WeaponObs::Gun {
            cooldown: 0.0,
            stats: GunStats::default(),
            ammo,
            reload: 0.0,
        }
    }

    fn obs_at(t: f64) -> Obs {
        Obs {
            tick: (t * 20.0).round() as u32,
            t,
            alive: 16,
            me: SelfObs {
                id: ME,
                pos: [0.0, 0.0, 0.0],
                hp: 1400.0,
                max_hp: 1400.0,
                stats: MechStats::default(),
                weapons: [Some(gun(12.0)), None],
                ..SelfObs::default()
            },
            ..Obs::default()
        }
    }

    fn mech(id: u32, pos: [f64; 3], vel: [f64; 3]) -> ActorObs {
        ActorObs {
            id,
            kind: ActorKind::Mech,
            pos,
            vel,
            aim: [0.0, 0.0],
            hp: Some(1400.0),
            max_hp: Some(1400.0),
            cubes: Some(0),
            weapons: Some(vec![WeaponKind::Gun]),
            charging: Some(false),
        }
    }

    fn npc(id: u32, pos: [f64; 3]) -> ActorObs {
        ActorObs {
            id,
            kind: ActorKind::Npc,
            pos,
            vel: [0.0; 3],
            aim: [0.0; 2],
            hp: Some(350.0),
            max_hp: Some(350.0),
            cubes: None,
            weapons: None,
            charging: None,
        }
    }

    fn damaged(amount: f64, weapon: DamageKind, bearing: Option<f64>, by: Option<u32>) -> GameEvent {
        GameEvent::Damaged {
            amount,
            weapon,
            bearing,
            by,
            rear: false,
        }
    }

    fn hit(target: u32, amount: f64, weapon: DamageKind) -> GameEvent {
        GameEvent::Hit {
            target,
            amount,
            weapon,
            effect: None,
            rear: false,
        }
    }

    fn beam(by: u32, from: [f64; 3], to: [f64; 3], age: f64) -> BeamObs {
        BeamObs {
            by,
            kind: BeamKind::Laser,
            path: vec![from, to],
            age,
        }
    }

    fn setup() -> (BotMemory, World, Rng64) {
        (BotMemory::new(ME, &Config::default()), World::flat(1000.0), Rng64::new(7, ME))
    }

    /// A wall from x = 10 to 20 across y = -50..50, 30 m tall.
    fn walled() -> World {
        let mut w = World::flat(1000.0);
        w.prisms.push(Prism::rect(PrismKind::Building, [10.0, -50.0], [20.0, 50.0], 0.0, 30.0));
        w
    }

    // ---- HitBandit ----

    #[test]
    fn bandit_starts_neutral_and_tries_every_arm() {
        let mut b = HitBandit::default();
        let mut rng = Rng64::new(1, 0);
        assert_eq!(b.current_km(), 1.0);
        b.record_shots(11, &mut rng);
        assert_eq!((b.active, b.bucket_shots), (1, 11));
        b.record_shots(1, &mut rng);
        assert_eq!((b.active, b.bucket_shots), (0, 0), "first untried arm, lowest index");
        b.record_shots(12, &mut rng);
        assert_eq!(b.active, 2);
        assert_eq!(b.current_km(), 1.3);
        b.record_shots(12, &mut rng);
        assert_eq!(b.shots, [12, 12, 12]);
    }

    #[test]
    fn bandit_exploits_the_best_arm_most_of_the_time() {
        let b = HitBandit {
            shots: [12, 12, 12],
            hits: [1, 10, 2],
            ..HitBandit::default()
        };
        let mut rng = Rng64::new(99, 3);
        let mut picks = [0u32; 3];
        for _ in 0..2000 {
            picks[b.choose(&mut rng)] += 1;
        }
        // 80% exploit + 1/3 of the 20% explore ≈ 86.7% for arm 1.
        assert!(picks[1] > 1600 && picks[1] < 1860, "{picks:?}");
        assert!(picks[0] > 60 && picks[2] > 60, "exploration reaches every arm: {picks:?}");
    }

    #[test]
    fn bandit_rate_caps_hits_at_shots() {
        let b = HitBandit {
            shots: [12, 12, 40],
            hits: [50, 0, 30],
            ..HitBandit::default()
        };
        // Arm 0 capped: 13/16 = 0.81; arm 2: 31/44 = 0.70.
        let mut rng = Rng64::new(0, 0);
        let mut zero = 0;
        for _ in 0..200 {
            if b.choose(&mut rng) == 0 {
                zero += 1;
            }
        }
        assert!(zero > 150);
    }

    #[test]
    fn bandit_counts_ammo_drops_only_at_mechs_and_not_reloads() {
        let mut b = HitBandit::default();
        let mut rng = Rng64::new(1, 0);
        assert_eq!(b.observe_ammo(&[Some(gun(12.0)), None], true, &mut rng), 0, "first sample is the baseline");
        assert_eq!(b.observe_ammo(&[Some(gun(10.0)), None], true, &mut rng), 2);
        assert_eq!(b.observe_ammo(&[Some(gun(9.0)), None], false, &mut rng), 0, "not at a mech");
        assert_eq!(b.observe_ammo(&[Some(gun(12.0)), None], true, &mut rng), 0, "reload");
        assert_eq!(b.observe_ammo(&[None, Some(gun(11.0))], true, &mut rng), 0, "slot change resets the baseline");
        assert_eq!(b.observe_ammo(&[None, Some(gun(10.0))], true, &mut rng), 1);
        assert_eq!(b.observe_ammo(&[None, None], true, &mut rng), 0);
        assert_eq!(b.shots[1], 3);
    }

    #[test]
    fn memory_feeds_bandit_from_obs() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(1.0);
        o.mechs = vec![mech(3, [100.0, 0.0, 0.0], [0.0; 3])];
        m.update(&o, &w, &mut rng);
        let mut o = obs_at(1.1);
        o.mechs = vec![mech(3, [100.0, 0.0, 0.0], [0.0; 3])];
        o.me.weapons = [Some(gun(9.0)), None];
        o.events = vec![hit(3, 45.0, DamageKind::Gun), hit(120, 45.0, DamageKind::Gun)];
        m.update(&o, &w, &mut rng);
        assert_eq!(m.hit_bandit().shots[1], 3);
        assert_eq!(m.hit_bandit().hits[1], 1, "NPC hits do not count");
        assert!(m.recently_engaged_or_damaged(1.5, 1.0));
        assert!(!m.recently_engaged_or_damaged(3.0, 1.0));
        assert_eq!(m.last_damage_dealt_t(), 1.1);

        // Facing away from the mech: shots are not at it.
        let mut o = obs_at(1.2);
        o.mechs = vec![mech(3, [100.0, 0.0, 0.0], [0.0; 3])];
        o.me.aim = [PI, 0.0];
        o.me.weapons = [Some(gun(5.0)), None];
        m.update(&o, &w, &mut rng);
        assert_eq!(m.hit_bandit().shots[1], 3);
    }

    // ---- tracks and acceleration ----

    fn feed_vel(m: &mut BotMemory, w: &World, rng: &mut Rng64, samples: &[(f64, [f64; 3])]) {
        for &(t, v) in samples {
            let mut o = obs_at(t);
            o.mechs = vec![mech(5, [50.0, 0.0, 0.0], v)];
            m.update(&o, w, rng);
        }
    }

    #[test]
    fn accel_is_damped_difference_against_oldest_valid_sample() {
        let (mut m, w, mut rng) = setup();
        feed_vel(&mut m, &w, &mut rng, &[(1.0, [0.0; 3]), (1.1, [1.0, 0.0, 0.0])]);
        assert_eq!(m.track_accel(5), [0.0; 3], "no sample old enough");
        feed_vel(&mut m, &w, &mut rng, &[(1.2, [2.0, 0.0, 0.0]), (1.3, [3.0, 0.0, 0.0]), (1.4, [4.0, 0.0, 0.0])]);
        // Oldest sample (t=1.0, age 0.4): 0.5 * 4 / 0.4 = 5.
        let a = m.track_accel(5);
        assert!((a[0] - 5.0).abs() < 1e-9 && a[1] == 0.0, "{a:?}");
        // Window slides: at t = 1.7 the t=1.0 sample is gone, oldest is 1.1.
        feed_vel(&mut m, &w, &mut rng, &[(1.5, [4.0, 0.0, 0.0]), (1.6, [4.0, 0.0, 0.0]), (1.7, [4.0, 0.0, 0.0])]);
        let a = m.track_accel(5);
        assert!((a[0] - 0.5 * 3.0 / 0.6).abs() < 1e-9, "{a:?}");
        assert_eq!(m.track_accel(9), [0.0; 3], "unknown id");
    }

    #[test]
    fn accel_clamps_ground_air_and_vertical() {
        let (mut m, w, mut rng) = setup();
        feed_vel(&mut m, &w, &mut rng, &[(1.0, [0.0; 3]), (1.2, [60.0, 80.0, 0.0])]);
        let a = m.track_accel(5);
        assert!((a[0].hypot(a[1]) - 58.0).abs() < 1e-9, "{a:?}");
        assert!((a[1] / a[0] - 80.0 / 60.0).abs() < 1e-9, "direction kept");

        let (mut m, w, mut rng) = setup();
        feed_vel(&mut m, &w, &mut rng, &[(1.0, [0.0; 3]), (1.2, [20.0, 0.0, 30.0])]);
        let a = m.track_accel(5);
        assert!((a[0] - 12.0).abs() < 1e-9, "airborne cap: {a:?}");
        assert_eq!(a[2], 34.0);

        let (mut m, w, mut rng) = setup();
        feed_vel(&mut m, &w, &mut rng, &[(1.0, [0.0; 3]), (1.2, [0.0, 0.0, -30.0])]);
        assert_eq!(m.track_accel(5)[2], -22.0);
    }

    #[test]
    fn accel_resets_when_out_of_sight() {
        let (mut m, w, mut rng) = setup();
        feed_vel(&mut m, &w, &mut rng, &[(1.0, [0.0; 3]), (1.3, [3.0, 0.0, 0.0])]);
        assert!(m.track_accel(5)[0] > 0.0);
        m.update(&obs_at(1.4), &w, &mut rng);
        assert_eq!(m.track_accel(5), [0.0; 3]);
    }

    #[test]
    fn mech_tracks_expire_after_18s_and_npcs_after_60s() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(0.0);
        o.mechs = vec![mech(5, [50.0, 0.0, 0.0], [0.0; 3]), npc(120, [30.0, 0.0, 0.0])];
        m.update(&o, &w, &mut rng);
        m.update(&obs_at(17.9), &w, &mut rng);
        assert_eq!(m.threats(&obs_at(17.9), &w).len(), 1);
        m.update(&obs_at(18.1), &w, &mut rng);
        assert!(m.threats(&obs_at(18.1), &w).is_empty());
        assert!(m.npcs.contains_key(&120));
        m.update(&obs_at(60.5), &w, &mut rng);
        assert!(!m.npcs.contains_key(&120));
    }

    #[test]
    fn self_and_npcs_are_not_threats() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(1.0);
        o.mechs = vec![mech(ME, [0.0; 3], [0.0; 3]), npc(130, [20.0, 0.0, 0.0])];
        m.update(&o, &w, &mut rng);
        assert!(m.threats(&o, &w).is_empty());
    }

    // ---- threats ----

    #[test]
    fn threats_report_los_awareness_and_damage_ledger() {
        let (mut m, _, mut rng) = setup();
        let w = walled();
        let mut o = obs_at(5.0);
        let mut facing = mech(3, [-60.0, 0.0, 0.0], [0.0; 3]);
        facing.aim = [0.0, 0.0]; // looking east, at us
        let mut away = mech(4, [60.0, 0.0, 0.0], [0.0; 3]);
        away.aim = [0.0, 0.0]; // looking east, away from us, behind the wall
        o.mechs = vec![facing, away];
        o.events = vec![damaged(45.0, DamageKind::Gun, Some(PI), Some(3))];
        m.update(&o, &w, &mut rng);
        let th = m.threats(&o, &w);
        assert_eq!(th.len(), 2);
        let a = th.iter().find(|x| x.id == 3).unwrap();
        let b = th.iter().find(|x| x.id == 4).unwrap();
        assert!(a.los && !b.los);
        assert!(a.is_aware && !b.is_aware);
        assert_eq!(a.last_hit_us_t, 5.0);
        assert!(a.visible_now && (a.dist - 60.0).abs() < 1e-9);
        assert_eq!(a.weapons, [Some(WeaponKind::Gun), None]);
        assert!(a.threat_dps > b.threat_dps && b.threat_dps > 0.0);
        assert!(!a.is_ghost);
        assert_eq!(m.ledger[&3].damage_dealt_to_us, 45.0);
        assert!(m.unseen_hit_bearing(5.0).is_none(), "attacker was visible");
    }

    #[test]
    fn threat_dps_depends_on_weapon_and_range() {
        let (m, _, _) = setup();
        let laser = [Some(WeaponKind::Laser), None];
        assert!(m.weapon_dps(&laser, 100.0) > 0.0);
        assert_eq!(m.weapon_dps(&laser, 300.0), 0.0);
        let zap = [Some(WeaponKind::Lightning), None];
        assert!(m.weapon_dps(&zap, 30.0) > m.weapon_dps(&zap, 50.0));
        assert!(m.weapon_dps(&[None, None], 50.0) > 0.0, "unknown means gun");
        let both = [Some(WeaponKind::Gun), Some(WeaponKind::Laser)];
        assert!(m.weapon_dps(&both, 100.0) > m.weapon_dps(&laser, 100.0));
    }

    // ---- ghost ----

    #[test]
    fn unseen_hit_makes_a_ghost_for_3_5s() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(10.0);
        o.events = vec![damaged(45.0, DamageKind::Gun, Some(PI / 2.0), Some(7))];
        m.update(&o, &w, &mut rng);
        let (b, age) = m.unseen_hit_bearing(10.0).unwrap();
        assert_eq!((b, age), (PI / 2.0, 0.0));
        let th = m.threats(&o, &w);
        assert_eq!(th.len(), 1);
        let g = th[0];
        assert!(g.is_ghost && g.is_aware && !g.los);
        assert!(g.pos[0].abs() < 1e-9 && (g.pos[1] - 110.0).abs() < 1e-9);
        assert!(g.threat_dps > 0.0);
        assert_eq!(m.unseen_hit_bearing(13.4).map(|x| x.0), Some(PI / 2.0));
        assert!(m.unseen_hit_bearing(13.6).is_none());
        m.update(&obs_at(14.0), &w, &mut rng);
        assert!(m.threats(&obs_at(14.0), &w).is_empty());
    }

    #[test]
    fn zone_npc_and_explained_hits_make_no_ghost() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(10.0);
        o.events = vec![
            damaged(10.0, DamageKind::Zone, None, None),
            damaged(25.0, DamageKind::Npc, Some(0.0), Some(120)),
        ];
        m.update(&o, &w, &mut rng);
        assert!(m.unseen_hit_bearing(10.0).is_none());
        assert_eq!(m.last_damage_t(), 10.0);

        // Anonymous hit, but a visible mech sits on that bearing.
        let mut o = obs_at(11.0);
        o.mechs = vec![mech(4, [0.0, 80.0, 0.0], [0.0; 3])];
        o.events = vec![damaged(45.0, DamageKind::Gun, Some(PI / 2.0 + 0.2), None)];
        m.update(&o, &w, &mut rng);
        assert!(m.unseen_hit_bearing(11.0).is_none());
    }

    #[test]
    fn ghost_is_hidden_while_a_visible_mech_explains_it() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(10.0);
        o.events = vec![damaged(45.0, DamageKind::Laser, Some(0.0), Some(7))];
        m.update(&o, &w, &mut rng);
        let mut o = obs_at(10.2);
        o.mechs = vec![mech(2, [90.0, 10.0, 0.0], [0.0; 3])];
        m.update(&o, &w, &mut rng);
        let th = m.threats(&o, &w);
        assert_eq!(th.len(), 1);
        assert!(!th[0].is_ghost);
        assert!(m.unseen_hit_bearing(10.2).is_some(), "bearing itself stays");
    }

    #[test]
    fn zone_damage_is_not_engagement() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(10.0);
        o.events = vec![damaged(10.0, DamageKind::Zone, None, None)];
        m.update(&o, &w, &mut rng);
        assert!(!m.recently_engaged_or_damaged(10.0, 5.0));
        let mut o = obs_at(11.0);
        o.events = vec![damaged(25.0, DamageKind::Npc, Some(0.0), Some(120))];
        m.update(&o, &w, &mut rng);
        assert!(m.recently_engaged_or_damaged(12.0, 5.0));
    }

    #[test]
    fn own_splash_is_not_a_ghost_or_engagement() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(10.0);
        o.events = vec![damaged(80.0, DamageKind::Mortar, Some(1.0), Some(ME))];
        m.update(&o, &w, &mut rng);
        assert!(m.unseen_hit_bearing(10.0).is_none());
        assert!(!m.recently_engaged_or_damaged(10.0, 5.0));
        assert_eq!(m.last_damage_t(), 10.0);
    }

    // ---- beams ----

    #[test]
    fn beam_from_unseen_mech_creates_a_track() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(20.0);
        o.beams = vec![beam(9, [100.0, 0.0, 7.0], [0.0, 0.0, 5.0], 0.1)];
        m.update(&o, &w, &mut rng);
        let th = m.threats(&o, &w);
        assert_eq!(th.len(), 1);
        assert_eq!(th[0].id, 9);
        assert!((th[0].pos[2]).abs() < 1e-9, "muzzle height removed");
        assert!((th[0].last_beam_t - 19.9).abs() < 1e-9);
        assert!((th[0].last_seen_t - 19.9).abs() < 1e-9);
        assert!(!th[0].visible_now);
        assert!(!m.is_ffa());
        assert!(m.third_party_beam_ts.is_empty(), "aimed at us");
    }

    // ---- FFA ----

    fn run_ffa(m: &mut BotMemory, w: &World, rng: &mut Rng64, mut make: impl FnMut(f64) -> Obs, ts: &[f64]) {
        for &t in ts {
            m.update(&make(t), w, rng);
        }
    }

    #[test]
    fn ffa_latches_on_two_spaced_third_party_beams() {
        let (mut m, w, mut rng) = setup();
        let make = |t: f64| {
            let mut o = obs_at(t);
            o.mechs = vec![mech(3, [100.0, 0.0, 0.0], [0.0; 3]), mech(4, [100.0, 60.0, 0.0], [0.0; 3])];
            o
        };
        // One beam 3 -> 4, seen on two decisions with growing age: one event.
        run_ffa(
            &mut m,
            &w,
            &mut rng,
            |t| {
                let mut o = make(t);
                o.beams = vec![beam(3, [100.0, 0.0, 7.0], [100.0, 59.0, 5.0], t - 10.0)];
                o
            },
            &[10.0, 10.1],
        );
        assert_eq!(m.third_party_beam_ts.len(), 1);
        run_ffa(&mut m, &w, &mut rng, make, &[10.2, 10.3, 10.4, 10.5]);
        assert!(!m.is_ffa());
        // A second beam only 0.5 s later does not count.
        run_ffa(
            &mut m,
            &w,
            &mut rng,
            |t| {
                let mut o = make(t);
                o.beams = vec![beam(4, [100.0, 60.0, 7.0], [100.0, 1.0, 5.0], 0.0)];
                o
            },
            &[10.5],
        );
        assert_eq!(m.third_party_beam_ts.len(), 1);
        run_ffa(
            &mut m,
            &w,
            &mut rng,
            |t| {
                let mut o = make(t);
                o.beams = vec![beam(4, [100.0, 60.0, 7.0], [100.0, 1.0, 5.0], 0.0)];
                o
            },
            &[11.5],
        );
        assert_eq!(m.third_party_beam_ts.len(), 2);
        assert!(!m.is_ffa(), "needs three decisions");
        run_ffa(&mut m, &w, &mut rng, make, &[11.6, 11.7]);
        assert!(m.is_ffa());
    }

    #[test]
    fn beams_into_empty_space_or_at_us_are_not_third_party() {
        let (mut m, w, mut rng) = setup();
        for (i, t) in [10.0, 12.0, 14.0, 16.0, 18.0].into_iter().enumerate() {
            let mut o = obs_at(t);
            o.mechs = vec![mech(3, [100.0, 0.0, 0.0], [0.0; 3]), mech(4, [100.0, 60.0, 0.0], [0.0; 3])];
            let end = if i % 2 == 0 { [50.0, 50.0, 0.0] } else { [1.0, 0.0, 5.0] };
            o.beams = vec![beam(3, [100.0, 0.0, 7.0], end, 0.0)];
            m.update(&o, &w, &mut rng);
        }
        assert!(m.third_party_beam_ts.is_empty());
        assert!(!m.is_ffa());
    }

    #[test]
    fn missing_hp_alone_never_latches_ffa() {
        let (mut m, w, mut rng) = setup();
        // An enemy missing 1000 hp we never dealt: NPCs or the zone could
        // have done it, so it is no proof of a third party.
        let hurt = |t: f64| {
            let mut o = obs_at(t);
            let mut e = mech(4, [100.0, 0.0, 0.0], [0.0; 3]);
            e.hp = Some(400.0);
            o.mechs = vec![e];
            o
        };
        run_ffa(&mut m, &w, &mut rng, hurt, &[1.0, 1.1, 1.2, 1.3, 1.4, 1.5]);
        assert!(!m.is_ffa());
    }

    #[test]
    fn missing_hp_outside_a_damaging_zone_is_not_ffa() {
        let (mut m, w, mut rng) = setup();
        run_ffa(
            &mut m,
            &w,
            &mut rng,
            |t| {
                let mut o = obs_at(t);
                o.zone = ZoneObs {
                    center: [0.0, 0.0],
                    radius: 50.0,
                    dps: 20.0,
                    ..ZoneObs::default()
                };
                let mut e = mech(4, [100.0, 0.0, 0.0], [0.0; 3]);
                e.hp = Some(900.0);
                o.mechs = vec![e];
                o
            },
            &[1.0, 1.1, 1.2, 1.3],
        );
        assert!(!m.is_ffa());
    }

    #[test]
    fn ffa_latches_on_foreign_deaths_under_a_weak_zone() {
        let (mut m, w, mut rng) = setup();
        let with = |alive: u32, dps: f64| {
            move |t: f64| {
                let mut o = Obs { alive, ..obs_at(t) };
                o.zone.dps = dps;
                o
            }
        };
        // We killed one; one other death: only one foreign.
        let mut o = with(14, 0.0)(30.0);
        o.events = vec![GameEvent::Kill { target: 3 }];
        m.update(&o, &w, &mut rng);
        run_ffa(&mut m, &w, &mut rng, with(14, 0.0), &[30.1, 30.2, 30.3]);
        assert!(!m.is_ffa());
        // Two foreign deaths, but the zone hurts enough to explain them.
        run_ffa(&mut m, &w, &mut rng, with(13, 36.0), &[60.0, 60.1, 60.2]);
        assert!(!m.is_ffa());
        // Three foreign deaths, but the zone is deadly.
        run_ffa(&mut m, &w, &mut rng, with(12, 101.0), &[70.0, 70.1, 70.2]);
        assert!(!m.is_ffa());
        // Three foreign deaths under a 100 dps zone latch, late or not.
        run_ffa(&mut m, &w, &mut rng, with(12, 100.0), &[120.0, 120.1]);
        assert!(!m.is_ffa(), "needs three decisions");
        run_ffa(&mut m, &w, &mut rng, with(12, 100.0), &[120.2]);
        assert!(m.is_ffa());
        // Latched for good.
        m.update(&obs_at(130.0), &w, &mut rng);
        assert!(m.is_ffa());
    }

    #[test]
    fn two_foreign_deaths_under_a_35_dps_zone_latch() {
        let (mut m, w, mut rng) = setup();
        let make = |t: f64| {
            let mut o = Obs { alive: 14, ..obs_at(t) };
            o.zone.dps = 35.0;
            o
        };
        run_ffa(&mut m, &w, &mut rng, make, &[200.0, 200.1]);
        m.update(&obs_at(200.2), &w, &mut rng);
        run_ffa(&mut m, &w, &mut rng, make, &[200.3, 200.4]);
        assert!(!m.is_ffa(), "interrupted streak resets");
        run_ffa(&mut m, &w, &mut rng, make, &[200.5]);
        assert!(m.is_ffa());
    }

    #[test]
    fn beams_near_npcs_crates_or_short_of_a_mech_are_not_third_party() {
        let (mut m, w, mut rng) = setup();
        for (i, t) in [10.0, 12.0, 14.0, 16.0].into_iter().enumerate() {
            let mut o = obs_at(t);
            // Mech 4 stands next to an NPC and a crate sits near mech 5.
            o.mechs = vec![
                mech(3, [100.0, 0.0, 0.0], [0.0; 3]),
                mech(4, [100.0, 60.0, 0.0], [0.0; 3]),
                mech(5, [-100.0, 60.0, 0.0], [0.0; 3]),
                mech(6, [0.0, 200.0, 0.0], [0.0; 3]),
                npc(120, [105.0, 65.0, 0.0]),
            ];
            o.crates = vec![ItemObs {
                id: 1001,
                pos: [-95.0, 62.0, 0.0],
                hp: Some(80.0),
            }];
            let end = match i {
                0 => [100.0, 59.0, 5.0],  // on mech 4, but beside the NPC
                1 => [-100.0, 59.0, 5.0], // on mech 5, but beside the crate
                2 => [100.0, 54.0, 5.0],  // 6 m short of mech 4
                _ => [0.0, 199.0, 12.0],  // above mech 6's head
            };
            o.beams = vec![beam(3, [100.0, 0.0, 7.0], end, 0.0)];
            m.update(&o, &w, &mut rng);
        }
        assert!(m.third_party_beam_ts.is_empty(), "{:?}", m.third_party_beam_ts);
    }

    #[test]
    fn early_two_foreign_deaths_latch_and_alive_zero_is_ignored() {
        let (mut m, w, mut rng) = setup();
        run_ffa(&mut m, &w, &mut rng, |t| Obs { alive: 0, ..obs_at(t) }, &[5.0, 5.1, 5.2, 5.3]);
        assert!(!m.is_ffa());
        run_ffa(&mut m, &w, &mut rng, |t| Obs { alive: 14, ..obs_at(t) }, &[40.0, 40.1, 40.2]);
        assert!(m.is_ffa());
    }

    #[test]
    fn small_lobby_starting_at_t0_uses_initial_alive_count() {
        let (mut m, w, mut rng) = setup();
        run_ffa(&mut m, &w, &mut rng, |t| Obs { alive: 2, ..obs_at(t) }, &[0.0, 0.1, 0.2, 5.0, 30.0]);
        assert!(!m.is_ffa(), "2-mech duel starting at t=0 has zero foreign deaths");
    }

    // ---- kills ----

    #[test]
    fn kill_records_time_and_position() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(50.0);
        o.mechs = vec![mech(6, [40.0, 30.0, 2.0], [0.0; 3])];
        m.update(&o, &w, &mut rng);
        assert_eq!(m.last_kill_pos(), None);
        let mut o = obs_at(50.5);
        o.events = vec![GameEvent::Kill { target: 6 }];
        m.update(&o, &w, &mut rng);
        assert_eq!(m.last_kill_t(), 50.5);
        assert_eq!(m.last_kill_pos(), Some([40.0, 30.0, 2.0]));
        assert!(m.threats(&o, &w).is_empty(), "dead mech forgotten");
        assert_eq!(m.my_kills, 1);
    }

    // ---- items ----

    fn item_obs(t: f64) -> Obs {
        let mut o = obs_at(t);
        o.cubes = vec![ItemObs {
            id: 5001,
            pos: [20.0, 0.0, 0.0],
            hp: None,
        }];
        o.crates = vec![ItemObs {
            id: 1001,
            pos: [200.0, 0.0, 0.0],
            hp: Some(80.0),
        }];
        o.pickups = vec![PickupObs {
            id: 8001,
            kind: WeaponKind::Laser,
            pos: [0.0, 150.0, 0.0],
        }];
        o.drops = vec![DropObs {
            id: 20001,
            pos: [-150.0, 0.0, 0.0],
            radius: 10.0,
            lands_in: 0.0,
            capture_s: 5.0,
            progress: 0.0,
            holder: None,
        }];
        o.mechs = vec![npc(110, [0.0, -150.0, 0.0])];
        o
    }

    fn item(m: &BotMemory, id: u32) -> Option<RememberedItem> {
        m.items().iter().find(|i| i.id == id).copied()
    }

    #[test]
    fn items_are_remembered_with_kind_and_details() {
        let (mut m, w, mut rng) = setup();
        m.update(&item_obs(1.0), &w, &mut rng);
        assert_eq!(m.items().len(), 5);
        assert_eq!(item(&m, 5001).unwrap().kind, ItemKind::Cube);
        let c = item(&m, 1001).unwrap();
        assert_eq!((c.kind, c.hp), (ItemKind::Crate, Some(80.0)));
        let p = item(&m, 8001).unwrap();
        assert_eq!((p.kind, p.weapon_kind), (ItemKind::Weapon, Some(WeaponKind::Laser)));
        assert_eq!(item(&m, 20001).unwrap().kind, ItemKind::Drop);
        let n = item(&m, 110).unwrap();
        assert_eq!((n.kind, n.hp), (ItemKind::Npc, Some(350.0)));
        assert_eq!(n.last_seen_t, 1.0);
    }

    #[test]
    fn cube_we_should_see_is_deleted_at_once() {
        let (mut m, w, mut rng) = setup();
        m.update(&item_obs(1.0), &w, &mut rng);
        m.update(&obs_at(1.1), &w, &mut rng);
        assert!(item(&m, 5001).is_none(), "cube 20 m away in plain view");
        assert!(item(&m, 8001).is_some(), "pickup outside the yaw wedge stays");
        assert!(item(&m, 1001).is_some(), "far crate stays");
        assert!(item(&m, 110).is_some(), "far NPC stays");
    }

    #[test]
    fn a_sighting_resets_the_miss_count() {
        let (mut m, w, mut rng) = setup();
        let with_npc = |t: f64| {
            let mut o = obs_at(t);
            o.mechs = vec![npc(115, [20.0, 0.0, 0.0])];
            o
        };
        m.update(&with_npc(1.0), &w, &mut rng);
        m.update(&obs_at(1.1), &w, &mut rng);
        m.update(&with_npc(1.2), &w, &mut rng);
        m.update(&obs_at(1.3), &w, &mut rng);
        assert!(item(&m, 115).is_some(), "NPC items still need two misses");
        m.update(&obs_at(1.4), &w, &mut rng);
        assert!(item(&m, 115).is_none());
    }

    #[test]
    fn far_pickup_in_view_is_deleted_at_once_and_out_of_view_kept() {
        let (mut m, w, mut rng) = setup();
        let pickup = |id: u32, pos: [f64; 3]| PickupObs {
            id,
            kind: WeaponKind::Gun,
            pos,
        };
        let mut o = obs_at(1.0);
        o.pickups = vec![
            pickup(8101, [150.0, 0.0, 0.0]),  // dead ahead
            pickup(8102, [0.0, -150.0, 0.0]), // 90 degrees off
            pickup(8103, [195.0, 0.0, 0.0]),  // past range - 8
        ];
        m.update(&o, &w, &mut rng);
        let o = obs_at(1.1);
        assert!(m.should_see_item(&o, &w, m.eye(&o), [150.0, 0.0, 0.0]));
        assert!(!m.should_see_item(&o, &w, m.eye(&o), [0.0, -150.0, 0.0]));
        m.update(&o, &w, &mut rng);
        assert!(item(&m, 8101).is_none());
        assert!(item(&m, 8102).is_some());
        assert!(item(&m, 8103).is_some());
        // Still listed: kept.
        let mut o = obs_at(1.2);
        o.pickups = vec![pickup(8102, [0.0, -150.0, 0.0])];
        o.me.aim = [-PI / 2.0, 0.0];
        m.update(&o, &w, &mut rng);
        assert!(item(&m, 8102).is_some());
    }

    #[test]
    fn vision_bonus_widens_what_we_should_see() {
        let (m, w, _) = setup();
        let mut o = obs_at(1.0);
        o.me.aim = [PI, 0.0];
        let eye = m.eye(&o);
        // 40 m behind us: past near range - 8 = 37.
        assert!(!m.should_see_item(&o, &w, eye, [40.0, 0.0, 0.0]));
        o.me.stats.near_range_bonus = 15.0;
        assert!(m.should_see_item(&o, &w, eye, [40.0, 0.0, 0.0]));
    }

    #[test]
    fn close_item_behind_a_wall_is_kept() {
        let (mut m, _, mut rng) = setup();
        let w = walled();
        let mut o = obs_at(1.0);
        o.cubes = vec![ItemObs {
            id: 5002,
            pos: [25.0, 0.0, 0.0],
            hp: None,
        }];
        m.update(&o, &w, &mut rng);
        for i in 1..6 {
            m.update(&obs_at(1.0 + 0.1 * f64::from(i)), &w, &mut rng);
        }
        assert!(item(&m, 5002).is_some());
    }

    #[test]
    fn items_expire_by_kind() {
        let (mut m, _, mut rng) = setup();
        let w = walled(); // keep line of sight out of the picture for x > 20
        let mut o = item_obs(0.0);
        o.cubes[0].pos = [30.0, 0.0, 0.0];
        m.update(&o, &w, &mut rng);
        m.update(&obs_at(30.5), &w, &mut rng);
        assert!(item(&m, 110).is_none(), "NPC after 30 s");
        assert!(item(&m, 5001).is_some());
        m.update(&obs_at(40.5), &w, &mut rng);
        assert!(item(&m, 5001).is_none(), "cube after 40 s");
        assert!(item(&m, 8001).is_some());
        m.update(&obs_at(60.5), &w, &mut rng);
        assert!(item(&m, 8001).is_none(), "weapon after 60 s");
        assert!(item(&m, 1001).is_some(), "crates do not expire");
    }

    #[test]
    fn taken_items_are_forgotten() {
        let (mut m, w, mut rng) = setup();
        m.update(&item_obs(1.0), &w, &mut rng);
        let mut o = obs_at(1.1);
        o.events = vec![
            GameEvent::Took {
                id: 8001,
                kind: WeaponKind::Laser,
                slot: 1,
            },
            GameEvent::TakeFailed {
                id: 20001,
                reason: TakeFailReason::Gone,
            },
            GameEvent::TakeFailed {
                id: 1001,
                reason: TakeFailReason::Range,
            },
        ];
        m.update(&o, &w, &mut rng);
        assert!(item(&m, 8001).is_none() && item(&m, 20001).is_none());
        assert!(item(&m, 1001).is_some());
    }

    #[test]
    fn blacklist_and_chase_timer() {
        let (mut m, w, mut rng) = setup();
        m.blacklist_item(42, 10.0);
        assert!(m.is_blacklisted(42, 9.9));
        assert!(!m.is_blacklisted(42, 10.0));
        assert!(!m.is_blacklisted(43, 0.0));

        // Standing 5 m from a cube that is never picked up.
        for i in 0..=90 {
            let t = 20.0 + 0.1 * f64::from(i);
            let mut o = obs_at(t);
            o.cubes = vec![ItemObs {
                id: 5005,
                pos: [5.0, 0.0, 0.0],
                hp: None,
            }];
            m.update(&o, &w, &mut rng);
            if t < 27.9 {
                assert!(!m.is_blacklisted(5005, t), "t={t}");
            }
        }
        assert!(m.is_blacklisted(5005, 29.0));
        assert!(!m.is_blacklisted(5005, 54.0));
        // The expired entry is pruned on update.
        m.update(&obs_at(60.0), &w, &mut rng);
        assert!(!m.blacklist.contains_key(&42));
    }

    // ---- sounds ----

    fn sound(kind: SoundKind, bearing: f64, dist: SoundDist, age: f64) -> SoundObs {
        SoundObs { kind, bearing, dist, age }
    }

    #[test]
    fn best_sound_prefers_near_combat_and_expires() {
        let (mut m, w, mut rng) = setup();
        let mut o = obs_at(10.0);
        o.sounds = vec![
            sound(SoundKind::Step, 1.0, SoundDist::Far, 0.0),
            sound(SoundKind::Gun, 2.0, SoundDist::Mid, 0.0),
            sound(SoundKind::NpcGun, 3.0, SoundDist::Near, 0.0),
            sound(SoundKind::Drop, 0.5, SoundDist::Near, 0.0),
        ];
        m.update(&o, &w, &mut rng);
        assert_eq!(m.best_sound(10.0), Some((2.0, SoundDist::Mid, SoundKind::Gun)));
        let mut o = obs_at(11.0);
        o.sounds = vec![sound(SoundKind::Laser, -1.0, SoundDist::Near, 0.2)];
        m.update(&o, &w, &mut rng);
        assert_eq!(m.best_sound(11.0).map(|s| s.0), Some(-1.0));
        // The laser was made at 10.8 s.
        assert_eq!(m.best_sound(16.7).map(|s| s.2), Some(SoundKind::Laser));
        assert!(m.best_sound(16.9).is_none());
        m.update(&obs_at(20.0), &w, &mut rng);
        assert!(m.sounds.is_empty());
    }

    #[test]
    fn repeated_sound_reports_are_deduplicated() {
        let (mut m, w, mut rng) = setup();
        for t in [5.0, 5.1] {
            let mut o = obs_at(t);
            o.sounds = vec![sound(SoundKind::Gun, 1.0, SoundDist::Far, t - 4.9)];
            m.update(&o, &w, &mut rng);
        }
        assert_eq!(m.sounds.len(), 1);
    }

    #[test]
    fn update_is_deterministic() {
        let run = || {
            let (mut m, w, mut rng) = setup();
            for i in 0..200 {
                let t = 0.1 * f64::from(i);
                let mut o = obs_at(t);
                o.mechs = vec![mech(3, [100.0, 0.0, 0.0], [f64::from(i % 7), 0.0, 0.0])];
                o.me.weapons = [Some(gun(f64::from(12 - (i % 13)))), None];
                m.update(&o, &w, &mut rng);
            }
            (m.hit_bandit().shots, m.hit_bandit().active, m.track_accel(3))
        };
        assert_eq!(run(), run());
    }
}
