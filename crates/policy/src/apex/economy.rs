//! Economy helpers for the apex bot: which upgrade to take, which weapon to
//! pick up or drop, which item, cube, drop, or farm target to go for, and how
//! urgently to head for the zone.
//!
//! The public functions take the concrete [`ApexNav`] and [`BotMemory`]; the
//! logic lives in private functions generic over [`NavView`] and [`MemView`]
//! so it can be tested without a map. The two `impl` blocks below are the only
//! place that touches those types' methods.

use super::genome::{ApexGenome, DropContest, UpgradeProfile, WeaponPref};
use super::memory::{BotMemory, RememberedItem};
use super::nav::ApexNav;
use super::types::{clamp, dist2d, ItemGoal, ItemKind, ThreatInfo};
use proto::{ActorKind, Obs, WeaponKind, WeaponObs};

/// Horizontal reach of a weapon pickup, meters (`loot.pickup_radius`).
const PICKUP_R: f64 = 4.0;
/// Vertical reach of a weapon pickup, meters.
const PICKUP_DZ: f64 = 3.7;
/// Items this far inside the current zone edge, or deeper, are allowed.
const ZONE_ITEM_MARGIN: f64 = 6.0;
/// A height difference above this needs a reachability check.
const LEVEL_DZ: f64 = 3.5;
/// How long a remembered weapon pickup stays worth walking to, seconds.
const WEAPON_FRESH_S: f64 = 55.0;
/// How long a remembered cube stays worth walking to, seconds.
const CUBE_FRESH_S: f64 = 35.0;
/// How long a remembered crate stays worth walking to, seconds.
const CRATE_FRESH_S: f64 = 45.0;
/// Score bonus for a laser or lightning while our second slot is empty.
const UNARMED_SECONDARY_BONUS: f64 = 120.0;
/// Fallback `step_up` when the stats leave it unset, meters.
const STEP_UP_FALLBACK: f64 = 1.2;
/// Height above both our feet and the terrain, beyond `step_up`, that marks
/// an item as on a roof we cannot walk onto, meters.
const ROOF_SLACK: f64 = 0.5;
/// Height above terrain beyond which any item/pickup/cube is considered on an
/// unreachable roof, meters.
const ROOF_GROUND_DZ: f64 = 3.2;
/// Calm-loot distance caps in practice, meters.
const PRACTICE_CUBE_CAP: f64 = 110.0;
const PRACTICE_CRATE_CAP: f64 = 100.0;
const PRACTICE_NPC_CAP: f64 = 110.0;
const PRACTICE_WEAPON_CAP: f64 = 150.0;
/// Calm-loot distance caps in stealth / FFA when unarmed (no laser), meters.
const FFA_UNARMED_CUBE_CAP: f64 = 80.0;
const FFA_UNARMED_CRATE_CAP: f64 = 75.0;
const FFA_UNARMED_NPC_CAP: f64 = 75.0;
const FFA_WEAPON_CAP: f64 = 120.0;
/// Calm-loot distance caps in stealth / FFA once armed with a Laser, meters.
const FFA_ARMED_CUBE_CAP: f64 = 65.0;
const FFA_ARMED_CRATE_CAP: f64 = 60.0;
const FFA_ARMED_NPC_CAP: f64 = 65.0;
/// Laser seek radius ceiling in stealth / free-for-all, meters.
const FFA_LASER_CAP: f64 = 240.0;
/// A threat seen this recently, and nearer the item by this margin, wins the race.
const RACE_SEEN_S: f64 = 5.0;
const RACE_MARGIN: f64 = 2.0;
/// Free-for-all: calm loot whose straight path passes this close to a threat
/// seen within `CORRIDOR_SEEN_S` is skipped, meters.
const CORRIDOR_R: f64 = 55.0;
const CORRIDOR_SEEN_S: f64 = 4.0;
/// Seconds before the shrink starts from which calm loot must lie inside the
/// next circle, and how far inside, meters.
const NEXT_ZONE_GATE_S: f64 = 25.0;
const NEXT_ZONE_MARGIN: f64 = 10.0;

/// What the economy needs from the map.
trait NavView {
    /// Whether `to` can be walked to from `from` (same or main component).
    fn reachable(&self, from: [f64; 3], to: [f64; 3]) -> bool;
    /// Ground height to stand on at `(x, y)`.
    fn ground_z(&self, x: f64, y: f64) -> f64;
    /// Whether `(x, y)` is walkable in the main connected component.
    fn is_main_xy(&self, x: f64, y: f64) -> bool;
}

/// What the economy needs from the bot's memory.
trait MemView {
    fn blacklisted(&self, id: u32, t: f64) -> bool;
    /// Match time of our last kill (`-inf` if none).
    fn kill_t(&self) -> f64;
    /// Where our last kill died, if known.
    fn kill_pos(&self) -> Option<[f64; 3]>;
    /// Items seen recently, including ones out of view now.
    fn items(&self) -> &[RememberedItem];
    /// True once a free-for-all has been detected.
    fn is_ffa(&self) -> bool;
}

impl NavView for ApexNav {
    fn reachable(&self, from: [f64; 3], to: [f64; 3]) -> bool {
        // Cheap per-item check: the target sits at (or near) ground level, not
        // on a roof, on the main walkable area or in the same area as us.
        let ours = self.area_of(from[0], from[1]);
        let same_area = ours.is_some() && ours == self.area_of(to[0], to[1]);
        (self.is_main_xy(to[0], to[1]) || same_area) && to[2] <= self.ground_at(to[0], to[1]) + LEVEL_DZ
    }
    fn ground_z(&self, x: f64, y: f64) -> f64 {
        self.ground_at(x, y)
    }
    fn is_main_xy(&self, x: f64, y: f64) -> bool {
        ApexNav::is_main_xy(self, x, y)
    }
}

impl MemView for BotMemory {
    fn blacklisted(&self, id: u32, t: f64) -> bool {
        self.is_blacklisted(id, t)
    }
    fn kill_t(&self) -> f64 {
        self.last_kill_t()
    }
    fn kill_pos(&self) -> Option<[f64; 3]> {
        self.last_kill_pos()
    }
    fn items(&self) -> &[RememberedItem] {
        self.items()
    }
    fn is_ffa(&self) -> bool {
        BotMemory::is_ffa(self)
    }
}

// ---------------------------------------------------------------- upgrades

const DEFENSE_FIRST: &[&str] = &[
    "plating",
    "armor",
    "nanites",
    "laser_damage",
    "gun_damage",
    "laser_charge",
    "gun_rate",
    "lightning_efficiency",
    "gyros",
    "servos",
    "gun_mag",
    "gun_velocity",
    "capacitor",
    "reactor",
    "lightning_range",
    "lightning_chain",
    "jets",
    "mortar_splash",
    "mortar_reload",
    "mortar_velocity",
    "sensors",
    "laser_pierce",
];

const LASER_ECON_HEAD: &[&str] = &[
    "plating",
    "armor",
    "laser_damage",
    "laser_charge",
    "nanites",
    "lightning_efficiency",
    "gun_damage",
    "gun_rate",
];

const GUN_MAG_HEAD: &[&str] = &[
    "plating",
    "gun_mag",
    "armor",
    "gun_damage",
    "gun_rate",
    "laser_damage",
    "nanites",
    "gun_velocity",
    "servos",
    "gyros",
];

/// Position of `id` in `head` followed by the rest of [`DEFENSE_FIRST`].
fn order_rank(head: &[&str], id: &str) -> Option<usize> {
    if let Some(i) = head.iter().position(|h| *h == id) {
        return Some(i);
    }
    DEFENSE_FIRST
        .iter()
        .filter(|d| !head.contains(d))
        .position(|d| *d == id)
        .map(|i| head.len() + i)
}

/// The weapon an upgrade id applies to, from its prefix.
fn upgrade_weapon(id: &str) -> Option<WeaponKind> {
    if id.starts_with("gun_") {
        Some(WeaponKind::Gun)
    } else if id.starts_with("laser_") {
        Some(WeaponKind::Laser)
    } else if id.starts_with("lightning_") {
        Some(WeaponKind::Lightning)
    } else if id.starts_with("mortar_") {
        Some(WeaponKind::Mortar)
    } else {
        None
    }
}

fn holds(obs: &Obs, kind: WeaponKind) -> bool {
    obs.me.weapons.iter().flatten().any(|w| w.kind() == kind)
}

fn hp_frac(obs: &Obs) -> f64 {
    if obs.me.max_hp > 0.0 {
        obs.me.hp / obs.me.max_hp
    } else {
        1.0
    }
}

/// Index of the offered upgrade to take, by `genome.upgrade_profile`. `None`
/// only when there is no offer or it has no choices.
pub fn pick_upgrade(obs: &Obs, genome: &ApexGenome) -> Option<u8> {
    let offer = obs.offer.as_ref()?;
    if offer.choices.is_empty() {
        return None;
    }
    if genome.upgrade_profile == UpgradeProfile::SmartHealDefense && hp_frac(obs) < 0.65 {
        if let Some(i) = offer.choices.iter().position(|c| c.id == "plating") {
            return Some(i as u8);
        }
    }
    let head: &[&str] = match genome.upgrade_profile {
        UpgradeProfile::DefenseFirst | UpgradeProfile::SmartHealDefense => &[],
        UpgradeProfile::LaserEcon => LASER_ECON_HEAD,
        UpgradeProfile::GunMagBrawl => GUN_MAG_HEAD,
    };
    let score = |id: &str| -> usize {
        let base = order_rank(head, id).unwrap_or(500);
        let unheld = upgrade_weapon(id).is_some_and(|k| !holds(obs, k));
        base + if unheld { 1000 } else { 0 }
    };
    let best = offer.choices.iter().enumerate().min_by_key(|(i, c)| (score(&c.id), *i)).map(|(i, _)| i)?;
    Some(best as u8)
}

// ----------------------------------------------------------------- weapons

/// How much `pref` values holding `kind`; 0 means never take it.
pub(super) fn weapon_rank(pref: WeaponPref, kind: WeaponKind) -> u32 {
    use WeaponKind::{Gun, Laser, Lightning, Mortar};
    match (pref, kind) {
        (_, Mortar) => 0,
        (WeaponPref::GunLaser, Gun) => 99,
        (WeaponPref::GunLaser, Laser) => 10,
        (WeaponPref::GunLaser, Lightning) => 5,
        (WeaponPref::LaserLightningDropGun, Laser) => 10,
        (WeaponPref::LaserLightningDropGun, Lightning) => 8,
        (WeaponPref::LaserLightningDropGun, Gun) => 3,
        (WeaponPref::GunLightning, Gun) => 99,
        (WeaponPref::GunLightning, Lightning) => 10,
        (WeaponPref::GunLightning, Laser) => 6,
    }
}

/// The best non-gun weapon for `pref`: what an early rush goes for.
fn top_wanted(pref: WeaponPref) -> WeaponKind {
    match pref {
        WeaponPref::GunLaser | WeaponPref::LaserLightningDropGun => WeaponKind::Laser,
        WeaponPref::GunLightning => WeaponKind::Lightning,
    }
}

fn laser_charging(w: &WeaponObs) -> bool {
    matches!(w, WeaponObs::Laser { charge, .. } if *charge > 0.0)
}

/// How to take a `kind` pickup: `Some(None)` into an empty slot,
/// `Some(Some(slot))` replacing `slot`, `None` if it is not worth taking.
fn take_plan(obs: &Obs, pref: WeaponPref, kind: WeaponKind) -> Option<Option<u8>> {
    let rank = weapon_rank(pref, kind);
    if rank == 0 || holds(obs, kind) {
        return None;
    }
    if obs.me.weapons.iter().any(Option::is_none) {
        return Some(None);
    }
    let (slot, held) = obs
        .me
        .weapons
        .iter()
        .enumerate()
        .filter_map(|(i, w)| w.as_ref().map(|w| (i, w)))
        .min_by_key(|(i, w)| (weapon_rank(pref, w.kind()), *i))?;
    // Never throw away a laser mid-charge.
    if laser_charging(held) || rank <= weapon_rank(pref, held.kind()) {
        return None;
    }
    Some(Some(slot as u8))
}

/// `(take_pickup_id, drop_slot)` for a pickup within reach, by `genome.weapon_pref`.
pub fn pick_weapon_action(obs: &Obs, genome: &ApexGenome) -> (Option<u32>, Option<u8>) {
    let me = obs.me.pos;
    let best = obs
        .pickups
        .iter()
        .filter(|p| dist2d(p.pos, me) <= PICKUP_R && (p.pos[2] - me[2]).abs() <= PICKUP_DZ)
        .filter_map(|p| take_plan(obs, genome.weapon_pref, p.kind).map(|plan| (p, plan)))
        .max_by(|(a, _), (b, _)| {
            let ra = weapon_rank(genome.weapon_pref, a.kind);
            let rb = weapon_rank(genome.weapon_pref, b.kind);
            ra.cmp(&rb).then(dist2d(b.pos, me).total_cmp(&dist2d(a.pos, me)))
        });
    match best {
        Some((p, drop)) => (Some(p.id), drop),
        None => (None, None),
    }
}

// -------------------------------------------------------------------- loot

fn walk_speed(obs: &Obs) -> f64 {
    obs.me.stats.walk_speed.max(1.0)
}

fn step_up(obs: &Obs) -> f64 {
    let s = obs.me.stats.step_up;
    if s > 0.0 {
        s
    } else {
        STEP_UP_FALLBACK
    }
}

/// True for an item above both us and the terrain under it by more than a
/// step: it sits on a roof, and walking to it pins us against the wall.
fn on_roof<N: NavView>(obs: &Obs, nav: &N, pos: [f64; 3]) -> bool {
    let lim = step_up(obs) + ROOF_SLACK;
    pos[2] - obs.me.pos[2] > lim && pos[2] - nav.ground_z(pos[0], pos[1]) > lim
}

/// Whether an item is allowed at all: not blacklisted, well inside the zone,
/// not on a roof (`z <= ground_z + 3.2` and `on_roof == false`), on the main
/// walkable area, and on our level or reachable.
fn item_ok<N: NavView, M: MemView>(obs: &Obs, nav: &N, mem: &M, id: u32, pos: [f64; 3]) -> bool {
    if mem.blacklisted(id, obs.t) {
        return false;
    }
    let z = &obs.zone;
    if dist2d(pos, [z.center[0], z.center[1], 0.0]) > z.radius - ZONE_ITEM_MARGIN {
        return false;
    }
    if pos[2] > nav.ground_z(pos[0], pos[1]) + ROOF_GROUND_DZ || !nav.is_main_xy(pos[0], pos[1]) {
        return false;
    }
    if on_roof(obs, nav, pos) {
        return false;
    }
    (pos[2] - obs.me.pos[2]).abs() <= LEVEL_DZ || nav.reachable(obs.me.pos, pos)
}

/// A real threat seen in the last 5 s is nearer `pos` than we are (`dist`):
/// it gets there first.
fn raced(obs: &Obs, threats: &[ThreatInfo], pos: [f64; 3], dist: f64) -> bool {
    threats
        .iter()
        .any(|th| !th.is_ghost && obs.t - th.last_seen_t <= RACE_SEEN_S && dist2d(th.pos, pos) < dist - RACE_MARGIN)
}

/// Horizontal distance from `p` to the segment `a -> b`.
fn seg_dist2d(a: [f64; 3], b: [f64; 3], p: [f64; 3]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let u = if len2 > 1e-12 {
        clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2, 0.0, 1.0)
    } else {
        0.0
    };
    (a[0] + u * dx - p[0]).hypot(a[1] + u * dy - p[1])
}

/// The straight walk to `pos` passes near a real threat seen within `CORRIDOR_SEEN_S`,
/// nearer than the threat already is to us.
fn corridor_blocked(obs: &Obs, threats: &[ThreatInfo], pos: [f64; 3]) -> bool {
    let me = obs.me.pos;
    threats
        .iter()
        .any(|th| !th.is_ghost && obs.t - th.last_seen_t < CORRIDOR_SEEN_S && seg_dist2d(me, pos, th.pos) < CORRIDOR_R.min(th.dist - RACE_MARGIN))
}

/// Within 25 s of the shrink, calm loot must lie 10 m inside the next circle.
fn next_zone_ok(obs: &Obs, pos: [f64; 3]) -> bool {
    let z = &obs.zone;
    if z.next_radius <= 0.0 || z.shrink_start - obs.t >= NEXT_ZONE_GATE_S {
        return true;
    }
    dist2d(pos, [z.next_center[0], z.next_center[1], 0.0]) <= z.next_radius - NEXT_ZONE_MARGIN
}

/// A drop is zone-safe when it is inside the current zone and, if the shrink
/// starts before we could finish capturing, inside the next circle too.
fn drop_zone_safe(obs: &Obs, pos: [f64; 3], eta: f64) -> bool {
    let z = &obs.zone;
    let cur = [z.center[0], z.center[1], 0.0];
    let next = [z.next_center[0], z.next_center[1], 0.0];
    if dist2d(pos, cur) > z.radius - ZONE_ITEM_MARGIN {
        return false;
    }
    let to_shrink = z.shrink_start - obs.t;
    to_shrink > eta + 10.0 || dist2d(pos, next) <= z.next_radius
}

/// Remembered items of `kind` seen within `fresh_s`, skipping ids in `seen`
/// (the visible copy wins).
fn remembered<'a, M: MemView>(obs: &Obs, mem: &'a M, kind: ItemKind, fresh_s: f64, seen: &[u32]) -> impl Iterator<Item = &'a RememberedItem> {
    let t = obs.t;
    let seen = seen.to_vec();
    mem.items()
        .iter()
        .filter(move |it| it.kind == kind && t - it.last_seen_t <= fresh_s && !seen.contains(&it.id))
}

/// Weapon pickups in view plus fresh remembered ones: `(id, kind, pos)`.
fn weapon_candidates<M: MemView>(obs: &Obs, mem: &M) -> Vec<(u32, WeaponKind, [f64; 3])> {
    let mut out: Vec<_> = obs.pickups.iter().map(|p| (p.id, p.kind, p.pos)).collect();
    let seen: Vec<u32> = out.iter().map(|c| c.0).collect();
    out.extend(remembered(obs, mem, ItemKind::Weapon, WEAPON_FRESH_S, &seen).filter_map(|it| it.weapon_kind.map(|k| (it.id, k, it.pos))));
    out
}

/// Cubes (or crates) in view plus fresh remembered ones: `(id, pos)`.
fn item_candidates<M: MemView>(obs: &Obs, mem: &M, kind: ItemKind) -> Vec<(u32, [f64; 3])> {
    let (visible, fresh_s) = match kind {
        ItemKind::Crate => (&obs.crates, CRATE_FRESH_S),
        _ => (&obs.cubes, CUBE_FRESH_S),
    };
    let mut out: Vec<_> = visible.iter().map(|c| (c.id, c.pos)).collect();
    let seen: Vec<u32> = out.iter().map(|c| c.0).collect();
    out.extend(remembered(obs, mem, kind, fresh_s, &seen).map(|it| (it.id, it.pos)));
    out
}

fn better(best: &mut Option<ItemGoal>, g: ItemGoal) {
    if best.is_none_or(|b| g.score > b.score) {
        *best = Some(g);
    }
}

fn pick_drop<N: NavView, M: MemView>(obs: &Obs, nav: &N, mem: &M, genome: &ApexGenome, threats: &[ThreatInfo]) -> Option<ItemGoal> {
    let (max_d, slack) = match genome.drop_contest {
        DropContest::Off => return None,
        DropContest::HealOnly95 => {
            if hp_frac(obs) >= 0.95 {
                return None;
            }
            (160.0, 6.0)
        }
        DropContest::AlwaysUncontested => (280.0, 6.0),
    };
    let me = obs.me.pos;
    let mut best = None;
    for d in &obs.drops {
        let dist = dist2d(d.pos, me);
        if dist > max_d || !item_ok(obs, nav, mem, d.id, d.pos) {
            continue;
        }
        let ours_or_free = d.holder.is_none() || d.holder == Some(obs.me.id);
        let contest_ok = match genome.drop_contest {
            DropContest::HealOnly95 => (ours_or_free || d.progress < 0.5 * d.capture_s) && !threats.iter().any(|t| !t.is_ghost && dist2d(t.pos, d.pos) < 70.0),
            _ => ours_or_free,
        };
        let eta = dist / walk_speed(obs);
        if !contest_ok || d.lands_in > eta + slack || !drop_zone_safe(obs, d.pos, eta) {
            continue;
        }
        better(
            &mut best,
            ItemGoal {
                kind: ItemKind::Drop,
                id: d.id,
                pos: d.pos,
                score: 1000.0 - dist,
                stand_off: 0.0,
            },
        );
    }
    best
}

fn pick_kill_cube<N: NavView, M: MemView>(obs: &Obs, nav: &N, mem: &M, genome: &ApexGenome) -> Option<ItemGoal> {
    if obs.t - mem.kill_t() > 4.0 || obs.me.cubes >= genome.cube_cap {
        return None;
    }
    let kill = mem.kill_pos().unwrap_or(obs.me.pos);
    let me = obs.me.pos;
    let mut best = None;
    for (id, pos) in item_candidates(obs, mem, ItemKind::Cube) {
        if dist2d(pos, kill) > 95.0 || (pos[2] - kill[2]).abs() >= 4.0 || !item_ok(obs, nav, mem, id, pos) {
            continue;
        }
        let dist = dist2d(pos, me);
        better(
            &mut best,
            ItemGoal {
                kind: ItemKind::Cube,
                id,
                pos,
                score: 900.0 - dist,
                stand_off: 0.0,
            },
        );
    }
    best
}

fn pick_fight_detour<N: NavView, M: MemView>(obs: &Obs, nav: &N, mem: &M, genome: &ApexGenome, threats: &[ThreatInfo]) -> Option<ItemGoal> {
    let me = obs.me.pos;
    let nearest = threats.iter().min_by(|a, b| a.dist.total_cmp(&b.dist));
    let mut best = None;
    if obs.me.cubes < genome.cube_cap && genome.fight_cube_r > 0.0 {
        let r = if hp_frac(obs) < 0.5 { genome.fight_cube_r * 0.5 } else { genome.fight_cube_r };
        for (id, pos) in item_candidates(obs, mem, ItemKind::Cube) {
            let dist = dist2d(pos, me);
            // Safer than the nearest threat: we reach it before they do.
            let safe = nearest.is_none_or(|t| dist < dist2d(pos, t.pos));
            if dist > r || !safe || !item_ok(obs, nav, mem, id, pos) {
                continue;
            }
            better(
                &mut best,
                ItemGoal {
                    kind: ItemKind::Cube,
                    id,
                    pos,
                    score: 500.0 - dist,
                    stand_off: 0.0,
                },
            );
        }
    }
    // Gun-only with a lull in the fight: a wanted laser is worth a longer walk.
    let gun_only = !holds(obs, WeaponKind::Laser) && !holds(obs, WeaponKind::Lightning);
    let lull = !threats.iter().any(|t| t.dist < 50.0 || obs.t - t.last_hit_us_t < 2.0);
    let laser_r = if gun_only && lull { genome.laser_seek_r.min(120.0) } else { 0.0 };
    for (id, kind, pos) in weapon_candidates(obs, mem) {
        let dist = dist2d(pos, me);
        let reach = if kind == WeaponKind::Laser { laser_r.max(45.0) } else { 45.0 };
        if dist > reach || take_plan(obs, genome.weapon_pref, kind).is_none() || raced(obs, threats, pos, dist) || !item_ok(obs, nav, mem, id, pos) {
            continue;
        }
        let score = 520.0 + f64::from(weapon_rank(genome.weapon_pref, kind)) - dist;
        better(
            &mut best,
            ItemGoal {
                kind: ItemKind::Weapon,
                id,
                pos,
                score,
                stand_off: 0.0,
            },
        );
    }
    best
}

/// No enemy mech within `r` meters, seen within `seen_s` seconds (or visible now).
fn quiet_within(obs: &Obs, threats: &[ThreatInfo], r: f64, seen_s: f64) -> bool {
    let me = obs.me.pos;
    let recent = threats.iter().any(|t| obs.t - t.last_seen_t <= seen_s && dist2d(t.pos, me) <= r);
    let visible = obs.mechs.iter().any(|m| m.is_mech() && m.id != obs.me.id && dist2d(m.pos, me) <= r);
    !recent && !visible
}

/// Whether a farm target sits safely inside the zone we are heading for.
/// Early (shrink more than 30 s away) the current circle with a 15 m margin
/// also counts.
fn farm_zone_ok(obs: &Obs, pos: [f64; 3]) -> bool {
    let z = &obs.zone;
    let next = [z.next_center[0], z.next_center[1], 0.0];
    if dist2d(pos, next) <= z.next_radius - 10.0 {
        return true;
    }
    let early = z.shrink_start - obs.t > 30.0;
    early && dist2d(pos, [z.center[0], z.center[1], 0.0]) <= z.radius - 15.0
}

fn pick_calm<N: NavView, M: MemView>(obs: &Obs, nav: &N, mem: &M, genome: &ApexGenome, threats: &[ThreatInfo]) -> Option<ItemGoal> {
    let me = obs.me.pos;
    let pref = genome.weapon_pref;
    let top = top_wanted(pref);
    let stealth = mem.is_ffa() || genome.quiet_walk;
    let has_laser = holds(obs, WeaponKind::Laser);
    let (cube_cap, crate_cap, npc_cap, weapon_cap, laser_cap) = if !stealth {
        (
            PRACTICE_CUBE_CAP,
            PRACTICE_CRATE_CAP,
            PRACTICE_NPC_CAP,
            PRACTICE_WEAPON_CAP,
            genome.laser_seek_r,
        )
    } else if has_laser {
        (
            FFA_ARMED_CUBE_CAP,
            FFA_ARMED_CRATE_CAP,
            FFA_ARMED_NPC_CAP,
            FFA_WEAPON_CAP,
            genome.laser_seek_r.min(FFA_LASER_CAP),
        )
    } else {
        (
            FFA_UNARMED_CUBE_CAP,
            FFA_UNARMED_CRATE_CAP,
            FFA_UNARMED_NPC_CAP,
            FFA_WEAPON_CAP,
            genome.laser_seek_r.min(FFA_LASER_CAP),
        )
    };
    // Only a gun (or nothing) in hand: grab any secondary fast.
    let unarmed = obs.me.weapons[1].is_none();
    // In stealth/FFA, never walk past a threat for loot or weapons.
    let exposed = |pos: [f64; 3]| stealth && corridor_blocked(obs, threats, pos);
    let mut best = None;
    for (id, kind, pos) in weapon_candidates(obs, mem) {
        let dist = dist2d(pos, me);
        let is_laser = kind == WeaponKind::Laser;
        let max_d = if is_laser { laser_cap } else { weapon_cap };
        if dist > max_d || take_plan(obs, pref, kind).is_none() || !next_zone_ok(obs, pos) || !item_ok(obs, nav, mem, id, pos) {
            continue;
        }
        if is_laser && raced(obs, threats, pos, dist) {
            continue;
        }
        if exposed(pos) {
            continue;
        }
        let base = if kind == top && dist <= laser_cap {
            // Early rush for the weapon the build is centred on.
            800.0 - 0.5 * dist
        } else {
            300.0 + f64::from(weapon_rank(pref, kind)) - dist
        };
        let secondary = matches!(kind, WeaponKind::Laser | WeaponKind::Lightning);
        let score = if unarmed && secondary { base + UNARMED_SECONDARY_BONUS } else { base };
        better(
            &mut best,
            ItemGoal {
                kind: ItemKind::Weapon,
                id,
                pos,
                score,
                stand_off: 0.0,
            },
        );
    }
    if obs.me.cubes < genome.cube_cap {
        for (id, pos) in item_candidates(obs, mem, ItemKind::Cube) {
            let dist = dist2d(pos, me);
            if dist > cube_cap || !next_zone_ok(obs, pos) || raced(obs, threats, pos, dist) || exposed(pos) || !item_ok(obs, nav, mem, id, pos) {
                continue;
            }
            better(
                &mut best,
                ItemGoal {
                    kind: ItemKind::Cube,
                    id,
                    pos,
                    score: 250.0 - dist,
                    stand_off: 0.0,
                },
            );
        }
    }
    let farm_ok = if !stealth {
        !(unarmed && obs.t < 3.0) && hp_frac(obs) >= genome.farm_hp_min && quiet_within(obs, threats, 150.0, 5.0)
    } else if has_laser {
        obs.t < 150.0 && hp_frac(obs) >= genome.farm_hp_min.max(0.70) && quiet_within(obs, threats, 130.0, 6.0)
    } else {
        obs.t >= 3.0 && hp_frac(obs) >= genome.farm_hp_min && quiet_within(obs, threats, 95.0, 5.0)
    };
    if farm_ok {
        let npc_stand = if holds(obs, WeaponKind::Lightning) { 25.0 } else { 45.0 };
        for n in obs.mechs.iter().filter(|m| m.kind == ActorKind::Npc) {
            let dist = dist2d(n.pos, me);
            if dist > npc_cap || !next_zone_ok(obs, n.pos) || !farm_zone_ok(obs, n.pos) || exposed(n.pos) || !item_ok(obs, nav, mem, n.id, n.pos) {
                continue;
            }
            better(
                &mut best,
                ItemGoal {
                    kind: ItemKind::Npc,
                    id: n.id,
                    pos: n.pos,
                    score: 150.0 - dist,
                    stand_off: npc_stand,
                },
            );
        }
        for (id, pos) in item_candidates(obs, mem, ItemKind::Crate) {
            let dist = dist2d(pos, me);
            if dist > crate_cap || !next_zone_ok(obs, pos) || !farm_zone_ok(obs, pos) || exposed(pos) || !item_ok(obs, nav, mem, id, pos) {
                continue;
            }
            better(
                &mut best,
                ItemGoal {
                    kind: ItemKind::Crate,
                    id,
                    pos,
                    score: 120.0 - dist,
                    stand_off: 26.0,
                },
            );
        }
    }
    best
}

fn loot_or_farm<N: NavView, M: MemView>(obs: &Obs, nav: &N, mem: &M, genome: &ApexGenome, in_fight: bool, threats: &[ThreatInfo]) -> Option<ItemGoal> {
    pick_drop(obs, nav, mem, genome, threats)
        .or_else(|| pick_kill_cube(obs, nav, mem, genome))
        .or_else(|| {
            if in_fight {
                pick_fight_detour(obs, nav, mem, genome, threats)
            } else {
                pick_calm(obs, nav, mem, genome, threats)
            }
        })
}

/// The item, drop, or farm target worth going for now, if any. Priority:
/// supply drop, then the cubes of a fresh kill, then a mid-fight detour or
/// calm looting and farming.
pub fn choose_loot_or_farm(obs: &Obs, nav: &ApexNav, mem: &BotMemory, genome: &ApexGenome, in_fight: bool, threats: &[ThreatInfo]) -> Option<ItemGoal> {
    loot_or_farm(obs, nav, mem, genome, in_fight, threats)
}

// -------------------------------------------------------------------- zone

fn zone_core<N: NavView>(obs: &Obs, nav: &N) -> (bool, bool, [f64; 3]) {
    let z = &obs.zone;
    let me = obs.me.pos;
    let (nx, ny) = (z.next_center[0], z.next_center[1]);
    let next_r = z.next_radius.max(0.0);
    let safe_r = if next_r <= 28.0 {
        0.0
    } else {
        (next_r - clamp(0.25 * next_r, 6.0, 28.0)).max(0.0)
    };
    let (dx, dy) = (me[0] - nx, me[1] - ny);
    let d_next = dx.hypot(dy);
    let (sx, sy) = if d_next > 1e-9 && safe_r > 0.0 {
        (nx + dx / d_next * safe_r, ny + dy / d_next * safe_r)
    } else {
        (nx, ny)
    };
    let safe = [sx, sy, nav.ground_z(sx, sy)];
    let eta = dist2d(me, safe) / walk_speed(obs) * 1.25 + 6.0;

    let to_start = z.shrink_start - obs.t;
    let to_end = z.shrink_end - obs.t;
    let shrinking = to_start <= 0.0 && to_end > 0.0;
    let d_cur = (me[0] - z.center[0]).hypot(me[1] - z.center[1]);
    let outside_next = d_next > next_r;

    let zone_must =
        !obs.me.in_zone || (shrinking && z.radius - d_cur <= 8.0) || (outside_next && ((shrinking && to_end <= eta) || (to_start > 0.0 && to_start <= eta)));
    let zone_should = d_next > safe_r + 10.0 && to_start <= eta + 14.0;
    (zone_must, zone_should, safe)
}

/// `(zone_must, zone_should, safe_goal_3d)`: whether we must head into the
/// zone now, whether we should drift toward it, and the point to aim for.
pub fn zone_assessment(obs: &Obs, nav: &ApexNav) -> (bool, bool, [f64; 3]) {
    zone_core(obs, nav)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proto::{ActorObs, DropObs, GunStats, ItemObs, LaserStats, LightningStats, Offer, OfferChoice, PickupObs};

    struct Nav {
        reach: bool,
        ground: f64,
    }
    impl NavView for Nav {
        fn reachable(&self, _: [f64; 3], _: [f64; 3]) -> bool {
            self.reach
        }
        fn ground_z(&self, _: f64, _: f64) -> f64 {
            self.ground
        }
        fn is_main_xy(&self, _: f64, _: f64) -> bool {
            self.reach
        }
    }

    struct Mem {
        black: Vec<u32>,
        kill_t: f64,
        kill_pos: Option<[f64; 3]>,
        items: Vec<RememberedItem>,
        ffa: bool,
    }
    impl MemView for Mem {
        fn blacklisted(&self, id: u32, _: f64) -> bool {
            self.black.contains(&id)
        }
        fn kill_t(&self) -> f64 {
            self.kill_t
        }
        fn kill_pos(&self) -> Option<[f64; 3]> {
            self.kill_pos
        }
        fn items(&self) -> &[RememberedItem] {
            &self.items
        }
        fn is_ffa(&self) -> bool {
            self.ffa
        }
    }

    const NAV: Nav = Nav { reach: true, ground: 0.0 };

    fn mem() -> Mem {
        Mem {
            black: vec![],
            kill_t: f64::NEG_INFINITY,
            kill_pos: None,
            items: vec![],
            ffa: false,
        }
    }

    fn remembered_item(id: u32, kind: ItemKind, weapon_kind: Option<WeaponKind>, pos: [f64; 3], last_seen_t: f64) -> RememberedItem {
        RememberedItem {
            id,
            kind,
            weapon_kind,
            pos,
            hp: None,
            last_seen_t,
        }
    }

    fn gun() -> WeaponObs {
        WeaponObs::Gun {
            cooldown: 0.0,
            stats: GunStats::default(),
            ammo: 12.0,
            reload: 0.0,
        }
    }
    fn laser(charge: f64) -> WeaponObs {
        WeaponObs::Laser {
            cooldown: 0.0,
            charge,
            stats: LaserStats::default(),
        }
    }
    fn lightning() -> WeaponObs {
        WeaponObs::Lightning {
            cooldown: 0.0,
            stats: LightningStats::default(),
        }
    }

    fn base_obs() -> Obs {
        let mut o = Obs { t: 10.0, ..Default::default() };
        o.me.id = 1;
        o.me.hp = 1400.0;
        o.me.max_hp = 1400.0;
        o.me.in_zone = true;
        o.me.weapons = [Some(gun()), None];
        o.zone.radius = 500.0;
        o.zone.next_radius = 300.0;
        o.zone.shrink_start = 100.0;
        o.zone.shrink_end = 150.0;
        o
    }

    fn genome() -> ApexGenome {
        ApexGenome::practice_default()
    }

    fn offer(obs: &mut Obs, ids: &[&str]) {
        obs.offer = Some(Offer {
            id: 1,
            choices: ids
                .iter()
                .map(|i| OfferChoice {
                    id: (*i).into(),
                    name: String::new(),
                    desc: String::new(),
                })
                .collect(),
            expires: 99.0,
        });
    }

    fn pickup(id: u32, kind: WeaponKind, pos: [f64; 3]) -> PickupObs {
        PickupObs { id, kind, pos }
    }

    fn item(id: u32, pos: [f64; 3]) -> ItemObs {
        ItemObs { id, pos, hp: None }
    }

    fn threat(pos: [f64; 3], dist: f64, seen: f64) -> ThreatInfo {
        ThreatInfo {
            id: 9,
            pos,
            vel: [0.0; 3],
            aim: [0.0; 2],
            hp: 1000.0,
            max_hp: 1400.0,
            cubes: 0,
            weapons: [None, None],
            charging: false,
            dist,
            los: true,
            visible_now: true,
            last_seen_t: seen,
            last_hit_us_t: f64::NEG_INFINITY,
            last_beam_t: f64::NEG_INFINITY,
            is_ghost: false,
            is_aware: false,
            threat_dps: 0.0,
            target_score: 0.0,
        }
    }

    #[test]
    fn upgrade_orders() {
        let mut o = base_obs();
        let mut g = genome();
        assert_eq!(pick_upgrade(&o, &g), None);
        offer(&mut o, &[]);
        assert_eq!(pick_upgrade(&o, &g), None);

        g.upgrade_profile = UpgradeProfile::DefenseFirst;
        offer(&mut o, &["servos", "nanites", "armor"]);
        assert_eq!(pick_upgrade(&o, &g), Some(2));

        // Laser upgrades are demoted while we hold no laser.
        g.upgrade_profile = UpgradeProfile::LaserEcon;
        offer(&mut o, &["laser_damage", "nanites"]);
        assert_eq!(pick_upgrade(&o, &g), Some(1));
        o.me.weapons[1] = Some(laser(0.0));
        assert_eq!(pick_upgrade(&o, &g), Some(0));

        g.upgrade_profile = UpgradeProfile::GunMagBrawl;
        offer(&mut o, &["armor", "gun_mag", "sensors"]);
        assert_eq!(pick_upgrade(&o, &g), Some(1));

        // Unknown ids still yield a choice.
        offer(&mut o, &["mystery", "mystery2"]);
        assert_eq!(pick_upgrade(&o, &g), Some(0));
    }

    #[test]
    fn smart_heal_takes_plating_when_hurt() {
        let mut o = base_obs();
        let mut g = genome();
        g.upgrade_profile = UpgradeProfile::SmartHealDefense;
        offer(&mut o, &["armor", "plating"]);
        o.me.hp = 0.5 * o.me.max_hp;
        assert_eq!(pick_upgrade(&o, &g), Some(1));
        offer(&mut o, &["nanites", "armor", "servos"]);
        assert_eq!(pick_upgrade(&o, &g), Some(1));
    }

    #[test]
    fn weapon_gun_laser() {
        let mut o = base_obs();
        let mut g = genome();
        g.weapon_pref = WeaponPref::GunLaser;
        o.pickups = vec![pickup(5, WeaponKind::Mortar, [1.0, 0.0, 0.0]), pickup(6, WeaponKind::Laser, [3.0, 0.0, 0.0])];
        assert_eq!(pick_weapon_action(&o, &g), (Some(6), None));
        // Out of reach vertically.
        o.pickups[1].pos[2] = 5.0;
        assert_eq!(pick_weapon_action(&o, &g), (None, None));
        // Lightning in slot 1 gets swapped for a laser; the gun stays.
        o.pickups[1].pos[2] = 0.0;
        o.me.weapons = [Some(lightning()), Some(gun())];
        assert_eq!(pick_weapon_action(&o, &g), (Some(6), Some(0)));
        // But not a lightning for a laser.
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        o.pickups = vec![pickup(7, WeaponKind::Lightning, [1.0, 0.0, 0.0])];
        assert_eq!(pick_weapon_action(&o, &g), (None, None));
    }

    #[test]
    fn weapon_action_never_drops_gun_with_slot_one_empty() {
        let kinds = [WeaponKind::Gun, WeaponKind::Laser, WeaponKind::Lightning, WeaponKind::Mortar];
        for pref in [WeaponPref::GunLaser, WeaponPref::LaserLightningDropGun, WeaponPref::GunLightning] {
            let mut g = genome();
            g.weapon_pref = pref;
            for kind in kinds {
                let mut o = base_obs();
                o.me.weapons = [Some(gun()), None];
                o.pickups = vec![pickup(8, kind, [1.0, 0.0, 0.0])];
                let (take, drop) = pick_weapon_action(&o, &g);
                assert_eq!(drop, None, "{pref:?} dropped a slot for {kind:?}");
                assert_eq!(take.is_some(), kind != WeaponKind::Gun && kind != WeaponKind::Mortar, "{pref:?} with {kind:?}");
            }
        }
    }

    #[test]
    fn fight_detour_reaches_for_a_laser_in_a_lull() {
        let mut o = base_obs();
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.weapon_pref = WeaponPref::GunLaser;
        g.laser_seek_r = 200.0;
        o.pickups = vec![pickup(6, WeaponKind::Laser, [-100.0, 0.0, 0.0])];
        let far = [threat([90.0, 0.0, 0.0], 90.0, 10.0)];
        assert_eq!(loot_or_farm(&o, &NAV, &mem(), &g, true, &far).map(|i| i.id), Some(6));
        // A close threat or a recent hit keeps us in the fight.
        let close = [threat([40.0, 0.0, 0.0], 40.0, 10.0)];
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, true, &close).is_none());
        let mut hit = far;
        hit[0].last_hit_us_t = 9.0;
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, true, &hit).is_none());
        // Capped at 120 m.
        o.pickups[0].pos = [-130.0, 0.0, 0.0];
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, true, &far).is_none());
    }

    #[test]
    fn weapon_laser_lightning_drops_gun() {
        let mut o = base_obs();
        let mut g = genome();
        g.weapon_pref = WeaponPref::LaserLightningDropGun;
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        o.pickups = vec![pickup(7, WeaponKind::Lightning, [1.0, 0.0, 0.0])];
        assert_eq!(pick_weapon_action(&o, &g), (Some(7), Some(0)));
    }

    #[test]
    fn weapon_gun_lightning_and_charging_veto() {
        let mut o = base_obs();
        let mut g = genome();
        g.weapon_pref = WeaponPref::GunLightning;
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        o.pickups = vec![pickup(7, WeaponKind::Lightning, [1.0, 0.0, 0.0])];
        assert_eq!(pick_weapon_action(&o, &g), (Some(7), Some(1)));
        o.me.weapons[1] = Some(laser(0.3));
        assert_eq!(pick_weapon_action(&o, &g), (None, None));
    }

    #[test]
    fn drops_follow_contest_mode() {
        let mut o = base_obs();
        let mut g = genome();
        o.drops = vec![DropObs {
            id: 40,
            pos: [50.0, 0.0, 0.0],
            radius: 8.0,
            lands_in: 0.0,
            capture_s: 4.0,
            progress: 0.0,
            holder: None,
        }];
        g.drop_contest = DropContest::Off;
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).is_none_or(|i| i.kind != ItemKind::Drop));
        g.drop_contest = DropContest::AlwaysUncontested;
        let goal = loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).unwrap();
        assert_eq!((goal.kind, goal.id, goal.stand_off), (ItemKind::Drop, 40, 0.0));
        o.drops[0].holder = Some(77);
        assert!(pick_drop(&o, &NAV, &mem(), &g, &[]).is_none());
        g.drop_contest = DropContest::HealOnly95;
        assert!(pick_drop(&o, &NAV, &mem(), &g, &[]).is_none());
        o.me.hp = 0.8 * o.me.max_hp;
        assert!(pick_drop(&o, &NAV, &mem(), &g, &[]).is_some());
        // Threat within 70 m of the drop rejects HealOnly95.
        let near_drop = [threat([100.0, 0.0, 0.0], 100.0, 10.0)];
        assert!(pick_drop(&o, &NAV, &mem(), &g, &near_drop).is_none());
        // Drop beyond 160 m rejects HealOnly95.
        o.drops[0].pos = [170.0, 0.0, 0.0];
        assert!(pick_drop(&o, &NAV, &mem(), &g, &[]).is_none());
        o.drops[0].pos = [50.0, 0.0, 0.0];
        // Lands far too late to wait for.
        o.drops[0].lands_in = 60.0;
        assert!(pick_drop(&o, &NAV, &mem(), &g, &[]).is_none());
    }

    #[test]
    fn cubes_respect_cap_and_blacklist() {
        let mut o = base_obs();
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.cube_cap = 8;
        o.cubes = vec![item(20, [10.0, 0.0, 0.0]), item(21, [30.0, 0.0, 0.0])];
        let goal = loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).unwrap();
        assert_eq!((goal.kind, goal.id), (ItemKind::Cube, 20));
        let m = Mem { black: vec![20], ..mem() };
        assert_eq!(loot_or_farm(&o, &NAV, &m, &g, false, &[]).unwrap().id, 21);
        o.me.cubes = 8;
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).is_none());
        // Unreachable cube on a roof is skipped.
        o.me.cubes = 0;
        o.cubes = vec![item(22, [10.0, 0.0, 9.0])];
        assert!(loot_or_farm(&o, &Nav { reach: false, ground: 0.0 }, &mem(), &g, false, &[]).is_none());
        // Outside the current zone margin.
        o.cubes = vec![item(23, [497.0, 0.0, 0.0])];
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).is_none());
    }

    #[test]
    fn kill_cube_rush_beats_calm_loot() {
        let mut o = base_obs();
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.cube_cap = 10;
        o.pickups = vec![pickup(6, WeaponKind::Laser, [20.0, 0.0, 0.0])];
        o.cubes = vec![item(30, [80.0, 0.0, 0.0])];
        let m = Mem {
            kill_t: 8.0,
            kill_pos: Some([75.0, 0.0, 0.0]),
            ..mem()
        };
        let goal = loot_or_farm(&o, &NAV, &m, &g, true, &[]).unwrap();
        assert_eq!((goal.kind, goal.id), (ItemKind::Cube, 30));
        let stale = Mem { kill_t: 1.0, ..m };
        assert_eq!(loot_or_farm(&o, &NAV, &stale, &g, false, &[]).unwrap().id, 6);
    }

    #[test]
    fn fight_detour_avoids_threat_side() {
        let mut o = base_obs();
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.cube_cap = 10;
        g.fight_cube_r = 60.0;
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        o.cubes = vec![item(31, [40.0, 0.0, 0.0])];
        let t = [threat([50.0, 0.0, 0.0], 50.0, 10.0)];
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, true, &t).is_none());
        o.cubes[0].pos = [-40.0, 0.0, 0.0];
        assert_eq!(loot_or_farm(&o, &NAV, &mem(), &g, true, &t).unwrap().id, 31);
        // Low HP halves the radius.
        o.me.hp = 0.4 * o.me.max_hp;
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, true, &t).is_none());
    }

    #[test]
    fn farm_needs_hp_quiet_and_zone() {
        let mut o = base_obs();
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.farm_hp_min = 0.6;
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        let npc = ActorObs {
            id: 50,
            kind: ActorKind::Npc,
            pos: [60.0, 0.0, 0.0],
            vel: [0.0; 3],
            aim: [0.0; 2],
            hp: None,
            max_hp: None,
            cubes: None,
            weapons: None,
            charging: None,
        };
        o.mechs = vec![npc];
        let goal = loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).unwrap();
        assert_eq!((goal.kind, goal.stand_off), (ItemKind::Npc, 45.0));
        let t = [threat([100.0, 0.0, 0.0], 100.0, 8.0)];
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, false, &t).is_none());
        o.me.hp = 0.5 * o.me.max_hp;
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).is_none());
        o.me.hp = o.me.max_hp;
        o.mechs.clear();
        o.crates = vec![ItemObs {
            id: 60,
            pos: [40.0, 0.0, 0.0],
            hp: Some(300.0),
        }];
        let goal = loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).unwrap();
        assert_eq!((goal.kind, goal.stand_off), (ItemKind::Crate, 26.0));
    }

    #[test]
    fn remembered_weapon_out_of_view_is_still_targeted() {
        let mut o = base_obs();
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.weapon_pref = WeaponPref::GunLaser;
        // Nothing in view; a laser seen 10 s ago behind us.
        let m = Mem {
            items: vec![remembered_item(70, ItemKind::Weapon, Some(WeaponKind::Laser), [-60.0, 0.0, 0.0], 0.0)],
            ..mem()
        };
        let goal = loot_or_farm(&o, &NAV, &m, &g, false, &[]).unwrap();
        assert_eq!((goal.kind, goal.id, goal.pos), (ItemKind::Weapon, 70, [-60.0, 0.0, 0.0]));
        // Also reached for mid-fight in a lull.
        let far = [threat([90.0, 0.0, 0.0], 90.0, 10.0)];
        assert_eq!(loot_or_farm(&o, &NAV, &m, &g, true, &far).map(|i| i.id), Some(70));
        // A weapon memory without a known kind is ignored.
        let unknown = Mem {
            items: vec![remembered_item(71, ItemKind::Weapon, None, [-60.0, 0.0, 0.0], 0.0)],
            ..mem()
        };
        assert!(loot_or_farm(&o, &NAV, &unknown, &g, false, &[]).is_none());
        // Past the 55 s window it is forgotten.
        o.t = 56.0;
        assert!(loot_or_farm(&o, &NAV, &m, &g, false, &[]).is_none());
        o.t = 55.0;
        assert_eq!(loot_or_farm(&o, &NAV, &m, &g, false, &[]).unwrap().id, 70);
    }

    #[test]
    fn visible_copy_beats_remembered_copy() {
        let mut o = base_obs();
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.weapon_pref = WeaponPref::GunLaser;
        o.pickups = vec![pickup(70, WeaponKind::Laser, [30.0, 0.0, 0.0])];
        let m = Mem {
            items: vec![remembered_item(70, ItemKind::Weapon, Some(WeaponKind::Laser), [-60.0, 0.0, 0.0], 9.0)],
            ..mem()
        };
        assert_eq!(loot_or_farm(&o, &NAV, &m, &g, false, &[]).unwrap().pos, [30.0, 0.0, 0.0]);
    }

    #[test]
    fn remembered_cubes_and_crates_respect_freshness() {
        let mut o = base_obs();
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.cube_cap = 8;
        let m = Mem {
            items: vec![remembered_item(80, ItemKind::Cube, None, [20.0, 0.0, 0.0], 0.0)],
            ..mem()
        };
        let goal = loot_or_farm(&o, &NAV, &m, &g, false, &[]).unwrap();
        assert_eq!((goal.kind, goal.id), (ItemKind::Cube, 80));
        // The blacklist still applies to remembered items.
        let black = Mem {
            black: vec![80],
            items: m.items.clone(),
            ..mem()
        };
        assert!(loot_or_farm(&o, &NAV, &black, &g, false, &[]).is_none());
        // Kill-cube rush uses memory too.
        let kill = Mem {
            kill_t: 8.0,
            kill_pos: Some([25.0, 0.0, 0.0]),
            items: m.items.clone(),
            ..mem()
        };
        let goal = loot_or_farm(&o, &NAV, &kill, &g, true, &[]).unwrap();
        assert_eq!((goal.kind, goal.id, goal.score), (ItemKind::Cube, 80, 880.0));
        // Stale beyond 35 s.
        o.t = 36.0;
        assert!(loot_or_farm(&o, &NAV, &m, &g, false, &[]).is_none());

        // Crates: remembered for 45 s, farm checks still apply.
        o.t = 10.0;
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        let crates = Mem {
            items: vec![remembered_item(90, ItemKind::Crate, None, [40.0, 0.0, 0.0], 0.0)],
            ..mem()
        };
        let goal = loot_or_farm(&o, &NAV, &crates, &g, false, &[]).unwrap();
        assert_eq!((goal.kind, goal.id, goal.stand_off), (ItemKind::Crate, 90, 26.0));
        o.t = 46.0;
        assert!(loot_or_farm(&o, &NAV, &crates, &g, false, &[]).is_none());
    }

    #[test]
    fn unarmed_bot_prefers_a_secondary() {
        let mut o = base_obs();
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.weapon_pref = WeaponPref::GunLaser;
        g.cube_cap = 8;
        g.laser_seek_r = 0.0;
        // Lightning at 60 m: 300 + 5 - 60 + 120 = 365 beats a cube at 5 m (245).
        o.pickups = vec![pickup(6, WeaponKind::Lightning, [60.0, 0.0, 0.0])];
        o.cubes = vec![item(20, [5.0, 0.0, 0.0])];
        let goal = loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).unwrap();
        assert_eq!((goal.id, goal.score), (6, 365.0));
    }

    fn calm_genome() -> ApexGenome {
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.weapon_pref = WeaponPref::GunLaser;
        g.cube_cap = 8;
        g.farm_hp_min = 0.0;
        g.laser_seek_r = 260.0;
        g
    }

    fn npc_at(id: u32, pos: [f64; 3]) -> ActorObs {
        ActorObs {
            id,
            kind: ActorKind::Npc,
            pos,
            vel: [0.0; 3],
            aim: [0.0; 2],
            hp: None,
            max_hp: None,
            cubes: None,
            weapons: None,
            charging: None,
        }
    }

    fn calm(o: &Obs, m: &Mem, g: &ApexGenome, threats: &[ThreatInfo]) -> Option<(ItemKind, u32)> {
        pick_calm(o, &NAV, m, g, threats).map(|i| (i.kind, i.id))
    }

    #[test]
    fn calm_caps_are_tighter_in_ffa() {
        let g = calm_genome();
        let practice = mem();
        let ffa = Mem { ffa: true, ..mem() };
        let armed = |mut o: Obs| {
            o.me.weapons = [Some(gun()), Some(laser(0.0))];
            o
        };
        // Cube at 100 m: inside the 110 m practice cap, past the 80 m/65 m FFA caps.
        let mut o = armed(base_obs());
        o.cubes = vec![item(20, [100.0, 0.0, 0.0])];
        assert_eq!(calm(&o, &practice, &g, &[]), Some((ItemKind::Cube, 20)));
        assert_eq!(calm(&o, &ffa, &g, &[]), None);
        o.cubes[0].pos = [115.0, 0.0, 0.0];
        assert_eq!(calm(&o, &practice, &g, &[]), None);
        // Armed with laser: cube cap is 65 m in FFA (75 m skipped, 60 m taken); unarmed is 80 m.
        o.cubes[0].pos = [75.0, 0.0, 0.0];
        assert_eq!(calm(&o, &ffa, &g, &[]), None);
        o.cubes[0].pos = [60.0, 0.0, 0.0];
        assert_eq!(calm(&o, &ffa, &g, &[]), Some((ItemKind::Cube, 20)));
        let mut ou = base_obs();
        ou.cubes = vec![item(20, [75.0, 0.0, 0.0])];
        assert_eq!(calm(&ou, &ffa, &g, &[]), Some((ItemKind::Cube, 20)));

        // Crate at 80 m: practice only (past 75 m/60 m FFA caps).
        let mut o = armed(base_obs());
        o.crates = vec![item(60, [80.0, 0.0, 0.0])];
        assert_eq!(calm(&o, &practice, &g, &[]), Some((ItemKind::Crate, 60)));
        assert_eq!(calm(&o, &ffa, &g, &[]), None);
        // Armed with laser: crate at 55 m allowed if t < 150, hp >= 0.70, quiet 130 m/6 s.
        o.crates[0].pos = [55.0, 0.0, 0.0];
        assert_eq!(calm(&o, &ffa, &g, &[]), Some((ItemKind::Crate, 60)));
        o.t = 155.0;
        assert_eq!(calm(&o, &ffa, &g, &[]), None);
        o.t = 10.0;
        o.me.hp = 0.65 * o.me.max_hp;
        assert_eq!(calm(&o, &ffa, &g, &[]), None);
        o.me.hp = o.me.max_hp;
        let far_watcher = [threat([-120.0, 0.0, 0.0], 120.0, 5.0)];
        assert_eq!(calm(&o, &ffa, &g, &far_watcher), None);

        // NPC at 100 m: practice only (past 65 m armed FFA cap). 60 m: both.
        let mut o = armed(base_obs());
        o.mechs = vec![npc_at(50, [100.0, 0.0, 0.0])];
        assert_eq!(calm(&o, &practice, &g, &[]), Some((ItemKind::Npc, 50)));
        assert_eq!(calm(&o, &ffa, &g, &[]), None);
        o.mechs[0].pos = [60.0, 0.0, 0.0];
        assert_eq!(calm(&o, &ffa, &g, &[]), Some((ItemKind::Npc, 50)));

        // Laser at 250 m: within laser_seek_r (260 m), past the 240 m FFA/stealth ceiling.
        let mut o = base_obs();
        o.pickups = vec![pickup(6, WeaponKind::Laser, [250.0, 0.0, 0.0])];
        assert_eq!(calm(&o, &practice, &g, &[]), Some((ItemKind::Weapon, 6)));
        assert_eq!(calm(&o, &ffa, &g, &[]), None);
        let mut gq = calm_genome();
        gq.quiet_walk = true;
        assert_eq!(calm(&o, &practice, &gq, &[]), None, "quiet_walk uses stealth laser cap");
        o.pickups[0].pos = [230.0, 0.0, 0.0];
        assert_eq!(calm(&o, &practice, &gq, &[]), Some((ItemKind::Weapon, 6)));
        o.pickups[0].pos = [270.0, 0.0, 0.0];
        assert_eq!(calm(&o, &practice, &g, &[]), None);

        // Lightning (not the top weapon) at 130 m: practice only (past 120 m FFA weapon cap). 110 m: both.
        let mut o = base_obs();
        o.pickups = vec![pickup(7, WeaponKind::Lightning, [130.0, 0.0, 0.0])];
        assert_eq!(calm(&o, &practice, &g, &[]), Some((ItemKind::Weapon, 7)));
        assert_eq!(calm(&o, &ffa, &g, &[]), None);
        o.pickups[0].pos = [110.0, 0.0, 0.0];
        assert_eq!(calm(&o, &ffa, &g, &[]), Some((ItemKind::Weapon, 7)));
        // A top-ranked lightning still obeys the 120 m FFA weapon cap.
        let mut gl = calm_genome();
        gl.weapon_pref = WeaponPref::GunLightning;
        o.pickups[0].pos = [130.0, 0.0, 0.0];
        assert_eq!(calm(&o, &practice, &gl, &[]), Some((ItemKind::Weapon, 7)));
        assert_eq!(calm(&o, &ffa, &gl, &[]), None);
    }

    #[test]
    fn roof_items_are_rejected() {
        let mut o = base_obs();
        let g = calm_genome();
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        // 2 m up on flat ground: a roof (step_up 1.2 + 0.5 = 1.7).
        o.cubes = vec![item(20, [20.0, 0.0, 2.0])];
        assert!(pick_calm(&o, &NAV, &mem(), &g, &[]).is_none());
        // Even when the nav would call it reachable and it is within LEVEL_DZ.
        assert!(!item_ok(&o, &NAV, &mem(), 20, [20.0, 0.0, 2.0]));
        // On a hill: the terrain rises with it.
        let hill = Nav { reach: true, ground: 2.0 };
        assert_eq!(pick_calm(&o, &hill, &mem(), &g, &[]).map(|i| i.id), Some(20));
        // We stand on the same roof: fine below ground + 3.2.
        o.me.pos[2] = 1.5;
        assert_eq!(pick_calm(&o, &NAV, &mem(), &g, &[]).map(|i| i.id), Some(20));
        // A bigger step_up raises the bar up to ground + 3.2.
        o.me.pos[2] = 0.0;
        o.me.stats.step_up = 2.0;
        assert_eq!(pick_calm(&o, &NAV, &mem(), &g, &[]).map(|i| i.id), Some(20));
        // Anything above ground + 3.2 is ALWAYS rejected even if we are elevated or step_up is large.
        o.me.pos[2] = 2.0;
        o.me.stats.step_up = 4.0;
        o.cubes = vec![item(21, [20.0, 0.0, 3.3])];
        assert!(!item_ok(&o, &NAV, &mem(), 21, [20.0, 0.0, 3.3]));
        assert!(pick_calm(&o, &NAV, &mem(), &g, &[]).is_none());
        // Weapon pickups too, including on real ApexNav with a roof building.
        o.me.pos[2] = 0.0;
        o.me.stats.step_up = 1.2;
        o.cubes.clear();
        o.me.weapons = [Some(gun()), None];
        o.pickups = vec![pickup(6, WeaponKind::Laser, [30.0, 0.0, 6.0])];
        assert!(pick_calm(&o, &NAV, &mem(), &g, &[]).is_none());

        let mut w = world::World::flat(100.0);
        w.prisms
            .push(world::Prism::rect(world::PrismKind::Building, [20.0, -5.0], [30.0, 5.0], 0.0, 4.0));
        let nav = ApexNav::build(std::sync::Arc::new(w), &proto::MechStats::default(), &proto::VisionCfg::default());
        o.pickups = vec![pickup(8, WeaponKind::Laser, [25.0, 0.0, 4.0])];
        assert!(pick_calm(&o, &nav, &mem(), &g, &[]).is_none());
    }

    #[test]
    fn lasers_and_cubes_lost_to_a_nearer_threat_are_skipped() {
        let g = calm_genome();
        let mut o = base_obs();
        o.pickups = vec![pickup(6, WeaponKind::Laser, [100.0, 0.0, 0.0])];
        // Threat 20 m from the laser, we are 100 m away.
        let near = [threat([100.0, 20.0, 0.0], 102.0, 8.0)];
        assert!(calm(&o, &mem(), &g, &near).is_none());
        // Seen more than 5 s ago: no longer counted.
        let old = [threat([100.0, 20.0, 0.0], 102.0, 4.9)];
        assert_eq!(calm(&o, &mem(), &g, &old), Some((ItemKind::Weapon, 6)));
        // A ghost does not race.
        let mut ghost = near;
        ghost[0].is_ghost = true;
        assert_eq!(calm(&o, &mem(), &g, &ghost), Some((ItemKind::Weapon, 6)));
        // Farther from the item than we are: fine.
        let far = [threat([-50.0, 0.0, 0.0], 50.0, 9.0)];
        assert_eq!(calm(&o, &mem(), &g, &far), Some((ItemKind::Weapon, 6)));
        // Cubes too.
        let mut o = base_obs();
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        o.cubes = vec![item(20, [40.0, 0.0, 0.0])];
        let near = [threat([40.0, 10.0, 0.0], 41.0, 9.0)];
        assert!(calm(&o, &mem(), &g, &near).is_none());
        // Mid-fight weapon detours check the race too.
        let mut o = base_obs();
        let mut g = calm_genome();
        g.laser_seek_r = 200.0;
        o.pickups = vec![pickup(6, WeaponKind::Laser, [-100.0, 0.0, 0.0])];
        let racer = [threat([-100.0, 60.0, 0.0], 116.0, 10.0)];
        assert!(pick_fight_detour(&o, &NAV, &mem(), &g, &racer).is_none());
    }

    #[test]
    fn ffa_skips_calm_loot_past_a_threat() {
        let g = calm_genome();
        let ffa = Mem { ffa: true, ..mem() };
        let mut o = base_obs();
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        o.cubes = vec![item(20, [60.0, 0.0, 0.0])];
        // Threat 48 m off the path's midpoint, farther from the cube than we
        // are (no race), but the walk closes on it (< CORRIDOR_R = 55 m).
        let side = [threat([20.0, 48.0, 0.0], 52.0, 9.0)];
        assert!(calm(&o, &ffa, &g, &side).is_none());
        assert_eq!(calm(&o, &mem(), &g, &side), Some((ItemKind::Cube, 20)), "practice ignores it");
        // Threat behind us: the path never nears it more than it already is.
        let behind = [threat([-60.0, 0.0, 0.0], 60.0, 9.0)];
        assert_eq!(calm(&o, &ffa, &g, &behind), Some((ItemKind::Cube, 20)));
        // Path clear by more than 55 m.
        let wide = [threat([60.0, 60.0, 0.0], 84.85, 9.0)];
        assert_eq!(calm(&o, &ffa, &g, &wide), Some((ItemKind::Cube, 20)));
        // Stale (>= 4 s) threats do not block.
        let stale = [threat([20.0, 48.0, 0.0], 52.0, 5.9)];
        assert_eq!(calm(&o, &ffa, &g, &stale), Some((ItemKind::Cube, 20)));
        // Stealth/FFA skips ALL exposed weapons (even when unarmed, including lasers).
        let mut o = base_obs();
        o.pickups = vec![pickup(7, WeaponKind::Lightning, [60.0, 0.0, 0.0])];
        assert!(calm(&o, &ffa, &g, &side).is_none());
        let mut gq = calm_genome();
        gq.quiet_walk = true;
        assert!(calm(&o, &mem(), &gq, &side).is_none(), "quiet_walk also skips exposed weapons");
        o.me.weapons = [Some(lightning()), Some(gun())];
        o.pickups = vec![pickup(8, WeaponKind::Laser, [60.0, 0.0, 0.0])];
        assert!(calm(&o, &ffa, &g, &side).is_none(), "lasers are also skipped when exposed");
        assert!(calm(&o, &mem(), &gq, &side).is_none());
        let mut gl = calm_genome();
        gl.weapon_pref = WeaponPref::GunLightning;
        let mut o = base_obs();
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        o.pickups = vec![pickup(7, WeaponKind::Lightning, [60.0, 0.0, 0.0])];
        assert!(calm(&o, &ffa, &gl, &side).is_none(), "armed: no detour past a threat");
        assert_eq!(calm(&o, &mem(), &gl, &side), Some((ItemKind::Weapon, 7)));
    }

    #[test]
    fn unarmed_skips_early_farming_and_fight_detour_upgrades_secondary() {
        let g = calm_genome();
        let mut o = base_obs();
        o.t = 1.5;
        o.crates = vec![item(60, [20.0, 0.0, 0.0])];
        o.mechs = vec![npc_at(50, [30.0, 0.0, 0.0])];
        assert!(calm(&o, &mem(), &g, &[]).is_none(), "unarmed skips crate/npc farm before t = 3.0");
        o.t = 3.5;
        assert_eq!(calm(&o, &mem(), &g, &[]), Some((ItemKind::Npc, 50)));

        // Fight detour upgrades Lightning -> Laser even when slot 1 is occupied.
        let mut o = base_obs();
        o.me.weapons = [Some(gun()), Some(lightning())];
        o.pickups = vec![pickup(6, WeaponKind::Laser, [30.0, 0.0, 0.0])];
        let far = [threat([90.0, 0.0, 0.0], 90.0, 10.0)];
        assert_eq!(loot_or_farm(&o, &NAV, &mem(), &g, true, &far).map(|i| i.id), Some(6));
    }

    #[test]
    fn calm_loot_waits_inside_the_next_zone_before_a_shrink() {
        let g = calm_genome();
        let mut o = base_obs();
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        o.zone.next_center = [0.0, 0.0];
        o.zone.next_radius = 50.0;
        o.cubes = vec![item(20, [45.0, 0.0, 0.0])];
        // 90 s before the shrink: anything in the current zone goes.
        assert_eq!(calm(&o, &mem(), &g, &[]), Some((ItemKind::Cube, 20)));
        // 20 s before: the cube is only 5 m inside the next circle.
        o.t = 80.0;
        assert!(calm(&o, &mem(), &g, &[]).is_none());
        o.cubes[0].pos = [39.0, 0.0, 0.0];
        assert_eq!(calm(&o, &mem(), &g, &[]), Some((ItemKind::Cube, 20)));
        // Weapons, crates, and NPCs are gated too.
        o.cubes.clear();
        o.pickups = vec![pickup(7, WeaponKind::Lightning, [45.0, 0.0, 0.0])];
        o.me.weapons = [Some(gun()), None];
        assert!(calm(&o, &mem(), &g, &[]).is_none());
        o.pickups.clear();
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        o.crates = vec![item(60, [45.0, 0.0, 0.0])];
        o.mechs = vec![npc_at(50, [0.0, 45.0, 0.0])];
        assert!(calm(&o, &mem(), &g, &[]).is_none());
        // No next circle yet: no gate.
        o.zone.next_radius = 0.0;
        o.crates.clear();
        o.mechs.clear();
        o.cubes = vec![item(20, [45.0, 0.0, 0.0])];
        assert_eq!(calm(&o, &mem(), &g, &[]), Some((ItemKind::Cube, 20)));
    }

    #[test]
    fn zone_safe_point_and_flags() {
        let mut o = base_obs();
        o.zone.next_center = [0.0, 0.0];
        o.zone.next_radius = 100.0;
        o.me.pos = [200.0, 0.0, 0.0];
        let (must, should, safe) = zone_core(&o, &NAV);
        assert!((safe[0] - 75.0).abs() < 1e-9 && safe[1].abs() < 1e-9);
        // ETA = 125/11.7*1.25+6 ≈ 19.4 s; shrink starts in 90 s.
        assert!(!must && !should);
        o.t = 80.0;
        assert!(zone_core(&o, &NAV).1);
        o.t = 95.0;
        assert!(zone_core(&o, &NAV).0);
        o.t = 10.0;
        o.me.in_zone = false;
        assert!(zone_core(&o, &NAV).0);
        // Tiny circle: exact centre, and no NaN standing on it.
        o.zone.next_radius = 20.0;
        o.me.pos = [0.0, 0.0, 0.0];
        let (_, _, safe) = zone_core(&o, &NAV);
        assert_eq!(safe, [0.0, 0.0, 0.0]);
        o.zone.next_radius = 100.0;
        let (_, _, safe) = zone_core(&o, &NAV);
        assert!(safe.iter().all(|v| v.is_finite()));
        // Near the live edge while shrinking.
        o.me.in_zone = true;
        o.t = 120.0;
        o.me.pos = [495.0, 0.0, 0.0];
        assert!(zone_core(&o, &NAV).0);
    }
}
