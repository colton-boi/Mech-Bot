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

/// What the economy needs from the map.
trait NavView {
    /// Whether `to` can be walked to from `from` (same or main component).
    fn reachable(&self, from: [f64; 3], to: [f64; 3]) -> bool;
    /// Ground or roof height to stand on at `(x, y)`.
    fn ground_z(&self, x: f64, y: f64) -> f64;
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
}

impl NavView for ApexNav {
    fn reachable(&self, from: [f64; 3], to: [f64; 3]) -> bool {
        // Cheap per-item check: the target sits at (or near) ground level, not
        // on a roof, on the main walkable area or in the same area as us.
        let ours = self.area_of(from[0], from[1]);
        let same_area = ours.is_some() && ours == self.area_of(to[0], to[1]);
        (self.is_main_xy(to[0], to[1]) || same_area)
            && to[2] <= self.ground_at(to[0], to[1]) + LEVEL_DZ
    }
    fn ground_z(&self, x: f64, y: f64) -> f64 {
        self.ground_at(x, y)
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
    let best = offer
        .choices
        .iter()
        .enumerate()
        .min_by_key(|(i, c)| (score(&c.id), *i))
        .map(|(i, _)| i)?;
    Some(best as u8)
}

// ----------------------------------------------------------------- weapons

/// How much `pref` values holding `kind`; 0 means never take it.
fn weapon_rank(pref: WeaponPref, kind: WeaponKind) -> u32 {
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
            ra.cmp(&rb)
                .then(dist2d(b.pos, me).total_cmp(&dist2d(a.pos, me)))
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

/// Whether an item is allowed at all: not blacklisted, well inside the zone,
/// and on our level or reachable.
fn item_ok<N: NavView, M: MemView>(obs: &Obs, nav: &N, mem: &M, id: u32, pos: [f64; 3]) -> bool {
    if mem.blacklisted(id, obs.t) {
        return false;
    }
    let z = &obs.zone;
    if dist2d(pos, [z.center[0], z.center[1], 0.0]) > z.radius - ZONE_ITEM_MARGIN {
        return false;
    }
    (pos[2] - obs.me.pos[2]).abs() <= LEVEL_DZ || nav.reachable(obs.me.pos, pos)
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
fn remembered<'a, M: MemView>(
    obs: &Obs,
    mem: &'a M,
    kind: ItemKind,
    fresh_s: f64,
    seen: &[u32],
) -> impl Iterator<Item = &'a RememberedItem> {
    let t = obs.t;
    let seen = seen.to_vec();
    mem.items().iter().filter(move |it| {
        it.kind == kind && t - it.last_seen_t <= fresh_s && !seen.contains(&it.id)
    })
}

/// Weapon pickups in view plus fresh remembered ones: `(id, kind, pos)`.
fn weapon_candidates<M: MemView>(obs: &Obs, mem: &M) -> Vec<(u32, WeaponKind, [f64; 3])> {
    let mut out: Vec<_> = obs.pickups.iter().map(|p| (p.id, p.kind, p.pos)).collect();
    let seen: Vec<u32> = out.iter().map(|c| c.0).collect();
    out.extend(
        remembered(obs, mem, ItemKind::Weapon, WEAPON_FRESH_S, &seen)
            .filter_map(|it| it.weapon_kind.map(|k| (it.id, k, it.pos))),
    );
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

fn pick_drop<N: NavView, M: MemView>(
    obs: &Obs,
    nav: &N,
    mem: &M,
    genome: &ApexGenome,
) -> Option<ItemGoal> {
    let (max_d, slack) = match genome.drop_contest {
        DropContest::Off => return None,
        DropContest::HealOnly95 => {
            if hp_frac(obs) >= 0.95 {
                return None;
            }
            (250.0, 8.0)
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
            DropContest::HealOnly95 => ours_or_free || d.progress < 0.5 * d.capture_s,
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

fn pick_kill_cube<N: NavView, M: MemView>(
    obs: &Obs,
    nav: &N,
    mem: &M,
    genome: &ApexGenome,
) -> Option<ItemGoal> {
    if obs.t - mem.kill_t() > 4.0 || obs.me.cubes >= genome.cube_cap {
        return None;
    }
    let kill = mem.kill_pos().unwrap_or(obs.me.pos);
    let me = obs.me.pos;
    let mut best = None;
    for (id, pos) in item_candidates(obs, mem, ItemKind::Cube) {
        if dist2d(pos, kill) > 95.0
            || (pos[2] - kill[2]).abs() >= 4.0
            || !item_ok(obs, nav, mem, id, pos)
        {
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

fn pick_fight_detour<N: NavView, M: MemView>(
    obs: &Obs,
    nav: &N,
    mem: &M,
    genome: &ApexGenome,
    threats: &[ThreatInfo],
) -> Option<ItemGoal> {
    let me = obs.me.pos;
    let nearest = threats.iter().min_by(|a, b| a.dist.total_cmp(&b.dist));
    let mut best = None;
    if obs.me.cubes < genome.cube_cap && genome.fight_cube_r > 0.0 {
        let r = if hp_frac(obs) < 0.5 {
            genome.fight_cube_r * 0.5
        } else {
            genome.fight_cube_r
        };
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
    let lull = !threats
        .iter()
        .any(|t| t.dist < 50.0 || obs.t - t.last_hit_us_t < 2.0);
    let laser_r = if gun_only && lull {
        genome.laser_seek_r.min(120.0)
    } else {
        0.0
    };
    if obs.me.weapons.iter().any(Option::is_none) {
        for (id, kind, pos) in weapon_candidates(obs, mem) {
            let dist = dist2d(pos, me);
            let reach = if kind == WeaponKind::Laser {
                laser_r.max(45.0)
            } else {
                45.0
            };
            if dist > reach
                || take_plan(obs, genome.weapon_pref, kind).is_none()
                || !item_ok(obs, nav, mem, id, pos)
            {
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
    }
    best
}

/// No enemy mech within 150 m, seen in the last 5 s.
fn area_quiet(obs: &Obs, threats: &[ThreatInfo]) -> bool {
    let me = obs.me.pos;
    let recent = threats
        .iter()
        .any(|t| obs.t - t.last_seen_t <= 5.0 && dist2d(t.pos, me) <= 150.0);
    let visible = obs
        .mechs
        .iter()
        .any(|m| m.is_mech() && m.id != obs.me.id && dist2d(m.pos, me) <= 150.0);
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

fn pick_calm<N: NavView, M: MemView>(
    obs: &Obs,
    nav: &N,
    mem: &M,
    genome: &ApexGenome,
    threats: &[ThreatInfo],
) -> Option<ItemGoal> {
    let me = obs.me.pos;
    let pref = genome.weapon_pref;
    let top = top_wanted(pref);
    // Only a gun (or nothing) in hand: grab any secondary fast.
    let unarmed = obs.me.weapons[1].is_none();
    let mut best = None;
    for (id, kind, pos) in weapon_candidates(obs, mem) {
        let dist = dist2d(pos, me);
        if take_plan(obs, pref, kind).is_none() || !item_ok(obs, nav, mem, id, pos) {
            continue;
        }
        let base = if kind == top && dist <= genome.laser_seek_r {
            // Early rush for the weapon the build is centred on.
            800.0 - 0.5 * dist
        } else if dist <= 150.0 {
            300.0 + f64::from(weapon_rank(pref, kind)) - dist
        } else {
            continue;
        };
        let secondary = matches!(kind, WeaponKind::Laser | WeaponKind::Lightning);
        let score = if unarmed && secondary {
            base + UNARMED_SECONDARY_BONUS
        } else {
            base
        };
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
            if !item_ok(obs, nav, mem, id, pos) {
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
    if hp_frac(obs) >= genome.farm_hp_min && area_quiet(obs, threats) {
        let npc_stand = if holds(obs, WeaponKind::Lightning) {
            25.0
        } else {
            45.0
        };
        for n in obs.mechs.iter().filter(|m| m.kind == ActorKind::Npc) {
            if !farm_zone_ok(obs, n.pos) || !item_ok(obs, nav, mem, n.id, n.pos) {
                continue;
            }
            let dist = dist2d(n.pos, me);
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
            if !farm_zone_ok(obs, pos) || !item_ok(obs, nav, mem, id, pos) {
                continue;
            }
            let dist = dist2d(pos, me);
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

fn loot_or_farm<N: NavView, M: MemView>(
    obs: &Obs,
    nav: &N,
    mem: &M,
    genome: &ApexGenome,
    in_fight: bool,
    threats: &[ThreatInfo],
) -> Option<ItemGoal> {
    pick_drop(obs, nav, mem, genome)
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
pub fn choose_loot_or_farm(
    obs: &Obs,
    nav: &ApexNav,
    mem: &BotMemory,
    genome: &ApexGenome,
    in_fight: bool,
    threats: &[ThreatInfo],
) -> Option<ItemGoal> {
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

    let zone_must = !obs.me.in_zone
        || (shrinking && z.radius - d_cur <= 8.0)
        || (outside_next && ((shrinking && to_end <= eta) || (to_start > 0.0 && to_start <= eta)));
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
    use proto::{
        ActorObs, DropObs, GunStats, ItemObs, LaserStats, LightningStats, Offer, OfferChoice,
        PickupObs,
    };

    struct Nav {
        reach: bool,
    }
    impl NavView for Nav {
        fn reachable(&self, _: [f64; 3], _: [f64; 3]) -> bool {
            self.reach
        }
        fn ground_z(&self, _: f64, _: f64) -> f64 {
            0.0
        }
    }

    struct Mem {
        black: Vec<u32>,
        kill_t: f64,
        kill_pos: Option<[f64; 3]>,
        items: Vec<RememberedItem>,
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
    }

    const NAV: Nav = Nav { reach: true };

    fn mem() -> Mem {
        Mem {
            black: vec![],
            kill_t: f64::NEG_INFINITY,
            kill_pos: None,
            items: vec![],
        }
    }

    fn remembered_item(
        id: u32,
        kind: ItemKind,
        weapon_kind: Option<WeaponKind>,
        pos: [f64; 3],
        last_seen_t: f64,
    ) -> RememberedItem {
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
        let mut o = Obs {
            t: 10.0,
            ..Default::default()
        };
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
        o.pickups = vec![
            pickup(5, WeaponKind::Mortar, [1.0, 0.0, 0.0]),
            pickup(6, WeaponKind::Laser, [3.0, 0.0, 0.0]),
        ];
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
        let kinds = [
            WeaponKind::Gun,
            WeaponKind::Laser,
            WeaponKind::Lightning,
            WeaponKind::Mortar,
        ];
        for pref in [
            WeaponPref::GunLaser,
            WeaponPref::LaserLightningDropGun,
            WeaponPref::GunLightning,
        ] {
            let mut g = genome();
            g.weapon_pref = pref;
            for kind in kinds {
                let mut o = base_obs();
                o.me.weapons = [Some(gun()), None];
                o.pickups = vec![pickup(8, kind, [1.0, 0.0, 0.0])];
                let (take, drop) = pick_weapon_action(&o, &g);
                assert_eq!(drop, None, "{pref:?} dropped a slot for {kind:?}");
                assert_eq!(
                    take.is_some(),
                    kind != WeaponKind::Gun && kind != WeaponKind::Mortar,
                    "{pref:?} with {kind:?}"
                );
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
        assert_eq!(
            loot_or_farm(&o, &NAV, &mem(), &g, true, &far).map(|i| i.id),
            Some(6)
        );
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
        assert!(
            loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).is_none_or(|i| i.kind != ItemKind::Drop)
        );
        g.drop_contest = DropContest::AlwaysUncontested;
        let goal = loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).unwrap();
        assert_eq!(
            (goal.kind, goal.id, goal.stand_off),
            (ItemKind::Drop, 40, 0.0)
        );
        o.drops[0].holder = Some(77);
        assert!(pick_drop(&o, &NAV, &mem(), &g).is_none());
        g.drop_contest = DropContest::HealOnly95;
        assert!(pick_drop(&o, &NAV, &mem(), &g).is_none());
        o.me.hp = 0.8 * o.me.max_hp;
        assert!(pick_drop(&o, &NAV, &mem(), &g).is_some());
        // Lands far too late to wait for.
        o.drops[0].lands_in = 60.0;
        assert!(pick_drop(&o, &NAV, &mem(), &g).is_none());
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
        let m = Mem {
            black: vec![20],
            ..mem()
        };
        assert_eq!(loot_or_farm(&o, &NAV, &m, &g, false, &[]).unwrap().id, 21);
        o.me.cubes = 8;
        assert!(loot_or_farm(&o, &NAV, &mem(), &g, false, &[]).is_none());
        // Unreachable cube on a roof is skipped.
        o.me.cubes = 0;
        o.cubes = vec![item(22, [10.0, 0.0, 9.0])];
        assert!(loot_or_farm(&o, &Nav { reach: false }, &mem(), &g, false, &[]).is_none());
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
        assert_eq!(
            loot_or_farm(&o, &NAV, &stale, &g, false, &[]).unwrap().id,
            6
        );
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
            items: vec![remembered_item(
                70,
                ItemKind::Weapon,
                Some(WeaponKind::Laser),
                [-60.0, 0.0, 0.0],
                0.0,
            )],
            ..mem()
        };
        let goal = loot_or_farm(&o, &NAV, &m, &g, false, &[]).unwrap();
        assert_eq!(
            (goal.kind, goal.id, goal.pos),
            (ItemKind::Weapon, 70, [-60.0, 0.0, 0.0])
        );
        // Also reached for mid-fight in a lull.
        let far = [threat([90.0, 0.0, 0.0], 90.0, 10.0)];
        assert_eq!(
            loot_or_farm(&o, &NAV, &m, &g, true, &far).map(|i| i.id),
            Some(70)
        );
        // A weapon memory without a known kind is ignored.
        let unknown = Mem {
            items: vec![remembered_item(
                71,
                ItemKind::Weapon,
                None,
                [-60.0, 0.0, 0.0],
                0.0,
            )],
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
            items: vec![remembered_item(
                70,
                ItemKind::Weapon,
                Some(WeaponKind::Laser),
                [-60.0, 0.0, 0.0],
                9.0,
            )],
            ..mem()
        };
        assert_eq!(
            loot_or_farm(&o, &NAV, &m, &g, false, &[]).unwrap().pos,
            [30.0, 0.0, 0.0]
        );
    }

    #[test]
    fn remembered_cubes_and_crates_respect_freshness() {
        let mut o = base_obs();
        let mut g = genome();
        g.drop_contest = DropContest::Off;
        g.cube_cap = 8;
        let m = Mem {
            items: vec![remembered_item(
                80,
                ItemKind::Cube,
                None,
                [20.0, 0.0, 0.0],
                0.0,
            )],
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
        assert_eq!(
            (goal.kind, goal.id, goal.score),
            (ItemKind::Cube, 80, 880.0)
        );
        // Stale beyond 35 s.
        o.t = 36.0;
        assert!(loot_or_farm(&o, &NAV, &m, &g, false, &[]).is_none());

        // Crates: remembered for 45 s, farm checks still apply.
        o.t = 10.0;
        o.me.weapons = [Some(gun()), Some(laser(0.0))];
        let crates = Mem {
            items: vec![remembered_item(
                90,
                ItemKind::Crate,
                None,
                [40.0, 0.0, 0.0],
                0.0,
            )],
            ..mem()
        };
        let goal = loot_or_farm(&o, &NAV, &crates, &g, false, &[]).unwrap();
        assert_eq!(
            (goal.kind, goal.id, goal.stand_off),
            (ItemKind::Crate, 90, 26.0)
        );
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
