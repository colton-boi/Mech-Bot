//! Shared plain-data types and small math helpers for the apex bot. Everything
//! here is `Copy` or fixed-size, so a decision allocates nothing for them.

use proto::WeaponKind;

/// A deterministic 64-bit PRNG (xorshift64*), seeded through splitmix64 so
/// nearby `(seed, slot)` pairs give unrelated streams.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rng64 {
    state: u64,
}

/// One splitmix64 step: advances `x` and returns a well-mixed output.
fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng64 {
    /// A generator for match seed `seed` and seat `slot`.
    pub fn new(seed: u64, slot: u32) -> Self {
        let mut x = seed ^ u64::from(slot).wrapping_mul(0xD1B5_4A32_D192_ED03);
        let mut state = splitmix64(&mut x);
        while state == 0 {
            // xorshift has a fixed point at zero.
            state = splitmix64(&mut x);
        }
        Self { state }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`, from the top 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[lo, hi)`.
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    /// True with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

/// `a` wrapped into `[-π, π)`. Non-finite input gives NaN rather than hanging.
pub fn wrap_pi(a: f64) -> f64 {
    use std::f64::consts::{PI, TAU};
    (a + PI).rem_euclid(TAU) - PI
}

/// Horizontal distance between two 3-D points (z ignored).
pub fn dist2d(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Full 3-D distance.
pub fn dist3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (dx, dy, dz) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Distance between two 2-D points.
pub fn dist_xy(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// `v` limited to `[lo, hi]`. Unlike [`f64::clamp`] it never panics when
/// `lo > hi` (it returns `lo`), and NaN stays NaN.
pub fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    let v = if v > hi { hi } else { v };
    if v < lo {
        lo
    } else {
        v
    }
}

/// What the bot is doing this decision, in priority order: an earlier mode
/// preempts a later one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Mode {
    /// Jet off an isolated roof or out of a dead end.
    Escape,
    /// Break out of being physically stuck.
    Unstuck,
    /// Must head into the zone now or take zone damage.
    ZoneMust,
    /// Fall back to cover to heal.
    Retreat,
    /// Go for or hold a supply drop.
    Drop,
    /// Engage a mech.
    Fight,
    /// Grab the cubes a kill just dropped.
    KillCube,
    /// Pick up a weapon or cube.
    Loot,
    /// Shoot NPCs or crates for cubes.
    Farm,
    /// Drift toward the next zone with time to spare.
    ZoneShould,
    /// Hold a low-exposure spot.
    Home,
    /// Nothing better to do: look around the map.
    Explore,
}

/// What the bot knows about one enemy mech.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThreatInfo {
    /// The mech's actor id.
    pub id: u32,
    /// Last known position, meters.
    pub pos: [f64; 3],
    /// Estimated velocity, m/s.
    pub vel: [f64; 3],
    /// Its aim as `[yaw, pitch]`, radians.
    pub aim: [f64; 2],
    /// Hit points, as last seen.
    pub hp: f64,
    /// Maximum hit points.
    pub max_hp: f64,
    /// Cubes it carries.
    pub cubes: u32,
    /// Weapons in its two slots, when known.
    pub weapons: [Option<WeaponKind>; 2],
    /// Its laser is charging.
    pub charging: bool,
    /// Distance from us, meters.
    pub dist: f64,
    /// We have line of sight to it.
    pub los: bool,
    /// It is in this decision's observation.
    pub visible_now: bool,
    /// Match time it was last seen, seconds.
    pub last_seen_t: f64,
    /// Match time it last hit us, seconds (`-inf` if never).
    pub last_hit_us_t: f64,
    /// Match time of its last laser beam, seconds (`-inf` if never).
    pub last_beam_t: f64,
    /// A pseudo-threat placed along the bearing of damage from an unseen attacker.
    pub is_ghost: bool,
    /// It is aware of us: aiming near us or recently hit us.
    pub is_aware: bool,
    /// Estimated damage per second it can deal to us at this range.
    pub threat_dps: f64,
    /// Target choice score; lower is a better target.
    pub target_score: f64,
}

impl ThreatInfo {
    /// Hit points as a fraction of maximum, 0 when `max_hp` is not positive.
    pub fn hp_frac(&self) -> f64 {
        if self.max_hp > 0.0 {
            self.hp / self.max_hp
        } else {
            0.0
        }
    }
}

/// The kind of thing an [`ItemGoal`] points at.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ItemKind {
    /// A weapon pickup.
    Weapon,
    /// A cube.
    Cube,
    /// A crate to shoot open.
    Crate,
    /// A supply drop to capture.
    Drop,
    /// An NPC drone to farm.
    Npc,
}

/// A chosen item or farm target.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ItemGoal {
    /// What it is.
    pub kind: ItemKind,
    /// Its actor or pickup id.
    pub id: u32,
    /// Its position, meters.
    pub pos: [f64; 3],
    /// Utility used to pick it; higher is better.
    pub score: f64,
    /// Distance to stand at: 0 to touch it, more to shoot it from range.
    pub stand_off: f64,
}

/// Where to walk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NavGoal {
    /// Target position, meters.
    pub pos: [f64; 3],
    /// Within this many meters counts as arrived.
    pub radius: f64,
    /// Stop on the point itself rather than anywhere in reach.
    pub exact: bool,
}

/// This decision's aim and trigger output.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AimDecision {
    /// `[yaw, pitch]` to aim at, radians, or `None` to hold the current aim.
    pub aim: Option<[f64; 2]>,
    /// Fire weapon slot 0 and slot 1.
    pub fire: [bool; 2],
    /// Reload the gun.
    pub reload: bool,
    /// The mech the shot is meant for, if any.
    pub fire_target_id: Option<u32>,
    /// Fire was held back so regeneration is not reset.
    pub suppress_for_heal: bool,
}

/// This decision's movement output.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MoveDecision {
    /// Desired horizontal direction in world frame; length 0..=1 scales speed.
    pub move_dir: [f64; 2],
    /// Use the boost.
    pub boost: bool,
    /// Jump this decision.
    pub jump: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn rng_is_deterministic_and_slots_differ() {
        let mut a = Rng64::new(42, 3);
        let mut b = Rng64::new(42, 3);
        let mut c = Rng64::new(42, 4);
        let xs: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let ys: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        let zs: Vec<u64> = (0..8).map(|_| c.next_u64()).collect();
        assert_eq!(xs, ys);
        assert_ne!(xs, zs);
        let mut z = Rng64::new(0, 0);
        for _ in 0..1000 {
            let f = z.next_f64();
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn math_helpers() {
        assert!((wrap_pi(3.0 * PI) + PI).abs() < 1e-12);
        assert!((wrap_pi(-0.5) + 0.5).abs() < 1e-12);
        assert!(wrap_pi(f64::INFINITY).is_nan());
        assert!((wrap_pi(1e18).abs()) <= PI);
        assert_eq!(dist2d([0.0, 0.0, 9.0], [3.0, 4.0, -9.0]), 5.0);
        assert_eq!(dist3d([0.0, 0.0, 0.0], [2.0, 3.0, 6.0]), 7.0);
        assert_eq!(dist_xy([1.0, 1.0], [4.0, 5.0]), 5.0);
        assert_eq!(clamp(5.0, 0.0, 1.0), 1.0);
        assert_eq!(clamp(-5.0, 0.0, 1.0), 0.0);
        assert_eq!(clamp(0.5, 1.0, 0.0), 1.0);
        assert!(clamp(f64::NAN, 0.0, 1.0).is_nan());
        assert!(Mode::Escape < Mode::Explore);
    }
}
