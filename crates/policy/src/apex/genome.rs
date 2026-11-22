//! [`ApexGenome`]: the 31 high-leverage behaviour genes of the apex bot, with
//! their bounds, JSON form, per-seat overrides, and the archetype presets.
//!
//! Unlike [`crate::Params`] (one `f32` per knob, tuned by CMA-ES), a genome
//! mixes discrete choices (enums, switches) with a few bounded numbers, so it
//! is searched by mutation and crossover rather than a continuous optimizer.
//! Every value that leaves this module has been through [`ApexGenome::clamp`].

use serde::{Deserialize, Serialize};

/// Which weapons to collect and keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeaponPref {
    /// Keep the gun and add a laser (sr, aqilc, the v8-econ family).
    GunLaser,
    /// Laser first, lightning second, and drop the gun for either (Lance, BD).
    LaserLightningDropGun,
    /// Keep the gun and add lightning.
    GunLightning,
}

/// Which upgrade order to follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UpgradeProfile {
    /// Plating, armor, nanites, then damage (the consensus order).
    DefenseFirst,
    /// Plating and armor, then laser damage and charge (giddy squad, v8-econ).
    LaserEcon,
    /// Plating, then gun magazine and damage (aqilc farm RANK).
    GunMagBrawl,
    /// Defense first, but take plating at once when HP is low (stormrush `SMART_UPG`).
    SmartHealDefense,
}

/// When to contest a supply drop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DropContest {
    /// Never go for drops.
    Off,
    /// Only to heal: HP below 95% and the drop is near (Th3r0n `drop_heal 0.95`).
    #[serde(rename = "heal_only_95", alias = "heal_only95")]
    HealOnly95,
    /// Any reachable drop that nobody else is capturing.
    AlwaysUncontested,
}

/// How a retreat starts and ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetreatMode {
    /// Enter below `retreat_hp`, stay until `retreat_exit_hp` (Lance, SR, DV).
    Latched,
    /// Each "losing" tick buys a 1.5 s pulse of retreat (sr `squad_lose_rule`).
    #[serde(rename = "pulse_1_5s")]
    Pulse15s,
    /// Latched, but fight on when the kill is faster than our time to live (sr last stand).
    TtlLastStand,
}

/// Strafe leg length while fighting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrafeStyle {
    /// Short random legs, about 0.35–1.0 s (TTUN, aqilc kiter, Lance `jink`).
    FastJink,
    /// Legs of 1.0–2.0 s.
    #[serde(rename = "mid_1_2s")]
    Mid12s,
    /// Full-speed legs of 3–5 s, the best real-board result (Juan REVIEW).
    #[serde(rename = "steady_3_5s")]
    Steady35s,
}

/// Reaction to an enemy laser charging at us.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LaserJukeMode {
    /// Ignore it (TTUN v10 reverted its laser dodge).
    None,
    /// Strafe harder and boost across the beam (giddy, Lance).
    StrafeBoost,
    /// Get out of its line of sight behind cover (SR, aqilc planner).
    BreakLosCover,
}

/// How the gun leads a moving target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GunLeadMode {
    /// Full velocity lead.
    #[serde(rename = "standard_1_0")]
    Standard10,
    /// Velocity lead scaled by 0.82 (aqilc uses 0.85).
    #[serde(rename = "damped_0_82")]
    Damped082,
    /// Acceleration-aware lead with a hit-feedback bandit over lead scales.
    AccelHitfbBandit,
}

/// The apex bot's behaviour genes. Field comments give the inclusive bounds
/// that [`ApexGenome::clamp`] enforces.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ApexGenome {
    // ---- loadout and upgrades ----
    /// Which weapons to collect and keep.
    pub weapon_pref: WeaponPref,
    /// Which upgrade order to follow.
    pub upgrade_profile: UpgradeProfile,
    /// Stop collecting cubes at this many. 6..=16. Past 10 a cube only adds size.
    pub cube_cap: u32,
    /// When to go for supply drops.
    pub drop_contest: DropContest,

    // ---- posture and FFA switch ----
    /// Switch to the FFA posture once an FFA is detected (deaths we did not cause).
    pub ffa_switch: bool,
    /// Proactively engage visible mechs within this many meters. 45..=190.
    pub engage_radius: f64,
    /// Proactively engage only "lone" mechs, with no other track near it or us.
    pub lone_only: bool,
    /// Walk at 4.6 m/s (below the step-sound threshold) when calm.
    pub quiet_walk: bool,
    /// Hold spot distance from the next centre, as a fraction of the next radius. 0.30..=0.78.
    pub home_ring_frac: f64,
    /// Hold-spot cost per unit of exposure to open sight lines. 0..=150.
    pub exposure_weight: f64,
    /// Hold-spot cost per nearby enemy or drop. 0..=60.
    pub crowd_pen: f64,

    // ---- retreat and survival ----
    /// How a retreat starts and ends.
    pub retreat_mode: RetreatMode,
    /// HP fraction that starts a retreat with a threat present. 0.25..=0.55.
    pub retreat_hp: f64,
    /// HP fraction that ends a latched retreat. 0.65..=0.92, and at least `retreat_hp + 0.15`.
    pub retreat_exit_hp: f64,
    /// Below this HP fraction, two or more threats start a retreat. 0.45..=0.85.
    pub multi_retreat_hp: f64,
    /// With no cover in reach, fight rather than flee in the open.
    pub commit_no_cover: bool,
    /// While retreating, turn the torso to face the attacker (rear hits deal ×1.5).
    pub rear_turn: bool,

    // ---- movement and dodging ----
    /// Preferred fighting range with a laser, meters. 75..=155.
    pub laser_range: f64,
    /// Preferred fighting range with only a gun, meters. 45..=115.
    pub gun_range: f64,
    /// Strafe leg length while fighting.
    pub strafe_style: StrafeStyle,
    /// Reverse the strafe at once when hit (Lance `hitflip`).
    pub hit_flip: bool,
    /// Weight of the projectile dodge against the commanded move. 0..=12.
    pub dodge_weight: f64,
    /// Reaction to an enemy laser charging at us.
    pub laser_juke_mode: LaserJukeMode,

    // ---- aiming and laser ----
    /// Pre-charge the laser behind cover and peek out to fire.
    pub laser_peek: bool,
    /// Release a partial charge when `k · damage ≥ target HP`. 0 disables, else 0.65..=1.0.
    pub laser_lethal_k: f64,
    /// Fire a held charge at the attacker when a retreat starts, rather than waste it.
    pub laser_retreat_dump: bool,
    /// How the gun leads a moving target.
    pub gun_lead_mode: GunLeadMode,

    // ---- economy ----
    /// Detour this far for a laser pickup, meters. 100..=300.
    pub laser_seek_r: f64,
    /// Detour this far for a cube during a fight, meters. 0..=95.
    pub fight_cube_r: f64,
    /// Farm NPCs and crates only above this HP fraction. 0.45..=0.90.
    pub farm_hp_min: f64,
    /// Close in and push while a laser target is in its post-shot cooldown (Lance `cdwin`).
    pub cdwin_push: bool,
}

impl Default for ApexGenome {
    fn default() -> Self {
        Self::ffa_default()
    }
}

/// Bounds that [`ApexGenome::clamp`] enforces, shared with docs and tests.
mod bounds {
    pub const CUBE_CAP: (u32, u32) = (6, 16);
    pub const ENGAGE_RADIUS: (f64, f64) = (45.0, 190.0);
    pub const HOME_RING_FRAC: (f64, f64) = (0.30, 0.78);
    pub const EXPOSURE_WEIGHT: (f64, f64) = (0.0, 150.0);
    pub const CROWD_PEN: (f64, f64) = (0.0, 60.0);
    pub const RETREAT_HP: (f64, f64) = (0.25, 0.55);
    pub const RETREAT_EXIT_HP: (f64, f64) = (0.65, 0.92);
    /// The least gap between `retreat_hp` and `retreat_exit_hp`.
    pub const RETREAT_GAP: f64 = 0.15;
    pub const MULTI_RETREAT_HP: (f64, f64) = (0.45, 0.85);
    pub const LASER_RANGE: (f64, f64) = (75.0, 155.0);
    pub const GUN_RANGE: (f64, f64) = (45.0, 115.0);
    pub const DODGE_WEIGHT: (f64, f64) = (0.0, 12.0);
    pub const LASER_LETHAL_K: (f64, f64) = (0.65, 1.0);
    pub const LASER_SEEK_R: (f64, f64) = (100.0, 300.0);
    pub const FIGHT_CUBE_R: (f64, f64) = (0.0, 95.0);
    pub const FARM_HP_MIN: (f64, f64) = (0.45, 0.90);
}

/// `v` clamped into `(lo, hi)`, or `fallback` when `v` is NaN or infinite.
fn bounded(v: f64, (lo, hi): (f64, f64), fallback: f64) -> f64 {
    if v.is_finite() {
        v.clamp(lo, hi)
    } else {
        fallback
    }
}

impl ApexGenome {
    /// Every name [`ApexGenome::preset`] knows.
    pub const PRESET_NAMES: &'static [&'static str] = &[
        "giddy_ghost",
        "giddy_aggro",
        "tigertown_snowball",
        "aqilc_far5",
        "poteto_steady",
        "lance_duel",
        "sr_ironclad",
        "selective_vulture",
    ];

    /// Forces every gene into its bounds. A NaN or infinite number takes the
    /// [`ApexGenome::ffa_default`] value. `laser_lethal_k` snaps to 0 below
    /// 0.325 (half of its lowest live value) and is clamped to 0.65..=1.0
    /// above, and `retreat_exit_hp` is raised to `retreat_hp + 0.15` if needed.
    pub fn clamp(&mut self) {
        use bounds::*;
        let d = Self::ffa_default();
        self.cube_cap = self.cube_cap.clamp(CUBE_CAP.0, CUBE_CAP.1);
        self.engage_radius = bounded(self.engage_radius, ENGAGE_RADIUS, d.engage_radius);
        self.home_ring_frac = bounded(self.home_ring_frac, HOME_RING_FRAC, d.home_ring_frac);
        self.exposure_weight = bounded(self.exposure_weight, EXPOSURE_WEIGHT, d.exposure_weight);
        self.crowd_pen = bounded(self.crowd_pen, CROWD_PEN, d.crowd_pen);
        self.retreat_hp = bounded(self.retreat_hp, RETREAT_HP, d.retreat_hp);
        self.retreat_exit_hp = bounded(self.retreat_exit_hp, RETREAT_EXIT_HP, d.retreat_exit_hp)
            .max(self.retreat_hp + RETREAT_GAP);
        self.multi_retreat_hp =
            bounded(self.multi_retreat_hp, MULTI_RETREAT_HP, d.multi_retreat_hp);
        self.laser_range = bounded(self.laser_range, LASER_RANGE, d.laser_range);
        self.gun_range = bounded(self.gun_range, GUN_RANGE, d.gun_range);
        self.dodge_weight = bounded(self.dodge_weight, DODGE_WEIGHT, d.dodge_weight);
        self.laser_lethal_k = if !self.laser_lethal_k.is_finite() {
            d.laser_lethal_k
        } else if self.laser_lethal_k < LASER_LETHAL_K.0 / 2.0 {
            0.0
        } else {
            self.laser_lethal_k
                .clamp(LASER_LETHAL_K.0, LASER_LETHAL_K.1)
        };
        self.laser_seek_r = bounded(self.laser_seek_r, LASER_SEEK_R, d.laser_seek_r);
        self.fight_cube_r = bounded(self.fight_cube_r, FIGHT_CUBE_R, d.fight_cube_r);
        self.farm_hp_min = bounded(self.farm_hp_min, FARM_HP_MIN, d.farm_hp_min);
    }

    /// `self` after [`ApexGenome::clamp`].
    fn clamped(mut self) -> Self {
        self.clamp();
        self
    }

    /// Reads a genome from JSON, `{"gene": value, ...}`. A missing gene keeps
    /// its [`ApexGenome::ffa_default`] value; an unknown gene or a value of the
    /// wrong type is an error, so a stale file is not half-applied. The result
    /// is clamped.
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str::<Self>(text)
            .map(Self::clamped)
            .map_err(|e| e.to_string())
    }

    /// Reads a genome file with [`ApexGenome::from_json`].
    pub fn load_file(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_json(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    /// The genome as pretty-printed JSON, every gene present.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self).map_err(|e| e.to_string())
    }

    /// `self` with a `/`-separated override spec applied left to right:
    ///
    /// - `key=value` sets one gene. Numbers parse as decimals, switches take
    ///   `true/false`, `on/off` or `1/0`, and enums take their JSON names.
    /// - `preset:<name>` replaces the whole genome with [`ApexGenome::preset`].
    /// - `file:<path>` replaces it with [`ApexGenome::load_file`]. The path
    ///   runs on across `/` until a segment holds `=` or starts a new
    ///   `preset:`/`file:`, so `file:/tmp/g.json/engage_radius=80` works.
    ///
    /// An unknown key, preset or value is an error. The result is clamped.
    pub fn with_overrides(&self, spec: &str) -> Result<Self, String> {
        let starts_item =
            |s: &str| s.contains('=') || s.starts_with("preset:") || s.starts_with("file:");
        let segments: Vec<&str> = spec.split('/').collect();
        let mut genome = self.clone();
        let mut i = 0;
        while i < segments.len() {
            let segment = segments[i];
            i += 1;
            if segment.is_empty() {
                continue;
            }
            if let Some(name) = segment.strip_prefix("preset:") {
                genome = Self::preset(name).ok_or_else(|| format!("unknown preset {name:?}"))?;
            } else if let Some(first) = segment.strip_prefix("file:") {
                let mut path = first.to_string();
                while i < segments.len() && !starts_item(segments[i]) {
                    path.push('/');
                    path.push_str(segments[i]);
                    i += 1;
                }
                genome = Self::load_file(std::path::Path::new(&path))?;
            } else {
                let (key, value) = segment
                    .split_once('=')
                    .ok_or_else(|| format!("{segment:?} is not key=value"))?;
                genome = genome.with_gene(key, value)?;
            }
        }
        Ok(genome.clamped())
    }

    /// `self` with the gene `key` set from the text `value`, going through the
    /// JSON form so names match [`ApexGenome::from_json`] exactly.
    fn with_gene(&self, key: &str, value: &str) -> Result<Self, String> {
        use serde_json::Value;
        let mut json = serde_json::to_value(self).map_err(|e| e.to_string())?;
        let map = json
            .as_object_mut()
            .ok_or("a genome serializes to a JSON object")?;
        let slot = map
            .get_mut(key)
            .ok_or_else(|| format!("unknown gene {key:?}"))?;
        let bad = || format!("gene {key:?}: {value:?} is not a valid value");
        *slot = match slot {
            Value::Bool(_) => Value::Bool(match value {
                "true" | "on" | "1" => true,
                "false" | "off" | "0" => false,
                _ => return Err(bad()),
            }),
            Value::Number(n) => {
                let x: f64 = value.parse().map_err(|_| bad())?;
                if !x.is_finite() {
                    return Err(format!("gene {key:?}: {value:?} is not a finite number"));
                }
                if n.is_u64() {
                    Value::from(x.round().clamp(0.0, f64::from(u32::MAX)) as u32)
                } else {
                    Value::from(x)
                }
            }
            Value::String(_) => Value::String(value.to_string()),
            _ => return Err(bad()),
        };
        serde_json::from_value(json).map_err(|e| format!("gene {key:?}: {e}"))
    }

    /// The genome for practice lobbies (score = 16 − place, mostly duels):
    /// fight readily, retreat in short pulses, and switch to the FFA posture
    /// when an FFA is detected.
    pub fn practice_default() -> Self {
        Self {
            weapon_pref: WeaponPref::GunLaser,
            upgrade_profile: UpgradeProfile::LaserEcon,
            cube_cap: 10,
            drop_contest: DropContest::HealOnly95,
            ffa_switch: true,
            engage_radius: 160.0,
            lone_only: false,
            quiet_walk: false,
            home_ring_frac: 0.55,
            exposure_weight: 40.0,
            crowd_pen: 20.0,
            retreat_mode: RetreatMode::Pulse15s,
            retreat_hp: 0.35,
            retreat_exit_hp: 0.80,
            multi_retreat_hp: 0.55,
            commit_no_cover: true,
            rear_turn: true,
            laser_range: 120.0,
            gun_range: 85.0,
            strafe_style: StrafeStyle::Steady35s,
            hit_flip: true,
            dodge_weight: 4.0,
            laser_juke_mode: LaserJukeMode::StrafeBoost,
            laser_peek: true,
            laser_lethal_k: 0.75,
            laser_retreat_dump: true,
            gun_lead_mode: GunLeadMode::Standard10,
            laser_seek_r: 240.0,
            fight_cube_r: 70.0,
            farm_hp_min: 0.55,
            cdwin_push: true,
        }
    }

    /// The genome for FFA matches: the evolved winner (`evo2` gen 1, cand 10):
    /// gun beside laser, smart-heal defense upgrades, fast-jink strafe with
    /// hit-flip, a tight 60 m engage radius, a low-exposure mid-ring hold,
    /// latched retreat, damped gun lead, and no drop contests or cube grabs.
    pub fn ffa_default() -> Self {
        Self {
            weapon_pref: WeaponPref::GunLaser,
            upgrade_profile: UpgradeProfile::SmartHealDefense,
            cube_cap: 10,
            drop_contest: DropContest::Off,
            ffa_switch: false,
            engage_radius: 60.0,
            lone_only: false,
            quiet_walk: true,
            home_ring_frac: 0.5548,
            exposure_weight: 120.0,
            crowd_pen: 40.0,
            retreat_mode: RetreatMode::Latched,
            retreat_hp: 0.38,
            retreat_exit_hp: 0.7771,
            multi_retreat_hp: 0.5728,
            commit_no_cover: false,
            rear_turn: false,
            laser_range: 85.0,
            gun_range: 70.0,
            strafe_style: StrafeStyle::FastJink,
            hit_flip: true,
            dodge_weight: 5.9185,
            laser_juke_mode: LaserJukeMode::StrafeBoost,
            laser_peek: true,
            laser_lethal_k: 0.8004,
            laser_retreat_dump: false,
            gun_lead_mode: GunLeadMode::Damped082,
            laser_seek_r: 174.3022,
            fight_cube_r: 0.0,
            farm_hp_min: 0.7,
            cdwin_push: false,
        }
    }

    /// The archetype preset called `name` (one of [`ApexGenome::PRESET_NAMES`]).
    /// Values come from `research/01..05`; a value outside a gene's bounds is
    /// taken at the nearest bound.
    pub fn preset(name: &str) -> Option<Self> {
        let base = Self::ffa_default();
        let genome = match name {
            // giddygiddyboy default "ghost" (research/01 §4–§6): gun kept in slot 0
            // with LaserFirst, plating first, engage only within 60 m, min-exposure
            // home at 0.7·next_r (120·exposure + 40·crowd), quiet walk, latched
            // 0.38/0.80 (multi 0.65), 0.3–1.0 s strafe, laser 85 m, peek and
            // 0.8 lethal release, strafe ×1.5 + boost vs a charging laser, no
            // drops, no fight cubes, farming above 70% HP.
            "giddy_ghost" => Self {
                weapon_pref: WeaponPref::GunLaser,
                upgrade_profile: UpgradeProfile::DefenseFirst,
                cube_cap: 10,
                drop_contest: DropContest::Off,
                ffa_switch: true,
                engage_radius: 60.0,
                lone_only: false,
                quiet_walk: true,
                home_ring_frac: 0.70,
                exposure_weight: 120.0,
                crowd_pen: 40.0,
                retreat_mode: RetreatMode::Latched,
                retreat_hp: 0.38,
                retreat_exit_hp: 0.80,
                multi_retreat_hp: 0.65,
                commit_no_cover: false,
                rear_turn: false,
                laser_range: 85.0,
                gun_range: 70.0,
                strafe_style: StrafeStyle::FastJink,
                hit_flip: false,
                dodge_weight: 6.0,
                laser_juke_mode: LaserJukeMode::StrafeBoost,
                laser_peek: true,
                laser_lethal_k: 0.8,
                laser_retreat_dump: false,
                gun_lead_mode: GunLeadMode::Standard10,
                laser_seek_r: 120.0,
                fight_cube_r: 0.0,
                farm_hp_min: 0.70,
                cdwin_push: false,
            },
            // giddygiddyboy squad/practice switch (research/01 §4–§6): LaserUp
            // upgrades, engage out to 190 m, non-latching 1.5 s lose-rule pulses
            // (≥2 near at <55%), laser 120 m, no peek, laser seek 240 m, fight
            // cubes within 70 m, farming above 50% HP.
            "giddy_aggro" => Self {
                upgrade_profile: UpgradeProfile::LaserEcon,
                ffa_switch: false,
                engage_radius: 190.0,
                quiet_walk: false,
                exposure_weight: 30.0,
                retreat_mode: RetreatMode::Pulse15s,
                retreat_hp: 0.35,
                multi_retreat_hp: 0.55,
                laser_range: 120.0,
                laser_peek: false,
                laser_seek_r: 240.0,
                fight_cube_r: 70.0,
                farm_hp_min: 0.50,
                ..Self::preset("giddy_ghost")?
            },
            // "tigertown" read as TTUN v8-econ and its cube-stack descendants
            // (research/02 §1, knob table; research/05 §6–§7), the 5.19 lineage:
            // gun always kept, laser 3.2 > lightning, cubes remembered without
            // limit (cap at the top bound), non-latching retreat at 0.35 with
            // ≥2 & <0.55 multi rule, 0.35–1.05 s strafe, laser 110 / gun 70,
            // full lead, no laser dodge (v10 reverted it), farming in 200 m calm.
            "tigertown_snowball" => Self {
                weapon_pref: WeaponPref::GunLaser,
                upgrade_profile: UpgradeProfile::LaserEcon,
                cube_cap: 16,
                drop_contest: DropContest::Off,
                engage_radius: 150.0,
                lone_only: false,
                quiet_walk: false,
                exposure_weight: 40.0,
                crowd_pen: 25.0,
                retreat_mode: RetreatMode::Pulse15s,
                retreat_hp: 0.35,
                retreat_exit_hp: 0.80,
                multi_retreat_hp: 0.55,
                commit_no_cover: false,
                rear_turn: false,
                laser_range: 110.0,
                gun_range: 70.0,
                strafe_style: StrafeStyle::FastJink,
                hit_flip: false,
                dodge_weight: 5.0,
                laser_juke_mode: LaserJukeMode::None,
                laser_peek: false,
                laser_lethal_k: 0.0,
                laser_retreat_dump: false,
                gun_lead_mode: GunLeadMode::Standard10,
                laser_seek_r: 300.0,
                fight_cube_r: 60.0,
                farm_hp_min: 0.60,
                cdwin_push: false,
                ..base
            },
            // aqilc far5 and its farx branches (research/02; research/05 §6–§7):
            // laser first while keeping the gun, gun_mag-heavy farm ranks, blank
            // cubes at 10, avoid drops, fight only aware enemies (or d ≤ 75),
            // long 90–130 m laser band and 60–110 m gun band, kiter ttd/ttk
            // retreat with any extra aware threat, +25 per extra watcher, rear
            // torso turn, 0.35–1.1 s strafe, lead ×0.85, laser cost planner,
            // kill-cube rush to 95 m, farming at HP ≥ 0.6.
            "aqilc_far5" => Self {
                weapon_pref: WeaponPref::GunLaser,
                upgrade_profile: UpgradeProfile::GunMagBrawl,
                cube_cap: 10,
                drop_contest: DropContest::Off,
                engage_radius: 75.0,
                lone_only: true,
                quiet_walk: false,
                home_ring_frac: 0.50,
                exposure_weight: 60.0,
                crowd_pen: 25.0,
                retreat_mode: RetreatMode::TtlLastStand,
                retreat_hp: 0.25,
                retreat_exit_hp: 0.85,
                multi_retreat_hp: 0.85,
                commit_no_cover: false,
                rear_turn: true,
                laser_range: 130.0,
                gun_range: 110.0,
                strafe_style: StrafeStyle::FastJink,
                hit_flip: false,
                dodge_weight: 6.0,
                laser_juke_mode: LaserJukeMode::BreakLosCover,
                laser_peek: false,
                laser_lethal_k: 0.0,
                laser_retreat_dump: false,
                gun_lead_mode: GunLeadMode::Damped082,
                laser_seek_r: 170.0,
                fight_cube_r: 95.0,
                farm_hp_min: 0.60,
                cdwin_push: false,
                ..base
            },
            // "poteto" has no matching note; this is the steady-strafe archetype
            // built from the real-board movement evidence (research/05 §1–§2,
            // §6–§7): 3–5 s full-speed legs plus hit-flip, quiet walk like the
            // FFA leaders, mid-ring settle at 0.55·r, `many:2` multi retreat,
            // commit with no cover, rear turn, smart plating, heal-only drops,
            // 0.75 lethal release, cdwin push, immediate kill-cube grabs.
            "poteto_steady" => Self {
                upgrade_profile: UpgradeProfile::SmartHealDefense,
                engage_radius: 120.0,
                lone_only: false,
                laser_range: 110.0,
                ..base
            },
            // LanceHuang (research/02, research/03, research/05 §2–§6): laser and
            // lightning with the gun dropped first, nano upgrades, past 10 cubes
            // only near a threshold, drops within 170 m, "patient" engage at
            // d < 55, latched 0.38/0.80 with many:2, jink + hitflip, beam-corridor
            // sidestep with boost, pre-charge, laser 110 / gun 85, +0.05 s lead,
            // loot to 240 m, cubes to 70 m, drones above 0.55, cdwin on, and
            // normal walking speed (sneak scored 4.54 against 4.67).
            "lance_duel" => Self {
                weapon_pref: WeaponPref::LaserLightningDropGun,
                upgrade_profile: UpgradeProfile::DefenseFirst,
                cube_cap: 12,
                drop_contest: DropContest::AlwaysUncontested,
                engage_radius: 55.0,
                lone_only: false,
                quiet_walk: false,
                home_ring_frac: 0.55,
                exposure_weight: 100.0,
                crowd_pen: 30.0,
                retreat_mode: RetreatMode::Latched,
                retreat_hp: 0.38,
                retreat_exit_hp: 0.80,
                multi_retreat_hp: 0.60,
                commit_no_cover: false,
                rear_turn: false,
                laser_range: 110.0,
                gun_range: 85.0,
                strafe_style: StrafeStyle::FastJink,
                hit_flip: true,
                dodge_weight: 6.0,
                laser_juke_mode: LaserJukeMode::StrafeBoost,
                laser_peek: true,
                laser_lethal_k: 0.0,
                laser_retreat_dump: false,
                gun_lead_mode: GunLeadMode::Standard10,
                laser_seek_r: 240.0,
                fight_cube_r: 70.0,
                farm_hp_min: 0.55,
                cdwin_push: true,
                ..base
            },
            // Sergio33113 sr-agent (research/03; research/05 §3, §6–§7): gun +
            // laser (keeping the gun was worth +1.07), plating > nanites first,
            // cubes until 21 (top bound 16), drops skipped when contested, latched
            // 0.45/0.85 with 2 hostiles at < 0.7, 0.35–1.1 s strafe with a flip
            // on inbound bullets, solved laser range around 125, blind
            // pre-charge, lethal k 0.7, boost to cover vs lasers, heal mode holds
            // fire, no mid-fight cubes, drones above 0.8, no quiet walk.
            "sr_ironclad" => Self {
                weapon_pref: WeaponPref::GunLaser,
                upgrade_profile: UpgradeProfile::DefenseFirst,
                cube_cap: 16,
                drop_contest: DropContest::AlwaysUncontested,
                engage_radius: 150.0,
                lone_only: false,
                quiet_walk: false,
                home_ring_frac: 0.55,
                exposure_weight: 80.0,
                crowd_pen: 20.0,
                retreat_mode: RetreatMode::Latched,
                retreat_hp: 0.45,
                retreat_exit_hp: 0.85,
                multi_retreat_hp: 0.70,
                commit_no_cover: false,
                rear_turn: false,
                laser_range: 125.0,
                gun_range: 90.0,
                strafe_style: StrafeStyle::FastJink,
                hit_flip: true,
                dodge_weight: 5.0,
                laser_juke_mode: LaserJukeMode::BreakLosCover,
                laser_peek: true,
                laser_lethal_k: 0.7,
                laser_retreat_dump: false,
                gun_lead_mode: GunLeadMode::Standard10,
                laser_seek_r: 200.0,
                fight_cube_r: 0.0,
                farm_hp_min: 0.80,
                cdwin_push: false,
                ..base
            },
            // biruk `Selective` FFA bot plus charliescull's vulture standoff
            // (research/03; research/04): survival first, laser > lightning with
            // the gun dropped, engage only lone mechs at close range, quiet walk,
            // anchor at 0.72·next_r with maximum exposure and crowd costs, 0.30
            // entry with a renewing retreat timer, 0.3–0.7 s strafe, EMA-damped
            // lead, 0.5–0.9 s laser sidestep, cubes to 40 m, no drops.
            "selective_vulture" => Self {
                weapon_pref: WeaponPref::LaserLightningDropGun,
                upgrade_profile: UpgradeProfile::DefenseFirst,
                cube_cap: 10,
                drop_contest: DropContest::Off,
                ffa_switch: true,
                engage_radius: 45.0,
                lone_only: true,
                quiet_walk: true,
                home_ring_frac: 0.72,
                exposure_weight: 150.0,
                crowd_pen: 60.0,
                retreat_mode: RetreatMode::Pulse15s,
                retreat_hp: 0.30,
                retreat_exit_hp: 0.80,
                multi_retreat_hp: 0.60,
                commit_no_cover: false,
                rear_turn: true,
                laser_range: 110.0,
                gun_range: 70.0,
                strafe_style: StrafeStyle::FastJink,
                hit_flip: false,
                dodge_weight: 6.0,
                laser_juke_mode: LaserJukeMode::StrafeBoost,
                laser_peek: false,
                laser_lethal_k: 0.8,
                laser_retreat_dump: false,
                gun_lead_mode: GunLeadMode::Damped082,
                laser_seek_r: 150.0,
                fight_cube_r: 40.0,
                farm_hp_min: 0.75,
                cdwin_push: false,
            },
            _ => return None,
        };
        Some(genome.clamped())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json_name<T: Serialize>(v: T) -> String {
        serde_json::to_value(v)
            .unwrap()
            .as_str()
            .unwrap()
            .to_string()
    }

    #[test]
    fn enum_names_are_exact() {
        assert_eq!(json_name(WeaponPref::GunLaser), "gun_laser");
        assert_eq!(
            json_name(WeaponPref::LaserLightningDropGun),
            "laser_lightning_drop_gun"
        );
        assert_eq!(json_name(WeaponPref::GunLightning), "gun_lightning");
        assert_eq!(json_name(UpgradeProfile::DefenseFirst), "defense_first");
        assert_eq!(json_name(UpgradeProfile::LaserEcon), "laser_econ");
        assert_eq!(json_name(UpgradeProfile::GunMagBrawl), "gun_mag_brawl");
        assert_eq!(
            json_name(UpgradeProfile::SmartHealDefense),
            "smart_heal_defense"
        );
        assert_eq!(json_name(DropContest::Off), "off");
        assert_eq!(json_name(DropContest::HealOnly95), "heal_only_95");
        assert_eq!(
            json_name(DropContest::AlwaysUncontested),
            "always_uncontested"
        );
        assert_eq!(json_name(RetreatMode::Latched), "latched");
        assert_eq!(json_name(RetreatMode::Pulse15s), "pulse_1_5s");
        assert_eq!(json_name(RetreatMode::TtlLastStand), "ttl_last_stand");
        assert_eq!(json_name(StrafeStyle::FastJink), "fast_jink");
        assert_eq!(json_name(StrafeStyle::Mid12s), "mid_1_2s");
        assert_eq!(json_name(StrafeStyle::Steady35s), "steady_3_5s");
        assert_eq!(json_name(LaserJukeMode::None), "none");
        assert_eq!(json_name(LaserJukeMode::StrafeBoost), "strafe_boost");
        assert_eq!(json_name(LaserJukeMode::BreakLosCover), "break_los_cover");
        assert_eq!(json_name(GunLeadMode::Standard10), "standard_1_0");
        assert_eq!(json_name(GunLeadMode::Damped082), "damped_0_82");
        assert_eq!(
            json_name(GunLeadMode::AccelHitfbBandit),
            "accel_hitfb_bandit"
        );
        let alias: DropContest = serde_json::from_str("\"heal_only95\"").unwrap();
        assert_eq!(alias, DropContest::HealOnly95);
    }

    #[test]
    fn defaults_and_presets_are_clamped_and_round_trip() {
        let mut all = vec![
            ApexGenome::default(),
            ApexGenome::practice_default(),
            ApexGenome::ffa_default(),
        ];
        for name in ApexGenome::PRESET_NAMES {
            all.push(ApexGenome::preset(name).unwrap_or_else(|| panic!("preset {name} missing")));
        }
        for g in all {
            let mut c = g.clone();
            c.clamp();
            assert_eq!(c, g, "not a fixed point of clamp");
            assert_eq!(ApexGenome::from_json(&g.to_json().unwrap()).unwrap(), g);
        }
        assert!(ApexGenome::preset("nope").is_none());
        assert_eq!(ApexGenome::default(), ApexGenome::ffa_default());
    }

    #[test]
    fn clamp_handles_non_finite_and_rules() {
        let mut g = ApexGenome {
            engage_radius: f64::NAN,
            laser_range: f64::INFINITY,
            gun_range: -1e9,
            cube_cap: 99,
            retreat_hp: 0.55,
            retreat_exit_hp: 0.65,
            laser_lethal_k: 0.3,
            ..ApexGenome::default()
        };
        g.clamp();
        let d = ApexGenome::ffa_default();
        assert_eq!(g.engage_radius, d.engage_radius);
        assert_eq!(g.laser_range, d.laser_range);
        assert_eq!(g.gun_range, 45.0);
        assert_eq!(g.cube_cap, 16);
        assert!((g.retreat_exit_hp - 0.70).abs() < 1e-12);
        assert_eq!(g.laser_lethal_k, 0.0);
        let once = g.clone();
        g.clamp();
        assert_eq!(g, once, "clamp is idempotent");

        for (k, want) in [(0.33, 0.65), (0.5, 0.65), (1.4, 1.0), (0.9, 0.9)] {
            let mut g = ApexGenome {
                laser_lethal_k: k,
                ..ApexGenome::default()
            };
            g.clamp();
            assert_eq!(g.laser_lethal_k, want, "k = {k}");
        }
    }

    #[test]
    fn json_rejects_unknown_and_fills_missing() {
        assert!(ApexGenome::from_json(r#"{"bogus": 1}"#).is_err());
        assert!(ApexGenome::from_json(r#"{"retreat_mode": "sideways"}"#).is_err());
        let g =
            ApexGenome::from_json(r#"{"engage_radius": 500, "strafe_style": "mid_1_2s"}"#).unwrap();
        assert_eq!(g.engage_radius, 190.0);
        assert_eq!(g.strafe_style, StrafeStyle::Mid12s);
        assert_eq!(g.cube_cap, ApexGenome::ffa_default().cube_cap);
    }

    #[test]
    fn overrides() {
        let base = ApexGenome::default();
        let g = base
            .with_overrides("engage_radius=80/quiet_walk=off/retreat_mode=pulse_1_5s/cube_cap=12.4")
            .unwrap();
        assert_eq!(g.engage_radius, 80.0);
        assert!(!g.quiet_walk);
        assert_eq!(g.retreat_mode, RetreatMode::Pulse15s);
        assert_eq!(g.cube_cap, 12);
        assert_eq!(base.with_overrides("").unwrap(), base);

        let p = base.with_overrides("preset:lance_duel/hit_flip=0").unwrap();
        assert_eq!(p.weapon_pref, WeaponPref::LaserLightningDropGun);
        assert!(!p.hit_flip);

        assert!(base.with_overrides("bogus=1").is_err());
        assert!(base.with_overrides("engage_radius=far").is_err());
        assert!(base.with_overrides("engage_radius=nan").is_err());
        assert!(base.with_overrides("quiet_walk=maybe").is_err());
        assert!(base.with_overrides("strafe_style=wobble").is_err());
        assert!(base.with_overrides("preset:nope").is_err());
        assert!(base.with_overrides("engage_radius").is_err());
    }

    #[test]
    fn override_file_path_spans_slashes() {
        let dir = std::env::temp_dir().join(format!("apex_genome_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("g.json");
        std::fs::write(&path, r#"{"laser_range": 150, "rear_turn": false}"#).unwrap();
        let spec = format!(
            "preset:sr_ironclad/file:{}/engage_radius=70",
            path.display()
        );
        let g = ApexGenome::default().with_overrides(&spec).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(g.laser_range, 150.0);
        assert!(!g.rear_turn);
        assert_eq!(g.engage_radius, 70.0);
        // The file replaces the preset entirely, so other genes are defaults.
        assert_eq!(g.retreat_hp, ApexGenome::ffa_default().retreat_hp);
    }
}
