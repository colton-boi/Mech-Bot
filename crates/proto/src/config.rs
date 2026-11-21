//! `hello.config`: the complete rule set. Defaults are the values recorded in
//! `mechfight-kit/check/stream.jsonl.gz`; a test keeps them equal.

use crate::{SoundKind, Vec2, WeaponKind};
use serde::{Deserialize, Serialize};

stat_struct!(MatchCfg {
    tick_hz = 20,
    decide_every = 2,
    duration_s = 480,
    mechs = 16,
    spawn_min_separation = 130,
    spawn_zone_fraction = 0.8,
    arena_margin = 30,
});

stat_struct!(BotCfg {
    tick_budget_ms = 50,
    hello_budget_ms = 20000,
    coast_s = 1,
    offer_timeout_s = 5,
    max_line_bytes = 65536,
});

stat_struct!(
    /// `config.mech`, and `self.stats` (the same numbers after upgrades and cube scaling).
    MechStats {
        radius = 2.5,
        height = 9,
        eye_height = 8,
        muzzle_height = 7,
        hp = 1400,
        walk_speed = 11.7,
        boost_speed = 19.5,
        ground_accel = 58.5,
        air_accel = 12,
        gravity = 22,
        jump_speed = 11,
        jet_accel = 34,
        max_rise_speed = 10,
        step_up = 1.2,
        max_slope_deg = 50,
        energy = 100,
        energy_regen = 18,
        energy_regen_delay = 0.6,
        boost_drain = 17,
        jet_drain = 15,
        turn_rate = 4,
        pitch_rate = 3,
        min_pitch = -0.7,
        max_pitch = 1.3,
        regen_delay = 5,
        regen_per_s = 45,
        charge_move_mult = 0.5,
        self_damage_mult = 0.5,
        armor = 0,
        rear_mult = 1.5,
        front_arc_deg = 180,
        near_range_bonus = 0,
        vision_bonus = 0,
        hearing = 1,
    }
);

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionCfg {
    pub range: f64,
    pub fov_deg: f64,
    pub near_range: f64,
    pub canopy_blocks: bool,
    pub show_enemy_hp: bool,
    pub memory_s: f64,
}

impl Default for VisionCfg {
    fn default() -> Self {
        Self {
            range: 200.0,
            fov_deg: 120.0,
            near_range: 45.0,
            canopy_blocks: true,
            show_enemy_hp: true,
            memory_s: 5.0,
        }
    }
}

stat_struct!(
    /// Carry range in meters of each sound kind.
    SoundRange {
        step = 40,
        boost = 90,
        reload = 60,
        gun = 320,
        mortar = 420,
        explosion = 420,
        laser_charge = 140,
        laser = 380,
        lightning = 160,
        jet = 90,
        death = 600,
        #[serde(rename = "crate")]
        crate_ = 220,
        npc_gun = 200,
        drop = 600,
    }
);

impl SoundRange {
    /// Carry range of `kind`, 0 for a kind this build does not know.
    pub fn of(&self, kind: &SoundKind) -> f64 {
        match kind {
            SoundKind::Step => self.step,
            SoundKind::Boost => self.boost,
            SoundKind::Reload => self.reload,
            SoundKind::Gun => self.gun,
            SoundKind::Mortar => self.mortar,
            SoundKind::Explosion => self.explosion,
            SoundKind::LaserCharge => self.laser_charge,
            SoundKind::Laser => self.laser,
            SoundKind::Lightning => self.lightning,
            SoundKind::Jet => self.jet,
            SoundKind::Death => self.death,
            SoundKind::Crate => self.crate_,
            SoundKind::NpcGun => self.npc_gun,
            SoundKind::Drop => self.drop,
            SoundKind::Other(_) => 0.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SoundCfg {
    pub bearing_noise_deg: f64,
    pub near: f64,
    pub mid: f64,
    pub step_speed: f64,
    pub step_every: f64,
    pub range: SoundRange,
}

impl Default for SoundCfg {
    fn default() -> Self {
        Self {
            bearing_noise_deg: 7.0,
            near: 60.0,
            mid: 160.0,
            step_speed: 5.0,
            step_every: 0.6,
            range: SoundRange::default(),
        }
    }
}

stat_struct!(GunStats {
    damage = 45,
    cooldown = 0.24,
    speed = 252,
    range = 230,
    spread_deg = 0.6,
    mag = 12,
    reload = 2.4,
});

stat_struct!(MortarStats {
    damage = 200,
    splash = 160,
    splash_radius = 7,
    splash_edge = 0.3,
    cooldown = 2.6,
    speed = 72,
    gravity = 22,
    ttl = 7,
    knock = 16,
    knock_up = 9,
    drain = 45,
});

stat_struct!(LaserStats {
    damage = 315,
    charge = 0.75,
    min_charge = 0.15,
    min_frac = 0.35,
    hold_s = 2,
    cooldown = 1.65,
    range = 280,
    pierce = 0,
    charge_turn_mult = 1,
    cookoff = 0.6,
});

stat_struct!(LightningStats {
    damage = 60,
    cooldown = 0.25,
    range = 45,
    cone_deg = 30,
    chain = 2,
    chain_range = 14,
    chain_falloff = 0.7,
    energy = 3,
    short_s = 2,
});

/// `config.weapons`, and a mech's per-kind weapon numbers after its upgrades.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WeaponsCfg {
    pub gun: GunStats,
    pub mortar: MortarStats,
    pub laser: LaserStats,
    pub lightning: LightningStats,
}

impl WeaponsCfg {
    pub fn cooldown(&self, kind: WeaponKind) -> f64 {
        match kind {
            WeaponKind::Gun => self.gun.cooldown,
            WeaponKind::Mortar => self.mortar.cooldown,
            WeaponKind::Laser => self.laser.cooldown,
            WeaponKind::Lightning => self.lightning.cooldown,
        }
    }

    fn stat_mut(&mut self, kind: &str, name: &str) -> Option<&mut f64> {
        match kind {
            "gun" => self.gun.get_mut(name),
            "mortar" => self.mortar.get_mut(name),
            "laser" => self.laser.get_mut(name),
            "lightning" => self.lightning.get_mut(name),
            _ => None,
        }
    }
}

stat_struct!(ProjectileCfg { gun_radius = 0.3, npc_radius = 0.3 });

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CubesCfg {
    pub pickup_radius: f64,
    pub hp_per_cube: f64,
    pub damage_per_cube: f64,
    pub stat_cap: f64,
    pub death_drop_extra: f64,
    pub size_per_cube: f64,
    pub upgrade_thresholds: Vec<f64>,
}

impl Default for CubesCfg {
    fn default() -> Self {
        Self {
            pickup_radius: 3.5,
            hp_per_cube: 0.04,
            damage_per_cube: 0.04,
            stat_cap: 10.0,
            death_drop_extra: 2.0,
            size_per_cube: 0.05,
            upgrade_thresholds: vec![1.0, 3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0, 45.0],
        }
    }
}

stat_struct!(CratesCfg { count = 36, roof_fraction = 0.25, hp = 300, radius = 1.4, height = 2.4, cubes = 1 });

stat_struct!(LootWeights { gun = 0, mortar = 1, laser = 1, lightning = 1 });

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LootCfg {
    pub pickup_radius: f64,
    pub equip_s: f64,
    pub floor_count: f64,
    pub floor_roof_fraction: f64,
    pub crate_weapon_chance: f64,
    pub death_drop: f64,
    pub weights: LootWeights,
}

impl Default for LootCfg {
    fn default() -> Self {
        Self {
            pickup_radius: 4.0,
            equip_s: 1.0,
            floor_count: 40.0,
            floor_roof_fraction: 0.25,
            crate_weapon_chance: 0.5,
            death_drop: 1.0,
            weights: LootWeights::default(),
        }
    }
}

stat_struct!(NpcCfg {
    count = 12,
    hp = 350,
    radius = 1.8,
    height = 3.5,
    speed = 5,
    leash = 30,
    chase_leash = 60,
    aggro_range = 45,
    deaggro_range = 90,
    spawn_clear = 0,
    damage = 25,
    cooldown = 1,
    projectile_speed = 90,
    spread_deg = 2.5,
    cubes = 2,
});

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DropsCfg {
    pub times: Vec<f64>,
    pub fall_s: f64,
    pub radius: f64,
    pub capture_s: f64,
    pub cubes: f64,
    pub repair: f64,
    pub spread: f64,
}

impl Default for DropsCfg {
    fn default() -> Self {
        Self {
            times: vec![70.0, 140.0, 210.0],
            fall_s: 25.0,
            radius: 8.0,
            capture_s: 4.0,
            cubes: 3.0,
            repair: 1.0,
            spread: 0.7,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZonePhase {
    pub hold_s: f64,
    pub shrink_s: f64,
    pub radius: f64,
    pub dps: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ZoneCfg {
    pub start_center: Vec2,
    pub start_radius: f64,
    pub phases: Vec<ZonePhase>,
}

impl Default for ZoneCfg {
    fn default() -> Self {
        let phase = |hold_s, shrink_s, radius, dps| ZonePhase { hold_s, shrink_s, radius, dps };
        Self {
            start_center: [41.3, -72.8],
            start_radius: 560.0,
            phases: vec![
                phase(60.0, 50.0, 380.0, 20.0),
                phase(45.0, 40.0, 240.0, 35.0),
                phase(35.0, 35.0, 140.0, 60.0),
                phase(30.0, 30.0, 70.0, 100.0),
                phase(25.0, 25.0, 25.0, 160.0),
                phase(15.0, 25.0, 0.0, 260.0),
            ],
        }
    }
}

stat_struct!(
    /// Caps on `action.debug`; the server silently cuts each drawing to these.
    DebugCaps {
        max_lines = 64,
        max_points = 64,
        max_text = 16,
        max_text_chars = 80,
        max_bytes = 8192,
        max_match_bytes = 8000000,
        hold_s = 0.5,
    }
);

stat_struct!(ReplayCfg { frame_every = 2 });

/// One `mul` or `add` on a stat addressed as `mech.<name>` or `weapons.<kind>.<name>`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Effect {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mul: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add: Option<f64>,
}

impl Effect {
    /// Applies the effect. Returns false, changing nothing, for a path this build does not know.
    pub fn apply(&self, mech: &mut MechStats, weapons: &mut WeaponsCfg) -> bool {
        let mut parts = self.path.split('.');
        let stat = match (parts.next(), parts.next(), parts.next(), parts.next()) {
            (Some("mech"), Some(name), None, None) => mech.get_mut(name),
            (Some("weapons"), Some(kind), Some(name), None) => weapons.stat_mut(kind, name),
            _ => None,
        };
        let Some(stat) = stat else { return false };
        if let Some(mul) = self.mul {
            *stat *= mul;
        }
        if let Some(add) = self.add {
            *stat += add;
        }
        true
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpgradeDef {
    pub id: String,
    pub name: String,
    pub desc: String,
    /// Offered only while the mech holds this weapon.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requires: Option<WeaponKind>,
    pub max: f64,
    pub effects: Vec<Effect>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    #[serde(rename = "match")]
    pub match_: MatchCfg,
    pub bot: BotCfg,
    pub mech: MechStats,
    pub vision: VisionCfg,
    pub sound: SoundCfg,
    pub weapons: WeaponsCfg,
    pub projectile: ProjectileCfg,
    pub cubes: CubesCfg,
    pub crates: CratesCfg,
    pub loot: LootCfg,
    pub npc: NpcCfg,
    /// Absent means the match has no supply drops.
    #[serde(default = "no_drops", skip_serializing_if = "Option::is_none")]
    pub drops: Option<DropsCfg>,
    pub zone: ZoneCfg,
    pub debug: DebugCaps,
    pub replay: ReplayCfg,
    pub upgrades: Vec<UpgradeDef>,
}

fn no_drops() -> Option<DropsCfg> {
    None
}

impl Default for Config {
    fn default() -> Self {
        Self {
            match_: MatchCfg::default(),
            bot: BotCfg::default(),
            mech: MechStats::default(),
            vision: VisionCfg::default(),
            sound: SoundCfg::default(),
            weapons: WeaponsCfg::default(),
            projectile: ProjectileCfg::default(),
            cubes: CubesCfg::default(),
            crates: CratesCfg::default(),
            loot: LootCfg::default(),
            npc: NpcCfg::default(),
            drops: Some(DropsCfg::default()),
            zone: ZoneCfg::default(),
            debug: DebugCaps::default(),
            replay: ReplayCfg::default(),
            upgrades: default_upgrades(),
        }
    }
}

impl Config {
    /// Seconds per sim tick.
    pub fn dt(&self) -> f64 {
        1.0 / self.match_.tick_hz
    }

    pub fn upgrade(&self, id: &str) -> Option<&UpgradeDef> {
        self.upgrades.iter().find(|u| u.id == id)
    }
}

fn default_upgrades() -> Vec<UpgradeDef> {
    use WeaponKind::{Gun, Laser, Lightning, Mortar};
    let mul = |path: &str, v: f64| Effect {
        path: path.to_owned(),
        mul: Some(v),
        add: None,
    };
    let add = |path: &str, v: f64| Effect {
        path: path.to_owned(),
        mul: None,
        add: Some(v),
    };
    let up = |id: &str, name: &str, desc: &str, requires: Option<WeaponKind>, max: u32, effects: Vec<Effect>| UpgradeDef {
        id: id.to_owned(),
        name: name.to_owned(),
        desc: desc.to_owned(),
        requires,
        max: f64::from(max),
        effects,
    };
    vec![
        up("plating", "Composite plating", "+15% max HP", None, 3, vec![mul("mech.hp", 1.15)]),
        up("armor", "Reactive armor", "-8% damage taken", None, 3, vec![add("mech.armor", 0.08)]),
        up(
            "servos",
            "Leg servos",
            "+8% walk and boost speed",
            None,
            3,
            vec![mul("mech.walk_speed", 1.08), mul("mech.boost_speed", 1.08)],
        ),
        up("capacitor", "Capacitor bank", "+30 max energy", None, 3, vec![add("mech.energy", 30.0)]),
        up("reactor", "Reactor tune", "+30% energy regen", None, 3, vec![mul("mech.energy_regen", 1.3)]),
        up(
            "jets",
            "Efficient jets",
            "-25% boost and jet energy drain",
            None,
            2,
            vec![mul("mech.boost_drain", 0.75), mul("mech.jet_drain", 0.75)],
        ),
        up(
            "nanites",
            "Repair nanites",
            "+60% out-of-combat regen, starts 1.5 s sooner",
            None,
            2,
            vec![mul("mech.regen_per_s", 1.6), add("mech.regen_delay", -1.5)],
        ),
        up(
            "gyros",
            "Torso gyros",
            "+30% aim turn rate",
            None,
            2,
            vec![mul("mech.turn_rate", 1.3), mul("mech.pitch_rate", 1.3)],
        ),
        up(
            "sensors",
            "Sensor suite",
            "+15 m proximity sensors, +25 m vision, halves sound bearing error",
            None,
            2,
            vec![add("mech.near_range_bonus", 15.0), add("mech.vision_bonus", 25.0), mul("mech.hearing", 0.5)],
        ),
        up(
            "gun_damage",
            "Tungsten rounds",
            "+15% arm gun damage",
            Some(Gun),
            3,
            vec![mul("weapons.gun.damage", 1.15)],
        ),
        up(
            "gun_rate",
            "Rotary feed",
            "+15% arm gun fire rate",
            Some(Gun),
            3,
            vec![mul("weapons.gun.cooldown", 0.87)],
        ),
        up(
            "gun_velocity",
            "Rail barrel",
            "+25% arm gun projectile speed",
            Some(Gun),
            2,
            vec![mul("weapons.gun.speed", 1.25), mul("weapons.gun.range", 1.15)],
        ),
        up(
            "gun_mag",
            "Drum magazine",
            "+8 rounds, -20% reload time",
            Some(Gun),
            2,
            vec![add("weapons.gun.mag", 8.0), mul("weapons.gun.reload", 0.8)],
        ),
        up(
            "mortar_splash",
            "Cluster shells",
            "+30% mortar splash radius",
            Some(Mortar),
            2,
            vec![mul("weapons.mortar.splash_radius", 1.3)],
        ),
        up(
            "mortar_reload",
            "Autoloader",
            "-18% mortar cooldown",
            Some(Mortar),
            3,
            vec![mul("weapons.mortar.cooldown", 0.82)],
        ),
        up(
            "mortar_velocity",
            "Long tube",
            "+20% shell speed",
            Some(Mortar),
            2,
            vec![mul("weapons.mortar.speed", 1.2)],
        ),
        up(
            "laser_charge",
            "Fast capacitors",
            "-25% laser charge time",
            Some(Laser),
            2,
            vec![mul("weapons.laser.charge", 0.75)],
        ),
        up(
            "laser_damage",
            "Focusing lens",
            "+18% laser damage",
            Some(Laser),
            3,
            vec![mul("weapons.laser.damage", 1.18)],
        ),
        up(
            "laser_pierce",
            "Phased beam",
            "Laser passes through one more target",
            Some(Laser),
            1,
            vec![add("weapons.laser.pierce", 1.0)],
        ),
        up(
            "lightning_chain",
            "Arc relay",
            "+1 lightning chain jump",
            Some(Lightning),
            2,
            vec![add("weapons.lightning.chain", 1.0)],
        ),
        up(
            "lightning_range",
            "Tesla coil",
            "+20% lightning range and cone",
            Some(Lightning),
            2,
            vec![mul("weapons.lightning.range", 1.2), mul("weapons.lightning.cone_deg", 1.2)],
        ),
        up(
            "lightning_efficiency",
            "Superconductor",
            "-35% lightning energy cost, +10% damage",
            Some(Lightning),
            2,
            vec![mul("weapons.lightning.energy", 0.65), mul("weapons.lightning.damage", 1.1)],
        ),
    ]
}
