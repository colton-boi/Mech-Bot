//! `obs`: what one mech perceives at a decision.

use crate::{GunStats, LaserStats, LightningStats, MechStats, MortarStats, Vec2, Vec3, WeaponKind};
use serde::{Deserialize, Serialize};

fn is_false(b: &bool) -> bool {
    !*b
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Obs {
    /// Sim tick (20 per second). Echo it in the action.
    pub tick: u32,
    pub t: f64,
    /// Mechs still alive, NPCs not counted.
    pub alive: u32,
    #[serde(rename = "self")]
    pub me: SelfObs,
    /// Visible mechs and NPC drones.
    pub mechs: Vec<ActorObs>,
    pub projectiles: Vec<ProjectileObs>,
    pub crates: Vec<ItemObs>,
    pub cubes: Vec<ItemObs>,
    pub pickups: Vec<PickupObs>,
    pub sounds: Vec<SoundObs>,
    pub beams: Vec<BeamObs>,
    pub events: Vec<GameEvent>,
    pub zone: ZoneObs,
    pub offer: Option<Offer>,
    pub drops: Vec<DropObs>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SelfObs {
    pub id: u32,
    /// Feet, bottom center.
    pub pos: Vec3,
    pub vel: Vec3,
    /// Actual torso `[yaw, pitch]`. Shots leave along this.
    pub aim: Vec2,
    pub hp: f64,
    pub max_hp: f64,
    pub energy: f64,
    pub max_energy: f64,
    pub grounded: bool,
    pub cubes: u32,
    pub upgrades: Vec<String>,
    pub weapons: [Option<WeaponObs>; 2],
    pub in_zone: bool,
    pub stats: MechStats,
}

/// A held weapon. `stats` are that weapon's numbers after the holder's upgrades.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum WeaponObs {
    Gun {
        cooldown: f64,
        stats: GunStats,
        #[serde(default)]
        ammo: f64,
        /// Seconds of reload left, 0 when not reloading.
        #[serde(default)]
        reload: f64,
    },
    Mortar {
        cooldown: f64,
        stats: MortarStats,
    },
    Laser {
        cooldown: f64,
        /// Seconds charged so far, 0 when not charging.
        #[serde(default)]
        charge: f64,
        stats: LaserStats,
    },
    Lightning {
        cooldown: f64,
        stats: LightningStats,
    },
}

impl WeaponObs {
    pub fn kind(&self) -> WeaponKind {
        match self {
            Self::Gun { .. } => WeaponKind::Gun,
            Self::Mortar { .. } => WeaponKind::Mortar,
            Self::Laser { .. } => WeaponKind::Laser,
            Self::Lightning { .. } => WeaponKind::Lightning,
        }
    }

    /// Seconds until it can fire again.
    pub fn cooldown(&self) -> f64 {
        match self {
            Self::Gun { cooldown, .. } | Self::Mortar { cooldown, .. } | Self::Laser { cooldown, .. } | Self::Lightning { cooldown, .. } => *cooldown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ActorKind {
    Mech,
    Npc,
}

/// Another mech or an NPC drone in view. NPCs carry no `cubes`, `weapons`, or `charging`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActorObs {
    pub id: u32,
    pub kind: ActorKind,
    pub pos: Vec3,
    pub vel: Vec3,
    pub aim: Vec2,
    /// Present while `vision.show_enemy_hp` is true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_hp: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cubes: Option<u32>,
    /// Weapons held, empty slots omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weapons: Option<Vec<WeaponKind>>,
    /// Its laser is charging and will fire along `aim`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub charging: Option<bool>,
}

impl ActorObs {
    pub fn is_mech(&self) -> bool {
        self.kind == ActorKind::Mech
    }

    pub fn holds(&self, kind: WeaponKind) -> bool {
        self.weapons.as_deref().is_some_and(|w| w.contains(&kind))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProjectileKind {
    Gun,
    Mortar,
    Npc,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectileObs {
    pub id: u32,
    pub kind: ProjectileKind,
    pub pos: Vec3,
    pub vel: Vec3,
}

/// A crate (with `hp`) or a power cube (without).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ItemObs {
    pub id: u32,
    pub pos: Vec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hp: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PickupObs {
    pub id: u32,
    pub kind: WeaponKind,
    pub pos: Vec3,
}

open_enum!(SoundKind {
    Step = "step",
    Boost = "boost",
    Reload = "reload",
    Gun = "gun",
    Mortar = "mortar",
    Explosion = "explosion",
    LaserCharge = "laser_charge",
    Laser = "laser",
    Lightning = "lightning",
    Jet = "jet",
    Death = "death",
    Crate = "crate",
    NpcGun = "npc_gun",
    Drop = "drop",
});

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SoundDist {
    Near,
    Mid,
    Far,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SoundObs {
    pub kind: SoundKind,
    /// World-frame yaw toward the source, with Gaussian error.
    pub bearing: f64,
    pub dist: SoundDist,
    pub age: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BeamKind {
    Laser,
    Lightning,
}

/// An instant shot by another mech. `path` starts at its muzzle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BeamObs {
    pub by: u32,
    pub kind: BeamKind,
    pub path: Vec<Vec3>,
    pub age: f64,
}

open_enum!(
    /// What dealt damage: a weapon kind, `npc`, or `zone`.
    DamageKind {
        Gun = "gun",
        Mortar = "mortar",
        Laser = "laser",
        Lightning = "lightning",
        Npc = "npc",
        Zone = "zone",
    }
);

impl From<WeaponKind> for DamageKind {
    fn from(kind: WeaponKind) -> Self {
        match kind {
            WeaponKind::Gun => Self::Gun,
            WeaponKind::Mortar => Self::Mortar,
            WeaponKind::Laser => Self::Laser,
            WeaponKind::Lightning => Self::Lightning,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EffectName {
    Cookoff,
    Short,
    Concuss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TakeFailReason {
    Gone,
    Range,
    Have,
    Full,
    Busy,
}

/// Something that happened to this mech since its last observation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GameEvent {
    Damaged {
        amount: f64,
        weapon: DamageKind,
        /// World-frame yaw toward the source, null for the zone.
        bearing: Option<f64>,
        by: Option<u32>,
        #[serde(default, skip_serializing_if = "is_false")]
        rear: bool,
    },
    Hit {
        /// A mech, NPC, or crate id.
        target: u32,
        amount: f64,
        weapon: DamageKind,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        effect: Option<EffectName>,
        #[serde(default, skip_serializing_if = "is_false")]
        rear: bool,
    },
    Effect {
        effect: EffectName,
        by: Option<u32>,
    },
    Drop {
        id: u32,
        cubes: u32,
    },
    Kill {
        target: u32,
    },
    Pickup {
        cubes: u32,
    },
    Upgrade {
        id: String,
    },
    Took {
        id: u32,
        kind: WeaponKind,
        slot: u8,
    },
    Dropped {
        id: u32,
        kind: WeaponKind,
        slot: u8,
    },
    TakeFailed {
        id: u32,
        reason: TakeFailReason,
    },
    /// An event type added to the protocol after this build.
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DropObs {
    pub id: u32,
    pub pos: Vec3,
    pub radius: f64,
    /// Seconds until it is on the ground, 0 when capturable.
    pub lands_in: f64,
    pub capture_s: f64,
    /// Seconds the holder has banked of `capture_s`.
    pub progress: f64,
    pub holder: Option<u32>,
}

/// The current circle, the circle it shrinks to, and the shrink window.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ZoneObs {
    pub center: Vec2,
    pub radius: f64,
    pub next_center: Vec2,
    pub next_radius: f64,
    pub shrink_start: f64,
    pub shrink_end: f64,
    pub dps: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Offer {
    pub id: u32,
    pub choices: Vec<OfferChoice>,
    /// Match time at which choice 0 is applied.
    pub expires: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OfferChoice {
    pub id: String,
    pub name: String,
    pub desc: String,
}
