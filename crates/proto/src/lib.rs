//! Serde types for the `mechfight/1` bot protocol (`mechfight-kit/protocol.md`,
//! `mechfight.schema.json`).
//!
//! These types are the only currency between the simulator, the policy, and the
//! live bot: the simulator emits [`Obs`] and consumes [`Action`], exactly as the
//! server does. Unknown fields are ignored on input.

/// A struct of named `f64` numbers with protocol defaults and by-name access.
///
/// Upgrade effects address stats by name (`mech.walk_speed`), so every stat
/// block needs `get`/`get_mut`. A missing field parses as its default.
macro_rules! stat_struct {
    ($(#[$m:meta])* $name:ident { $($(#[$fm:meta])* $f:ident = $d:expr),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
        #[serde(default)]
        pub struct $name { $($(#[$fm])* pub $f: f64),* }

        impl Default for $name {
            fn default() -> Self { Self { $($f: $d as f64),* } }
        }

        impl $name {
            pub const FIELDS: &'static [&'static str] = &[$(stringify!($f)),*];

            pub fn get(&self, name: &str) -> Option<f64> {
                match name { $(stringify!($f) => Some(self.$f),)* _ => None }
            }

            pub fn get_mut(&mut self, name: &str) -> Option<&mut f64> {
                match name { $(stringify!($f) => Some(&mut self.$f),)* _ => None }
            }
        }
    };
}

/// A string enum that keeps values it does not know.
///
/// The schema types these fields as plain strings, so a new value must parse.
macro_rules! open_enum {
    ($(#[$m:meta])* $name:ident { $($v:ident = $s:literal),* $(,)? }) => {
        $(#[$m])*
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        pub enum $name { $($v,)* Other(String) }

        impl $name {
            pub fn as_str(&self) -> &str {
                match self { $(Self::$v => $s,)* Self::Other(s) => s }
            }

            pub fn parse(s: &str) -> Self {
                match s { $($s => Self::$v,)* _ => Self::Other(s.to_owned()) }
            }
        }

        impl serde::Serialize for $name {
            fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                s.serialize_str(self.as_str())
            }
        }

        impl<'de> serde::Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                Ok(Self::parse(&String::deserialize(d)?))
            }
        }
    };
}

mod action;
mod config;
mod obs;

pub use action::*;
pub use config::*;
pub use obs::*;

use serde::{Deserialize, Serialize};

/// `[x, y]`, or `[yaw, pitch]` for an aim.
pub type Vec2 = [f64; 2];
/// `[x east, y north, z up]` in meters.
pub type Vec3 = [f64; 3];

pub const PROTOCOL: &str = "mechfight/1";

pub const NPC_ID_BASE: u32 = 100;
pub const CRATE_ID_BASE: u32 = 1000;
pub const CUBE_ID_BASE: u32 = 5000;
pub const PICKUP_ID_BASE: u32 = 8000;
pub const DROP_ID_BASE: u32 = 20000;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WeaponKind {
    Gun,
    Mortar,
    Laser,
    Lightning,
}

impl WeaponKind {
    pub const ALL: [WeaponKind; 4] = [Self::Gun, Self::Mortar, Self::Laser, Self::Lightning];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gun => "gun",
            Self::Mortar => "mortar",
            Self::Laser => "laser",
            Self::Lightning => "lightning",
        }
    }
}

/// Every message the server sends a bot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ServerMessage {
    Hello(Box<Hello>),
    Obs(Box<Obs>),
    End(EndMsg),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hello {
    pub protocol: String,
    pub slot: u32,
    pub seed: u64,
    pub map: MapRef,
    pub config: Config,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MapRef {
    pub dir: String,
    pub files: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EndMsg {
    pub place: u32,
    pub kills: u32,
}

/// The bot's answer to `hello`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename = "ready")]
pub struct Ready {
    pub name: String,
}
