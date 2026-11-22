//! The bot's answer to one observation.

use crate::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Action {
    /// Echo of the observation tick. Any other value is dropped by the server.
    pub tick: u32,
    /// Desired horizontal direction in the world frame, length at most 1.
    #[serde(rename = "move")]
    pub move_dir: Vec2,
    pub boost: bool,
    pub jump: bool,
    /// Desired `[yaw, pitch]`. `None` keeps the previous commanded aim.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aim: Option<Vec2>,
    /// Trigger of slot 0 and slot 1, held until the next decision.
    pub fire: [bool; 2],
    pub reload: bool,
    /// Index into the current offer's choices.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pick: Option<u8>,
    /// Id of a weapon pickup in reach.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub take: Option<u32>,
    /// Slot to empty. With `take`, the slot the new weapon replaces.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drop: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug: Option<DebugDraw>,
}

impl Action {
    pub fn new(tick: u32) -> Self {
        Self { tick, ..Self::default() }
    }

    /// Replaces non-finite numbers, which JSON cannot carry, so the line stays schema-valid.
    pub fn sanitize(&mut self) {
        if !self.move_dir.iter().all(|v| v.is_finite()) {
            self.move_dir = [0.0, 0.0];
        }
        if self.aim.is_some_and(|a| !a.iter().all(|v| v.is_finite())) {
            self.aim = None;
        }
    }

    /// The action a mech coasts on when no answer arrives: continuous parts kept, one-shots dropped.
    pub fn coast(&self, tick: u32) -> Self {
        Self {
            tick,
            move_dir: self.move_dir,
            boost: self.boost,
            aim: self.aim,
            ..Self::default()
        }
    }
}

/// Debug drawing for the team's own tools. It never affects the match.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DebugDraw {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lines: Vec<DebugLine>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub points: Vec<DebugPoint>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub text: Vec<DebugText>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugLine {
    pub a: Vec3,
    pub b: Vec3,
    /// `#rgb` or `#rrggbb`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugPoint {
    pub at: Vec3,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
    /// Radius in screen pixels, 1 to 12.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub size: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebugText {
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub at: Option<Vec3>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<String>,
}
