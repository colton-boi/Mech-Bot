//! The bot's brain. A [`Policy`] turns the protocol's `Obs` into the protocol's
//! `Action`, so the same code runs in the simulator and in a live match.
//!
//! This crate must never depend on `sim`: the submitted binary is built from
//! `bot`, `policy`, `world`, and `proto` only.

mod cbot;
mod example;
mod house;
pub mod nav;
mod params;
pub mod pyrand;
pub mod squad;

pub use cbot::{Cbot, Strategy};
pub use example::Example;
pub use house::{make_squad, House, SquadHandle};
pub use params::{Params, ParamsF64};
pub use squad::{Squad, SquadParams, Tier};

use proto::{Action, Hello, Obs};

pub trait Policy: Send {
    /// Shown on the spectator stream. At most 32 characters.
    fn name(&self) -> &str {
        "mech-rs"
    }

    /// Called once, before the first observation. Read the rules from
    /// `hello.config` and load the map from `hello.map.dir` here.
    fn on_hello(&mut self, hello: &Hello);

    /// One decision. It has `bot.tick_budget_ms` (50 ms) on one core in a match.
    fn act(&mut self, obs: &Obs) -> Action;
}

/// Sends the default action every decision: stand still, aim held.
pub struct Idle;

impl Policy for Idle {
    fn name(&self) -> &str {
        "idle"
    }

    fn on_hello(&mut self, _hello: &Hello) {}

    fn act(&mut self, obs: &Obs) -> Action {
        Action::new(obs.tick)
    }
}

/// Every policy `make` knows. House bots also take `@VARIANT` (see [`is_house`]).
pub const NAMES: &[&str] = &[
    "aggro",
    "survive",
    "hide",
    "roof",
    "example",
    "idle",
    "house",
    "house-easy",
    "house-mid",
    "house-expert",
];

/// Builds the policy called `name`, or `None` for a name [`is_known`] rejects.
/// A house bot built here is a squad of one; build squads with [`make_squad`].
pub fn make(name: &str, params: &Params) -> Option<Box<dyn Policy>> {
    match name {
        "example" => Some(Box::new(Example::new(params.clone()))),
        "idle" => Some(Box::new(Idle)),
        _ if is_house(name) => make_squad(&[name], params)?.0.pop(),
        _ => Strategy::parse(name).map(|s| Box::new(Cbot::new(s, params.clone())) as Box<dyn Policy>),
    }
}

/// `house`, `house-easy`, `house-mid`, `house-expert`, each optionally with
/// `@v1` (a fitted variant) or `@a=1/t1=150/...` (overrides; see [`SquadParams`]).
/// All house seats of one match must share one squad, built by [`make_squad`].
pub fn is_house(name: &str) -> bool {
    house::parse_name(name).is_some()
}

/// True for every name [`make`] builds.
pub fn is_known(name: &str) -> bool {
    NAMES.contains(&name) || is_house(name)
}
