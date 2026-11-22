//! The bot's brain. A [`Policy`] turns the protocol's `Obs` into the protocol's
//! `Action`, so the same code runs in the simulator and in a live match.
//!
//! This crate must never depend on `sim`: the submitted binary is built from
//! `bot`, `policy`, `world`, and `proto` only.

pub mod apex;
mod cbot;
mod example;
pub mod nav;
mod params;
pub mod pyrand;

pub use apex::{Apex, ApexGenome};
pub use cbot::{Cbot, Strategy};
pub use example::Example;
pub use params::{Params, ParamsF64};

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

/// Every built-in policy `make` knows.
pub const NAMES: &[&str] = &[
    "apex", "aggro", "survive", "hide", "roof", "example", "idle",
];

/// Builds the policy called `name`, or `None` for a name [`is_known`] rejects.
/// Any policy also takes `@key=value/...` [`Params`] overrides on top of
/// `params` (`aggro@recover_on=1/lead_lag_s=0.075`), so one lineup can seat
/// variants side by side.
pub fn make(name: &str, params: &Params) -> Option<Box<dyn Policy>> {
    if name == "apex" {
        return Some(make_apex(ApexGenome::default()));
    }
    if let Some(spec) = name.strip_prefix("apex@") {
        return Some(make_apex(ApexGenome::default().with_overrides(spec).ok()?));
    }
    if let Some((base, spec)) = name.split_once('@') {
        return make(base, &params.with_overrides(spec).ok()?);
    }
    match name {
        "example" => Some(Box::new(Example::new(params.clone()))),
        "idle" => Some(Box::new(Idle)),
        _ => {
            Strategy::parse(name).map(|s| Box::new(Cbot::new(s, params.clone())) as Box<dyn Policy>)
        }
    }
}

/// True for every name [`make`] builds.
pub fn is_known(name: &str) -> bool {
    if name == "apex" {
        return true;
    }
    if let Some(spec) = name.strip_prefix("apex@") {
        return ApexGenome::default().with_overrides(spec).is_ok();
    }
    match name.split_once('@') {
        Some((base, spec)) => {
            NAMES.contains(&base) && Params::default().with_overrides(spec).is_ok()
        }
        None => NAMES.contains(&name),
    }
}

/// The apex policy playing `genome`. `make("apex@...")` builds the same thing
/// from the default genome; this is for a genome loaded from a file.
pub fn make_apex(genome: ApexGenome) -> Box<dyn Policy> {
    Box::new(Apex::new(genome))
}
