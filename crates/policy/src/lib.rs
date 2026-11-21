
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
    fn name(&self) -> &str {
        "mech-rs"
    }

    fn on_hello(&mut self, hello: &Hello);

    fn act(&mut self, obs: &Obs) -> Action;
}

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

pub fn make(name: &str, params: &Params) -> Option<Box<dyn Policy>> {
    match name {
        "example" => Some(Box::new(Example::new(params.clone()))),
        "idle" => Some(Box::new(Idle)),
        _ if is_house(name) => make_squad(&[name], params)?.0.pop(),
        _ => Strategy::parse(name).map(|s| Box::new(Cbot::new(s, params.clone())) as Box<dyn Policy>),
    }
}

pub fn is_house(name: &str) -> bool {
    house::parse_name(name).is_some()
}

pub fn is_known(name: &str) -> bool {
    NAMES.contains(&name) || is_house(name)
}
