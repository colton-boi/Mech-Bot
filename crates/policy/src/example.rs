//! Port of `mechfight-kit/python/example_bot.py`: walk to the zone, grab a
//! weapon, shoot the nearest mech. A baseline and a template, not a contender.

use crate::{Params, Policy};
use proto::{Action, DebugDraw, DebugLine, DebugPoint, Hello, Obs, WeaponKind};
use std::path::Path;
use std::sync::Arc;
use world::World;

pub struct Example {
    pub params: Params,
    /// Attach debug drawings to each action. Off in the simulator, where nothing reads them.
    pub debug: bool,
    reach: f64,
    tick_hz: f64,
    /// The map, for policies that navigate. `None` if it could not be loaded.
    pub world: Option<Arc<World>>,
}

impl Example {
    pub fn new(params: Params) -> Example {
        Example {
            params,
            debug: false,
            reach: 4.0,
            tick_hz: 20.0,
            world: None,
        }
    }
}

fn dist2d(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn dist(a: [f64; 3], b: [f64; 3]) -> f64 {
    dist2d(a, b).hypot(a[2] - b[2])
}

impl Policy for Example {
    fn name(&self) -> &str {
        "example-rs"
    }

    fn on_hello(&mut self, hello: &Hello) {
        self.reach = hello.config.loot.pickup_radius;
        self.tick_hz = hello.config.match_.tick_hz;
        self.world = match World::shared(Path::new(&hello.map.dir)) {
            Ok(world) => Some(world),
            Err(e) => {
                if !hello.map.dir.is_empty() {
                    eprintln!("example: no map, continuing without it: {e}");
                }
                None
            }
        };
    }

    fn act(&mut self, obs: &Obs) -> Action {
        let me = &obs.me;
        let p = &self.params;
        let goal = obs.zone.next_center;
        let mut act = Action::new(obs.tick);
        if obs.offer.is_some() {
            act.pick = Some(0);
        }
        let in_reach = obs
            .pickups
            .iter()
            .find(|pickup| dist2d(pickup.pos, me.pos) < self.reach);
        if let Some(pickup) =
            in_reach.filter(|pickup| me.weapons[1].is_none() && pickup.kind != WeaponKind::Gun)
        {
            act.take = Some(pickup.id);
        }

        let yaw = (goal[1] - me.pos[1]).atan2(goal[0] - me.pos[0]);
        act.move_dir = [yaw.cos(), yaw.sin()];
        act.aim = Some([yaw, 0.0]);
        let to_edge = obs.zone.radius
            - (me.pos[0] - obs.zone.center[0]).hypot(me.pos[1] - obs.zone.center[1]);
        act.boost = !me.in_zone || to_edge < f64::from(p.boost_margin);
        let jump_every = (f64::from(p.jump_period_s) * self.tick_hz).round().max(1.0) as u32;
        act.jump = obs.tick.is_multiple_of(jump_every);

        let nearest = obs
            .mechs
            .iter()
            .filter(|m| m.is_mech())
            .min_by(|a, b| dist(a.pos, me.pos).total_cmp(&dist(b.pos, me.pos)));
        if let Some(target) = nearest {
            let rise =
                target.pos[2] + f64::from(p.aim_height) - (me.pos[2] + me.stats.muzzle_height);
            act.aim = Some([
                (target.pos[1] - me.pos[1]).atan2(target.pos[0] - me.pos[0]),
                rise.atan2(dist2d(target.pos, me.pos)),
            ]);
            act.fire = [true, true];
        }

        if self.debug {
            let mut draw = DebugDraw::default();
            if let Some(target) = nearest {
                draw.lines.push(DebugLine {
                    a: [me.pos[0], me.pos[1], me.pos[2] + me.stats.muzzle_height],
                    b: [
                        target.pos[0],
                        target.pos[1],
                        target.pos[2] + f64::from(p.aim_height),
                    ],
                    color: Some("#ff5040".to_owned()),
                });
            }
            draw.points.push(DebugPoint {
                at: [goal[0], goal[1], me.pos[2]],
                color: Some("#40c0ff".to_owned()),
                size: Some(6.0),
            });
            draw.label = Some(format!(
                "hp {:.0} · {} in sight",
                me.hp,
                obs.mechs.iter().filter(|m| m.is_mech()).count()
            ));
            act.debug = Some(draw);
        }
        act
    }
}
