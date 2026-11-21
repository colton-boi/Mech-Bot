//! The practice house bots: 15 cooperating mechs that hunt the one mech that
//! is not theirs (STRATEGY.md §5a, analysis/REPLAYS.md §5).
//!
//! Each house bot is a [`House`] policy holding one shared [`Squad`]. It sees
//! only its own `Obs`; the squad pools sightings and decides who hunts.
//!
//! Implemented: loot the nearest pickup early, roam inside the next circle,
//! stay in the zone, hunters close to a per-tier range band and strafe,
//! non-hunters shoot the target inside `r_opp`, hit back at an attacker,
//! per-tier aim noise, lead and reaction time, hop and stuck jumps, upgrades
//! (plating first).
//!
//! Friendly fire is the policy's job, not the simulator's: a bot fires only
//! when no squadmate is near any shot it could make (see [`House::clear`]).
//!
//! Not implemented: retreat at low HP (no tier in §5a retreats except hard),
//! crate and NPC farming, drops, weapon swaps, sound-based search.

use crate::squad::{Post, Sighting, Squad, SquadParams, Tier};
use crate::{Params, Policy};
use proto::{Action, GameEvent, Hello, Obs, Vec3, WeaponKind, WeaponObs, NPC_ID_BASE};
use std::sync::{Arc, Mutex, MutexGuard};

/// Skill numbers of one tier. STRATEGY.md §5a; all `GUESS:` except the expert reaction.
struct Skill {
    aim_noise: f64,
    lead: f64,
    reaction_s: f64,
    strafe: bool,
    /// Range band a hunter holds, meters (REPLAYS.md: median hit range 55-77 m).
    band: (f64, f64),
    /// Longest gun shot taken, meters (rivals' replay reading: medium lands nothing past 120 m).
    fire_range: f64,
}

fn skill(tier: Tier) -> Skill {
    match tier {
        Tier::Easy => Skill {
            aim_noise: 0.06,
            lead: 0.3,
            reaction_s: 0.6,
            strafe: false,
            band: (50.0, 95.0),
            fire_range: 100.0,
        },
        Tier::Mid => Skill {
            aim_noise: 0.035,
            lead: 0.7,
            reaction_s: 0.3,
            strafe: true,
            band: (50.0, 85.0),
            fire_range: 120.0,
        },
        Tier::Expert => Skill {
            aim_noise: 0.012,
            lead: 1.0,
            reaction_s: 0.2,
            strafe: true,
            band: (45.0, 75.0),
            fire_range: 200.0,
        },
    }
}

/// GUESS: aim noise is redrawn this often, so a miss persists (rivals model 0.3-0.5 s).
const NOISE_EVERY_S: f64 = 0.4;
/// GUESS: a sighting older than this is not chased.
const TARGET_MEMORY_S: f64 = 45.0;
/// GUESS: a damage bearing places the attacker this far out.
const BEARING_GUESS_M: f64 = 70.0;
/// Seconds a house bot keeps looking for a first weapon (REPLAYS.md: median first take 5.3 s).
const LOOT_UNTIL_S: f64 = 20.0;
const LOOT_RADIUS_M: f64 = 110.0;
/// Clearance kept between any possible shot and a squadmate's body, beyond its radius, meters.
const FF_CLEARANCE_M: f64 = 2.5;
/// A squadmate's radius is assumed at least this many times the base radius (cubes grow mechs).
const FF_SIZE_MULT: f64 = 1.6;

/// One squad member.
pub struct House {
    member: usize,
    tier: Tier,
    squad: Arc<Mutex<Squad>>,
    #[allow(dead_code)]
    params: Params,
    name: String,
    rng: u64,
    id: u32,
    pickup_radius: f64,
    radius: f64,
    height: f64,
    // Memory.
    waypoint: Option<[f64; 2]>,
    waypoint_until: f64,
    strafe_sign: f64,
    strafe_until: f64,
    engaged_id: Option<u32>,
    engaged_since: f64,
    attacker: Option<(u32, f64)>,
    slow_decisions: u32,
    laser_since: Option<f64>,
    noise: [f64; 2],
    noise_until: f64,
}

/// The shared record of one squad, for reading its counters after a match.
pub type SquadHandle = Arc<Mutex<Squad>>;

/// Builds `names.len()` house bots sharing one squad. Every name must be one
/// [`parse_name`] accepts, and every `@` variant in them must agree.
pub fn make_squad(names: &[&str], params: &Params) -> Option<(Vec<Box<dyn Policy>>, SquadHandle)> {
    let parsed: Vec<(Option<Tier>, Option<&str>)> = names.iter().map(|n| parse_name(n)).collect::<Option<_>>()?;
    let mut variant = None;
    for (_, v) in &parsed {
        if let Some(v) = v {
            if variant.is_some_and(|old| old != *v) {
                return None;
            }
            variant = Some(*v);
        }
    }
    let squad_params = match variant {
        Some(spec) => SquadParams::lookup(spec).ok()?,
        None => SquadParams::default(),
    };
    let mut plain = 0;
    let tiers: Vec<Tier> = parsed
        .iter()
        .map(|(tier, _)| {
            tier.unwrap_or_else(|| {
                plain += 1;
                squad_params.tier_of(plain - 1)
            })
        })
        .collect();
    let squad = Arc::new(Mutex::new(Squad::new(squad_params, tiers.clone())));
    let policies = tiers
        .into_iter()
        .enumerate()
        .map(|(member, tier)| Box::new(House::new(member, tier, squad.clone(), params.clone())) as Box<dyn Policy>)
        .collect();
    Some((policies, squad))
}

/// `house[-TIER][@VARIANT]` → (explicit tier, variant spec). `None` for any other name.
pub fn parse_name(name: &str) -> Option<(Option<Tier>, Option<&str>)> {
    let (base, variant) = match name.split_once('@') {
        Some((b, v)) => (b, Some(v)),
        None => (name, None),
    };
    let tier = match base.strip_prefix("house")? {
        "" => None,
        rest => Some(Tier::parse(rest.strip_prefix('-')?)?),
    };
    if let Some(v) = variant {
        SquadParams::lookup(v).ok()?;
    }
    Some((tier, variant))
}

impl House {
    fn new(member: usize, tier: Tier, squad: Arc<Mutex<Squad>>, params: Params) -> House {
        House {
            member,
            tier,
            squad,
            params,
            name: format!("house-{}", tier.name()),
            rng: 0x9e37_79b9_7f4a_7c15 ^ member as u64,
            id: 0,
            pickup_radius: 4.0,
            radius: 2.5,
            height: 9.0,
            waypoint: None,
            waypoint_until: 0.0,
            strafe_sign: 1.0,
            strafe_until: 0.0,
            engaged_id: None,
            engaged_since: 0.0,
            attacker: None,
            slow_decisions: 0,
            laser_since: None,
            noise: [0.0; 2],
            noise_until: 0.0,
        }
    }

    fn lock(squad: &Mutex<Squad>) -> MutexGuard<'_, Squad> {
        squad.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// splitmix64: the policy's own deterministic stream, seeded from the match seed and slot.
    fn next_u64(&mut self) -> u64 {
        self.rng = self.rng.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn gauss(&mut self) -> f64 {
        let u = self.uniform().max(1e-12);
        let v = self.uniform();
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }

    /// True when no squadmate is near the path of a shot leaving `origin` along any
    /// aim between `from` and `to` (the torso slews between them during the decision),
    /// out to `range`, with `spread` radians of scatter and `speed` m/s of flight.
    /// Misses fly on, so the whole range counts, walls ignored.
    fn clear(&self, shot: Shot, mates: &[(u32, Vec3, Vec3)]) -> bool {
        let Shot {
            origin,
            from,
            to,
            range,
            spread,
            speed,
        } = shot;
        let r = self.radius * FF_SIZE_MULT + FF_CLEARANCE_M;
        let dyaw = wrap_pi(to[0] - from[0]);
        (0..=4).all(|k| {
            let f = f64::from(k) / 4.0;
            let dir = aim_dir([from[0] + dyaw * f, from[1] + (to[1] - from[1]) * f]);
            mates.iter().all(|&(_, pos, vel)| {
                let along = dot(sub(pos, origin), dir).max(0.0);
                let flight = if speed > 0.0 { along / speed + 0.25 } else { 0.25 };
                // Where the mate is now and where it will be when the shot gets there.
                [pos, add(pos, scale(vel, flight))].iter().all(|&p| {
                    let lo = p;
                    let hi = [p[0], p[1], p[2] + self.height];
                    let (d, t) = seg_seg(origin, add(origin, scale(dir, range)), lo, hi);
                    d > r + t * range * spread
                })
            })
        })
    }

    fn mates_near(point: Vec3, within: f64, mates: &[(u32, Vec3, Vec3)]) -> bool {
        mates
            .iter()
            .any(|&(_, p, v)| dist3(add(p, scale(v, 0.5)), point) < within || dist3(p, point) < within)
    }
}

/// A shot a bot could make this decision, for the friendly-fire check.
#[derive(Clone, Copy)]
struct Shot {
    origin: Vec3,
    /// Torso aim now and the aim commanded; the shot leaves somewhere between.
    from: [f64; 2],
    to: [f64; 2],
    range: f64,
    /// Scatter, radians.
    spread: f64,
    /// Flight speed, m/s; 0 for a beam.
    speed: f64,
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn scale(a: Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn dist2(a: Vec3, b: Vec3) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}
fn dist3(a: Vec3, b: Vec3) -> f64 {
    dist2(a, b).hypot(a[2] - b[2])
}
fn wrap_pi(a: f64) -> f64 {
    let t = std::f64::consts::TAU;
    a - t * ((a + std::f64::consts::PI) / t).floor()
}
fn aim_dir(aim: [f64; 2]) -> Vec3 {
    let (yaw, pitch) = (aim[0], aim[1]);
    [yaw.cos() * pitch.cos(), yaw.sin() * pitch.cos(), pitch.sin()]
}
fn yaw_to(from: Vec3, to: [f64; 2]) -> f64 {
    (to[1] - from[1]).atan2(to[0] - from[0])
}

/// Closest distance between segments `p0-p1` and `q0-q1`, and the fraction along `p` where it occurs.
fn seg_seg(p0: Vec3, p1: Vec3, q0: Vec3, q1: Vec3) -> (f64, f64) {
    let d1 = sub(p1, p0);
    let d2 = sub(q1, q0);
    let r = sub(p0, q0);
    let (a, e, f) = (dot(d1, d1), dot(d2, d2), dot(d2, r));
    let (mut s, mut t);
    if a <= 1e-12 && e <= 1e-12 {
        return (dist3(p0, q0), 0.0);
    }
    if a <= 1e-12 {
        s = 0.0;
        t = (f / e).clamp(0.0, 1.0);
    } else {
        let c = dot(d1, r);
        if e <= 1e-12 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            s = if denom > 1e-12 { ((b * f - c * e) / denom).clamp(0.0, 1.0) } else { 0.0 };
            t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
        }
    }
    let cp = add(p0, scale(d1, s));
    let cq = add(q0, scale(d2, t));
    (dist3(cp, cq), s)
}

impl Policy for House {
    fn name(&self) -> &str {
        &self.name
    }

    fn on_hello(&mut self, hello: &Hello) {
        self.id = hello.slot;
        self.rng ^= hello.seed.wrapping_mul(0xa076_1d64_78bd_642f) ^ u64::from(hello.slot).wrapping_mul(0xe703_7ed1_a0b4_28db);
        self.pickup_radius = hello.config.loot.pickup_radius;
        self.radius = hello.config.mech.radius;
        self.height = hello.config.mech.height;
        let mut squad = House::lock(&self.squad);
        squad.decide_every = hello.config.match_.decide_every as u32;
        squad.members[self.member].id = Some(hello.slot);
    }

    fn act(&mut self, obs: &Obs) -> Action {
        let me = &obs.me;
        let t = obs.t;
        let mut act = Action::new(obs.tick);
        let sk = skill(self.tier);

        // 1. Talk to the squad: update once per tick, post what I see, read the plan.
        let shared = self.squad.clone();
        let mut squad = House::lock(&shared);
        squad.update(obs.tick, t);
        let is_enemy = |id: u32| id < NPC_ID_BASE && id != me.id && !squad.is_member(id);
        let mut sightings: Vec<Sighting> = obs
            .mechs
            .iter()
            .filter(|m| m.is_mech() && is_enemy(m.id))
            .map(|m| Sighting {
                id: m.id,
                pos: m.pos,
                vel: m.vel,
                t,
                seen: true,
            })
            .collect();
        let (mut friendly_hits, mut friendly_damaged) = (0, 0);
        for e in &obs.events {
            match e {
                GameEvent::Hit { target, .. } if squad.is_member(*target) => friendly_hits += 1,
                GameEvent::Damaged { by: Some(by), .. } if squad.is_member(*by) => friendly_damaged += 1,
                GameEvent::Damaged {
                    by: Some(by),
                    bearing: Some(bearing),
                    ..
                } if is_enemy(*by) => {
                    self.attacker = Some((*by, t));
                    if !sightings.iter().any(|s| s.id == *by) {
                        sightings.push(Sighting {
                            id: *by,
                            pos: [
                                me.pos[0] + BEARING_GUESS_M * bearing.cos(),
                                me.pos[1] + BEARING_GUESS_M * bearing.sin(),
                                me.pos[2],
                            ],
                            vel: [0.0; 3],
                            t,
                            seen: false,
                        });
                    }
                }
                _ => {}
            }
        }
        squad.post(
            self.member,
            Post {
                tick: obs.tick,
                pos: me.pos,
                vel: me.vel,
                sightings: &sightings,
                friendly_hits,
                friendly_damaged,
            },
        );
        let hunter = squad.hunters.contains(&self.member);
        let target = squad.target.filter(|s| t - s.t < TARGET_MEMORY_S);
        let r_opp = squad.params.r_opp;
        let aim_mult = squad.params.aim;
        let mut mates = squad.mates(self.member);
        let members: Vec<u32> = squad.members.iter().filter_map(|m| m.id).collect();
        drop(squad);
        // Visible mates are fresher than their posts: check both.
        mates.extend(
            obs.mechs
                .iter()
                .filter(|m| m.is_mech() && members.contains(&m.id) && m.id != me.id)
                .map(|m| (m.id, m.pos, m.vel)),
        );

        // Upgrades: plating first (REPLAYS.md), else the first choice.
        if let Some(offer) = &obs.offer {
            act.pick = Some(offer.choices.iter().position(|c| c.id == "plating").unwrap_or(0) as u8);
        }

        // 2. Whom to shoot: a visible enemy mech, if I hunt, if it is close, or if it hit me.
        let visible = obs
            .mechs
            .iter()
            .filter(|m| m.is_mech() && !members.contains(&m.id) && m.id != me.id)
            .min_by(|a, b| dist3(a.pos, me.pos).total_cmp(&dist3(b.pos, me.pos)));
        let hit_me_recently = |id: u32| self.attacker.is_some_and(|(a, at)| a == id && t - at < 5.0);
        let engage = visible.filter(|m| hunter || dist3(m.pos, me.pos) < r_opp || hit_me_recently(m.id));
        match engage {
            Some(m) if self.engaged_id != Some(m.id) => {
                self.engaged_id = Some(m.id);
                self.engaged_since = t;
            }
            None => self.engaged_id = None,
            _ => {}
        }

        // 3. Legs.
        let zone_d = (me.pos[0] - obs.zone.center[0]).hypot(me.pos[1] - obs.zone.center[1]);
        let to_center = yaw_to(me.pos, obs.zone.center);
        let mut move_yaw: Option<f64> = None;
        let mut boost = false;
        if !me.in_zone || zone_d > obs.zone.radius - 12.0 {
            move_yaw = Some(to_center);
            boost = !me.in_zone;
        } else if let Some(goal) = target.filter(|_| hunter) {
            let d = dist2(goal.pos, me.pos);
            let yaw = yaw_to(me.pos, [goal.pos[0], goal.pos[1]]);
            let (lo, hi) = sk.band;
            if d > hi || engage.is_none() {
                move_yaw = Some(yaw);
                boost = d > hi + 80.0 && me.energy > 0.5 * me.max_energy;
            } else if d < lo {
                move_yaw = Some(yaw + std::f64::consts::PI);
            } else if sk.strafe {
                if t >= self.strafe_until {
                    self.strafe_sign = if self.uniform() < 0.5 { -1.0 } else { 1.0 };
                    self.strafe_until = t + 1.0 + 1.4 * self.uniform();
                }
                move_yaw = Some(yaw + self.strafe_sign * std::f64::consts::FRAC_PI_2);
            }
        } else if let Some(p) = obs
            .pickups
            .iter()
            .filter(|_| t < LOOT_UNTIL_S && me.weapons[1].is_none())
            .filter(|p| p.kind != WeaponKind::Gun && dist2(p.pos, me.pos) < LOOT_RADIUS_M)
            .min_by(|a, b| dist2(a.pos, me.pos).total_cmp(&dist2(b.pos, me.pos)))
        {
            if dist2(p.pos, me.pos) < self.pickup_radius {
                act.take = Some(p.id);
            } else {
                move_yaw = Some(yaw_to(me.pos, [p.pos[0], p.pos[1]]));
            }
        } else if engage.is_none() {
            // Roam: wander between points inside the next circle (REPLAYS.md: weak zone bias).
            let reached = self.waypoint.is_some_and(|w| (w[0] - me.pos[0]).hypot(w[1] - me.pos[1]) < 8.0);
            if self.waypoint.is_none() || reached || t > self.waypoint_until {
                let ang = std::f64::consts::TAU * self.uniform();
                let rad = obs.zone.next_radius * 0.7 * self.uniform().sqrt();
                self.waypoint = Some([obs.zone.next_center[0] + rad * ang.cos(), obs.zone.next_center[1] + rad * ang.sin()]);
                self.waypoint_until = t + 20.0 + 20.0 * self.uniform();
            }
            if let Some(w) = self.waypoint {
                move_yaw = Some(yaw_to(me.pos, w));
            }
        }
        if let Some(yaw) = move_yaw {
            act.move_dir = [yaw.cos(), yaw.sin()];
            act.boost = boost;
            let speed = me.vel[0].hypot(me.vel[1]);
            self.slow_decisions = if speed < 1.5 && me.grounded && t > 1.0 { self.slow_decisions + 1 } else { 0 };
        }
        // Hop now and then (REPLAYS.md: 22-33% of frames airborne), and when stuck.
        act.jump = me.grounded && (self.slow_decisions >= 4 || self.uniform() < 0.04);
        if self.slow_decisions >= 4 {
            self.slow_decisions = 0;
        }

        // 4. Torso and triggers.
        let muzzle = [me.pos[0], me.pos[1], me.pos[2] + me.stats.muzzle_height];
        let mut want = move_yaw.map(|y| [y, 0.0]).unwrap_or(me.aim);
        if let Some(m) = engage {
            let d = dist3(m.pos, me.pos);
            let speed = me.weapons.iter().flatten().find_map(|w| match w {
                WeaponObs::Gun { stats, .. } => Some(stats.speed),
                _ => None,
            });
            let tof = speed.map_or(0.0, |s| d / s);
            let aim_at = add(add(m.pos, scale(m.vel, tof * sk.lead)), [0.0, 0.0, 4.5]);
            let horiz = dist2(aim_at, muzzle);
            if t >= self.noise_until {
                let sigma = sk.aim_noise * aim_mult;
                self.noise = [sigma * self.gauss(), sigma * 0.5 * self.gauss()];
                self.noise_until = t + NOISE_EVERY_S;
            }
            want = [
                yaw_to(muzzle, [aim_at[0], aim_at[1]]) + self.noise[0],
                (aim_at[2] - muzzle[2]).atan2(horiz) + self.noise[1],
            ];
            let reacted = t - self.engaged_since >= sk.reaction_s;
            let on_target = wrap_pi(want[0] - me.aim[0]).abs() < 0.12 && (want[1] - me.aim[1]).abs() < 0.12;
            for (slot, weapon) in me.weapons.iter().enumerate() {
                let Some(weapon) = weapon else { continue };
                act.fire[slot] = match weapon {
                    WeaponObs::Gun { stats, ammo, reload, .. } => {
                        if *ammo <= 0.0 && *reload <= 0.0 {
                            act.reload = true;
                        }
                        reacted
                            && on_target
                            && d < stats.range.min(sk.fire_range)
                            && self.clear(
                                Shot {
                                    origin: muzzle,
                                    from: me.aim,
                                    to: want,
                                    range: stats.range,
                                    spread: 3.0 * stats.spread_deg.to_radians(),
                                    speed: stats.speed,
                                },
                                &mates,
                            )
                    }
                    WeaponObs::Lightning { stats, .. } => {
                        // The bolt takes the nearest thing in its cone and chains: no mate anywhere near.
                        let reach = stats.range + stats.chain * stats.chain_range + 10.0;
                        reacted && d < stats.range && !House::mates_near(me.pos, reach, &mates)
                    }
                    WeaponObs::Mortar { stats, .. } => {
                        reacted
                            && on_target
                            && (40.0..140.0).contains(&d)
                            && !House::mates_near(aim_at, stats.splash_radius * 2.0 + 8.0, &mates)
                            && self.clear(
                                Shot {
                                    origin: muzzle,
                                    from: me.aim,
                                    to: want,
                                    range: d + 20.0,
                                    spread: 0.08,
                                    speed: stats.speed,
                                },
                                &mates,
                            )
                    }
                    WeaponObs::Laser { stats, charge, .. } => {
                        // Charging is safe; the release is the shot. Release only on a clear line.
                        let clear = self.clear(
                            Shot {
                                origin: muzzle,
                                from: me.aim,
                                to: want,
                                range: stats.range,
                                spread: 0.01,
                                speed: 0.0,
                            },
                            &mates,
                        );
                        if *charge > 0.0 {
                            let since = *self.laser_since.get_or_insert(t);
                            let full = *charge >= stats.charge;
                            if full && clear && on_target {
                                false
                            } else if t - since > stats.hold_s - 0.4 && !clear {
                                // GUESS: the laser fires itself after hold_s. Point it at the sky first.
                                want = [me.aim[0], me.stats.max_pitch];
                                true
                            } else {
                                true
                            }
                        } else {
                            self.laser_since = None;
                            reacted && on_target && clear && d < stats.range
                        }
                    }
                };
            }
        } else if let Some(WeaponObs::Laser { charge, .. }) = &me.weapons[1] {
            // Lost the target mid-charge: aim at the sky and let go.
            if *charge > 0.0 {
                want = [me.aim[0], me.stats.max_pitch];
                act.fire[1] = (me.aim[1] - me.stats.max_pitch).abs() > 0.05;
            }
        }
        act.aim = Some(want);
        act
    }
}
