
use proto::Vec3;
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Tier {
    Easy,
    Mid,
    Expert,
}

impl Tier {
    pub const ALL: [Tier; 3] = [Tier::Easy, Tier::Mid, Tier::Expert];

    pub fn name(self) -> &'static str {
        match self {
            Tier::Easy => "easy",
            Tier::Mid => "mid",
            Tier::Expert => "expert",
        }
    }

    pub fn parse(text: &str) -> Option<Tier> {
        Tier::ALL.into_iter().find(|t| t.name() == text)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    Weakest,
    Nearest,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SquadParams {
    pub a: f64,
    pub t1: f64,
    pub tau: f64,
    pub t_all: f64,
    pub replace_s: f64,
    pub r_opp: f64,
    pub pick: Pick,
    pub aim: f64,
    pub mix: [u32; 3],
}

pub const FITTED: &[(&str, &str)] = &[
    ("v1", "a=1/t1=150/tau=45/tall=330/replace=10/ropp=45/pick=weak/mix=4.7.4"),
    ("v2", "a=1/t1=150/tau=45/tall=330/replace=10/ropp=45/pick=near/mix=4.7.4"),
    ("v3", "a=1/t1=1000/tau=45/tall=240/replace=0/ropp=120/pick=weak/mix=4.7.4"),
    ("v4", "a=1/t1=1000/tau=45/tall=300/replace=0/ropp=120/pick=weak/mix=4.7.4"),
];

impl Default for SquadParams {
    fn default() -> Self {
        SquadParams::parse(FITTED[0].1).expect("the first fitted squad parses")
    }
}

impl SquadParams {
    pub fn parse_onto(base: &SquadParams, spec: &str) -> Result<SquadParams, String> {
        let mut p = base.clone();
        for part in spec.split('/').filter(|s| !s.is_empty()) {
            let (key, value) = part.split_once('=').ok_or_else(|| format!("squad: {part:?} is not key=value"))?;
            let num = || value.parse::<f64>().map_err(|_| format!("squad: {key}={value:?} is not a number"));
            match key {
                "a" => p.a = num()?,
                "t1" => p.t1 = num()?,
                "tau" => p.tau = num()?.max(1.0),
                "tall" => p.t_all = num()?,
                "replace" => p.replace_s = num()?,
                "ropp" => p.r_opp = num()?,
                "aim" => p.aim = num()?.max(0.0),
                "pick" => {
                    p.pick = match value {
                        "weak" => Pick::Weakest,
                        "near" => Pick::Nearest,
                        _ => return Err(format!("squad: pick={value:?}, want weak or near")),
                    }
                }
                "mix" => {
                    let parts: Vec<u32> = value
                        .split('.')
                        .map(str::parse)
                        .collect::<Result<_, _>>()
                        .map_err(|_| format!("squad: mix={value:?}"))?;
                    p.mix = parts.try_into().map_err(|_| format!("squad: mix={value:?} needs easy.mid.expert"))?;
                }
                _ => return Err(format!("squad: unknown key {key:?}")),
            }
        }
        Ok(p)
    }

    pub fn parse(spec: &str) -> Result<SquadParams, String> {
        let base = SquadParams {
            a: 1.0,
            t1: 150.0,
            tau: 45.0,
            t_all: 330.0,
            replace_s: 10.0,
            r_opp: 45.0,
            pick: Pick::Weakest,
            aim: 1.0,
            mix: [4, 7, 4],
        };
        SquadParams::parse_onto(&base, spec)
    }

    pub fn lookup(spec: &str) -> Result<SquadParams, String> {
        if spec.contains('=') {
            return SquadParams::parse_onto(&SquadParams::default(), spec);
        }
        let known = FITTED.iter().map(|(n, _)| *n).collect::<Vec<_>>().join(", ");
        let (_, text) = FITTED
            .iter()
            .find(|(name, _)| *name == spec)
            .ok_or_else(|| format!("squad: no variant {spec:?}; known: {known}"))?;
        SquadParams::parse(text)
    }

    pub fn hunters_wanted(&self, alive: usize, dead: usize, t: f64) -> usize {
        if t >= self.t_all {
            return alive;
        }
        let ramp = ((t - self.t1).max(0.0) / self.tau).floor();
        let n = 1.0 + (self.a * dead as f64).floor() + ramp;
        (n.max(0.0) as usize).min(alive)
    }

    pub fn tier_of(&self, nth: usize) -> Tier {
        let [easy, mid, _] = self.mix.map(|n| n as usize);
        if nth < easy {
            Tier::Easy
        } else if nth < easy + mid {
            Tier::Mid
        } else {
            Tier::Expert
        }
    }
}

impl fmt::Display for SquadParams {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let pick = match self.pick {
            Pick::Weakest => "weak",
            Pick::Nearest => "near",
        };
        let [e, m, x] = self.mix;
        write!(
            f,
            "a={}/t1={}/tau={}/tall={}/replace={}/ropp={}/aim={}/pick={pick}/mix={e}.{m}.{x}",
            self.a, self.t1, self.tau, self.t_all, self.replace_s, self.r_opp, self.aim
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sighting {
    pub id: u32,
    pub pos: Vec3,
    pub vel: Vec3,
    pub t: f64,
    pub seen: bool,
}

#[derive(Clone, Debug)]
pub(crate) struct Member {
    pub id: Option<u32>,
    pub tier: Tier,
    pub pos: Vec3,
    pub vel: Vec3,
    pub last_tick: Option<u32>,
    pub alive: bool,
}

pub(crate) struct Post<'a> {
    pub tick: u32,
    pub pos: Vec3,
    pub vel: Vec3,
    pub sightings: &'a [Sighting],
    pub friendly_hits: u64,
    pub friendly_damaged: u64,
}

#[derive(Debug)]
pub struct Squad {
    pub params: SquadParams,
    pub(crate) decide_every: u32,
    pub(crate) members: Vec<Member>,
    updated_tick: Option<u32>,
    pub(crate) hunters: Vec<usize>,
    last_loss_t: f64,
    sightings: BTreeMap<u32, Sighting>,
    pub(crate) target: Option<Sighting>,
    friendly_hits: u64,
    friendly_damaged: u64,
}

impl Squad {
    pub(crate) fn new(params: SquadParams, tiers: Vec<Tier>) -> Squad {
        let members = tiers
            .into_iter()
            .map(|tier| Member {
                id: None,
                tier,
                pos: [0.0; 3],
                vel: [0.0; 3],
                last_tick: None,
                alive: true,
            })
            .collect();
        Squad {
            params,
            decide_every: 2,
            members,
            updated_tick: None,
            hunters: Vec::new(),
            last_loss_t: f64::NEG_INFINITY,
            sightings: BTreeMap::new(),
            target: None,
            friendly_hits: 0,
            friendly_damaged: 0,
        }
    }

    pub(crate) fn is_member(&self, id: u32) -> bool {
        self.members.iter().any(|m| m.id == Some(id))
    }

    pub fn friendly_hits(&self) -> u64 {
        self.friendly_hits.max(self.friendly_damaged)
    }

    pub(crate) fn update(&mut self, tick: u32, t: f64) {
        if self.updated_tick.is_some_and(|u| u >= tick) {
            return;
        }
        self.updated_tick = Some(tick);
        let cutoff = tick.saturating_sub(self.decide_every);
        for m in &mut self.members {
            if m.alive && m.last_tick.is_some_and(|last| last < cutoff) {
                m.alive = false;
            }
        }
        let before = self.hunters.len();
        let members = &self.members;
        self.hunters.retain(|&i| members[i].alive);
        if self.hunters.len() < before {
            self.last_loss_t = t;
        }
        self.target = self
            .sightings
            .values()
            .max_by(|a, b| (a.t + if a.seen { 0.5 } else { 0.0 }).total_cmp(&(b.t + if b.seen { 0.5 } else { 0.0 })))
            .copied();
        let alive = self.members.iter().filter(|m| m.alive).count();
        let dead = self.members.len() - alive;
        let wanted = self.params.hunters_wanted(alive, dead, t);
        if self.hunters.len() >= wanted || t - self.last_loss_t < self.params.replace_s {
            return;
        }
        let mut candidates: Vec<usize> = (0..self.members.len())
            .filter(|&i| self.members[i].alive && !self.hunters.contains(&i))
            .collect();
        match (self.params.pick, self.target) {
            (Pick::Nearest, Some(target)) => {
                let d = |i: usize| dist2(self.members[i].pos, target.pos);
                candidates.sort_by(|&x, &y| d(x).total_cmp(&d(y)).then(x.cmp(&y)));
            }
            _ => candidates.sort_by_key(|&i| (self.members[i].tier, i)),
        }
        let need = wanted - self.hunters.len();
        self.hunters.extend(candidates.into_iter().take(need));
    }

    pub(crate) fn post(&mut self, member: usize, post: Post) {
        let m = &mut self.members[member];
        m.pos = post.pos;
        m.vel = post.vel;
        m.last_tick = Some(post.tick);
        for s in post.sightings {
            let keep = self.sightings.get(&s.id).is_none_or(|old| s.t > old.t || (s.t == old.t && s.seen && !old.seen));
            if keep {
                self.sightings.insert(s.id, *s);
            }
        }
        self.friendly_hits += post.friendly_hits;
        self.friendly_damaged += post.friendly_damaged;
    }

    pub(crate) fn mates(&self, except: usize) -> Vec<(u32, Vec3, Vec3)> {
        self.members
            .iter()
            .enumerate()
            .filter(|(i, m)| *i != except && m.alive)
            .filter_map(|(_, m)| m.id.map(|id| (id, m.pos, m.vel)))
            .collect()
    }
}

fn dist2(a: Vec3, b: Vec3) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_round_trips_and_rejects_junk() {
        let p = SquadParams::parse("a=2/t1=90/tau=30/tall=300/replace=5/ropp=60/pick=near/mix=3.8.4").unwrap();
        assert_eq!(SquadParams::parse(&p.to_string()).unwrap(), p);
        assert!(SquadParams::parse("a=x").is_err());
        assert!(SquadParams::parse("zz=1").is_err());
        assert!(SquadParams::parse("mix=1.2").is_err());
        assert!(SquadParams::lookup("nope").is_err());
        for (name, _) in FITTED {
            SquadParams::lookup(name).unwrap();
        }
    }

    #[test]
    fn hunter_count_follows_the_formula() {
        let p = SquadParams::parse("a=1.5/t1=100/tau=50/tall=300").unwrap();
        assert_eq!(p.hunters_wanted(15, 0, 0.0), 1);
        assert_eq!(p.hunters_wanted(13, 2, 0.0), 4);
        assert_eq!(p.hunters_wanted(15, 0, 149.0), 1);
        assert_eq!(p.hunters_wanted(15, 0, 150.0), 2);
        assert_eq!(p.hunters_wanted(2, 13, 0.0), 2);
        assert_eq!(p.hunters_wanted(9, 6, 300.0), 9);
    }

    #[test]
    fn weakest_hunt_first_and_dead_hunters_are_replaced_after_the_delay() {
        let p = SquadParams::parse("a=1/t1=1000/tall=1000/replace=10").unwrap();
        let mut squad = Squad::new(p, vec![Tier::Expert, Tier::Easy, Tier::Mid, Tier::Easy]);
        for (i, m) in squad.members.iter_mut().enumerate() {
            m.id = Some(i as u32);
        }
        squad.update(0, 0.0);
        assert_eq!(squad.hunters, [1]);
        let post = |tick| Post {
            tick,
            pos: [0.0; 3],
            vel: [0.0; 3],
            sightings: &[],
            friendly_hits: 0,
            friendly_damaged: 0,
        };
        for i in 0..4 {
            squad.post(i, post(0));
        }
        for i in [0, 2, 3] {
            squad.post(i, post(2));
        }
        squad.update(4, 0.2);
        assert!(!squad.members[1].alive);
        assert!(squad.members.iter().filter(|m| m.alive).count() == 3);
        assert!(squad.hunters.is_empty(), "replacement waits");
        for tick in (4..=400).step_by(2) {
            for i in [0, 2, 3] {
                squad.post(i, post(tick));
            }
            let t = f64::from(tick + 2) * 0.05;
            squad.update(tick + 2, t);
            if t < 10.2 {
                assert!(squad.hunters.is_empty(), "at {t}");
            }
        }
        assert_eq!(squad.hunters, [3, 2]);
    }
}
