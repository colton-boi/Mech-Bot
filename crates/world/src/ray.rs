
use crate::{Prism, Vec3, World};

const TERRAIN_MARCH_STEP: f64 = 0.5;
pub const NO_HINT: u32 = u32::MAX;
const TERRAIN_HINT: u32 = u32::MAX - 1;

const MARCH_RUN: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Surface {
    Terrain,
    Prism(usize),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub t: f64,
    pub pos: Vec3,
    pub surface: Surface,
}

fn lerp(a: Vec3, b: Vec3, t: f64) -> Vec3 {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

fn prism_entry(prism: &Prism, a: Vec3, b: Vec3) -> Option<f64> {
    let mut first: Option<f64> = None;
    prism_crossings(prism, a, b, |t| {
        first = Some(first.map_or(t, |f| f.min(t)));
        true
    });
    first
}

fn prism_hit(prism: &Prism, a: Vec3, b: Vec3) -> bool {
    let mut hit = false;
    prism_crossings(prism, a, b, |_| {
        hit = true;
        false
    });
    hit
}

fn prism_crossings(prism: &Prism, a: Vec3, b: Vec3, mut found: impl FnMut(f64) -> bool) {
    let [min_x, min_y, max_x, max_y] = prism.bbox;
    if a[0].max(b[0]) < min_x
        || a[0].min(b[0]) > max_x
        || a[1].max(b[1]) < min_y
        || a[1].min(b[1]) > max_y
        || a[2].max(b[2]) < prism.base
        || a[2].min(b[2]) > prism.top
    {
        return;
    }
    let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let (mut t0, mut t1) = (0.0f64, 1.0f64);
    for (p, dp, lo, hi) in [(a[0], d[0], min_x, max_x), (a[1], d[1], min_y, max_y), (a[2], d[2], prism.base, prism.top)] {
        if dp == 0.0 {
            continue;
        }
        let (ta, tb) = ((lo - p) / dp, (hi - p) / dp);
        t0 = t0.max(ta.min(tb));
        t1 = t1.min(ta.max(tb));
    }
    if t0 > t1 + 1e-9 {
        return;
    }
    if prism.contains(a) {
        found(0.0);
        return;
    }
    let (s0, s1) = ((t0 - 1e-6).max(0.0), (t1 + 1e-6).min(1.0));
    let near_a = [a[0] + d[0] * s0, a[1] + d[1] * s0];
    let near_b = [a[0] + d[0] * s1, a[1] + d[1] * s1];
    for (p, q) in prism.edges_near(near_a, near_b) {
        let e = [q[0] - p[0], q[1] - p[1]];
        let denom = d[0] * e[1] - d[1] * e[0];
        if denom.abs() < 1e-12 {
            continue;
        }
        let ap = [p[0] - a[0], p[1] - a[1]];
        let t = (ap[0] * e[1] - ap[1] * e[0]) / denom;
        let u = (ap[0] * d[1] - ap[1] * d[0]) / denom;
        if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
            let z = a[2] + d[2] * t;
            if z >= prism.base && z <= prism.top && !found(t) {
                return;
            }
        }
    }
    if d[2] != 0.0 {
        for cap in [prism.top, prism.base] {
            let t = (cap - a[2]) / d[2];
            if (0.0..=1.0).contains(&t) && prism.contains_xy(a[0] + d[0] * t, a[1] + d[1] * t) && !found(t) {
                return;
            }
        }
    }
}

impl World {
    fn terrain_entry(&self, a: Vec3, b: Vec3) -> Option<f64> {
        if a[2].min(b[2]) > self.terrain.max_height {
            return None;
        }
        let below = |t: f64| {
            let p = lerp(a, b, t);
            p[2] < self.terrain.height(p[0], p[1])
        };
        if below(0.0) {
            return Some(0.0);
        }
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
        let steps = (len / TERRAIN_MARCH_STEP).ceil().max(1.0) as usize;
        let mut above = 0.0;
        let mut k = 1;
        while k <= steps {
            let run_end = (k + MARCH_RUN - 1).min(steps);
            let (p, q) = (lerp(a, b, k as f64 / steps as f64), lerp(a, b, run_end as f64 / steps as f64));
            if p[2].min(q[2]) > self.terrain.max_in(p[0], p[1], q[0], q[1]) + 1e-6 {
                above = run_end as f64 / steps as f64;
                k = run_end + 1;
                continue;
            }
            for k in k..=run_end {
                let t = k as f64 / steps as f64;
                if below(t) {
                    let mut under = t;
                    for _ in 0..20 {
                        let mid = 0.5 * (above + under);
                        if below(mid) {
                            under = mid;
                        } else {
                            above = mid;
                        }
                    }
                    return Some(under);
                }
                above = t;
            }
            k = run_end + 1;
        }
        None
    }

    pub fn raycast(&self, a: Vec3, b: Vec3) -> Option<RayHit> {
        let Some(grid) = self.grid() else {
            return self.raycast_reference(a, b);
        };
        let mut near = Vec::new();
        grid.prisms_on(a, b, &mut near);
        let mut best = self.terrain_entry(a, b).map(|t| (t, Surface::Terrain));
        for i in near.into_iter().map(|i| i as usize) {
            if let Some(t) = prism_entry(&self.prisms[i], a, b) {
                if best.is_none_or(|(bt, _)| t < bt) {
                    best = Some((t, Surface::Prism(i)));
                }
            }
        }
        best.map(|(t, surface)| RayHit {
            t,
            pos: lerp(a, b, t),
            surface,
        })
    }

    #[doc(hidden)]
    pub fn raycast_reference(&self, a: Vec3, b: Vec3) -> Option<RayHit> {
        let mut best = self.terrain_entry(a, b).map(|t| (t, Surface::Terrain));
        for (i, prism) in self.prisms.iter().enumerate() {
            if let Some(t) = prism_entry(prism, a, b) {
                if best.is_none_or(|(bt, _)| t < bt) {
                    best = Some((t, Surface::Prism(i)));
                }
            }
        }
        best.map(|(t, surface)| RayHit {
            t,
            pos: lerp(a, b, t),
            surface,
        })
    }

    pub fn los(&self, a: Vec3, b: Vec3) -> bool {
        self.los_hinted(a, b, &mut NO_HINT.clone())
    }

    pub fn los_hinted(&self, a: Vec3, b: Vec3, hint: &mut u32) -> bool {
        let blocked = match *hint {
            TERRAIN_HINT => self.terrain_entry(a, b).is_some(),
            i => self.prisms.get(i as usize).is_some_and(|p| prism_hit(p, a, b)),
        };
        if blocked {
            return false;
        }
        let prism = match self.grid() {
            Some(grid) => grid.find_prism_on(a, b, |i| prism_hit(&self.prisms[i], a, b)),
            None => match self.raycast_reference(a, b) {
                Some(RayHit {
                    surface: Surface::Prism(i), ..
                }) => Some(i),
                Some(_) => {
                    *hint = TERRAIN_HINT;
                    return false;
                }
                None => None,
            },
        };
        *hint = match prism {
            Some(i) => i as u32,
            None if self.terrain_entry(a, b).is_some() => TERRAIN_HINT,
            None => NO_HINT,
        };
        *hint == NO_HINT
    }
}
