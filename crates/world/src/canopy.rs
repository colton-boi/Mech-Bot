
use crate::{Vec3, World};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanopyRules {
    pub density: f64,
    pub block_t: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tree {
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub height: f64,
    pub crown_r: f64,
    pub crown_base: f64,
    pub trunk_r: f64,
    pub crown_x: f64,
    pub crown_y: f64,
}

fn dist2_to_segment(a: Vec3, b: Vec3, c: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let len2 = d[0] * d[0] + d[1] * d[1];
    let t = if len2 > 0.0 {
        (((c[0] - a[0]) * d[0] + (c[1] - a[1]) * d[1]) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (a[0] + d[0] * t - c[0]).powi(2) + (a[1] + d[1] * t - c[1]).powi(2)
}

impl Tree {
    fn trunk_blocks(&self, a: Vec3, b: Vec3) -> bool {
        let d = [b[0] - a[0], b[1] - a[1]];
        let f = [a[0] - self.x, a[1] - self.y];
        let qa = d[0] * d[0] + d[1] * d[1];
        let qb = 2.0 * (f[0] * d[0] + f[1] * d[1]);
        let qc = f[0] * f[0] + f[1] * f[1] - self.trunk_r * self.trunk_r;
        let (t0, t1) = if qa < 1e-12 {
            if qc > 0.0 {
                return false;
            }
            (0.0, 1.0)
        } else {
            let disc = qb * qb - 4.0 * qa * qc;
            if disc < 0.0 {
                return false;
            }
            let root = disc.sqrt();
            (((-qb - root) / (2.0 * qa)).max(0.0), ((-qb + root) / (2.0 * qa)).min(1.0))
        };
        if t0 > t1 {
            return false;
        }
        let (z0, z1) = (a[2] + (b[2] - a[2]) * t0, a[2] + (b[2] - a[2]) * t1);
        z0.max(z1) >= self.z && z0.min(z1) <= self.z + self.crown_base
    }

    fn crown_chord(&self, a: Vec3, b: Vec3) -> f64 {
        let half_h = 0.5 * (self.height - self.crown_base);
        if half_h <= 0.0 || self.crown_r <= 0.0 {
            return 0.0;
        }
        let center = [self.crown_x, self.crown_y, self.z + self.crown_base + half_h];
        let scale = [1.0 / self.crown_r, 1.0 / self.crown_r, 1.0 / half_h];
        let f: [f64; 3] = std::array::from_fn(|i| (a[i] - center[i]) * scale[i]);
        let d: [f64; 3] = std::array::from_fn(|i| (b[i] - a[i]) * scale[i]);
        let qa: f64 = d.iter().map(|v| v * v).sum();
        if qa < 1e-12 {
            return 0.0;
        }
        let qb: f64 = 2.0 * f.iter().zip(&d).map(|(p, q)| p * q).sum::<f64>();
        let qc: f64 = f.iter().map(|v| v * v).sum::<f64>() - 1.0;
        let disc = qb * qb - 4.0 * qa * qc;
        if disc <= 0.0 {
            return 0.0;
        }
        let root = disc.sqrt();
        let t0 = ((-qb - root) / (2.0 * qa)).max(0.0);
        let t1 = ((-qb + root) / (2.0 * qa)).min(1.0);
        let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
        (t1 - t0).max(0.0) * len
    }
}

impl World {
    pub fn canopy_transmission(&self, a: Vec3, b: Vec3) -> f64 {
        let Some(grid) = self.grid() else {
            return self.canopy_transmission_reference(a, b);
        };
        let mut near = Vec::new();
        grid.trees_on(a, b, &mut near);
        self.transmission_over(a, b, near.into_iter().map(|i| &self.trees[i as usize]))
    }

    #[doc(hidden)]
    pub fn canopy_transmission_reference(&self, a: Vec3, b: Vec3) -> f64 {
        self.transmission_over(a, b, self.trees.iter())
    }

    fn transmission_over<'t>(&self, a: Vec3, b: Vec3, trees: impl Iterator<Item = &'t Tree>) -> f64 {
        let mut foliage = 0.0;
        for tree in trees {
            let reach = crate::index::tree_reach(tree);
            if dist2_to_segment(a, b, [tree.x, tree.y]) > reach * reach {
                continue;
            }
            if tree.trunk_blocks(a, b) {
                return 0.0;
            }
            foliage += tree.crown_chord(a, b);
        }
        (-self.canopy.density * foliage).exp()
    }

    pub fn sight(&self, a: Vec3, b: Vec3) -> bool {
        self.los(a, b) && self.canopy_transmission(a, b) >= self.canopy.block_t
    }

    pub fn sight_hinted(&self, a: Vec3, b: Vec3, hint: &mut u32) -> bool {
        self.los_hinted(a, b, hint) && self.canopy_transmission(a, b) >= self.canopy.block_t
    }
}
