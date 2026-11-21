
use crate::{Prism, Vec3, World};

const CLEAR: f64 = 1e-6;

impl Prism {
    pub fn nearest_edge(&self, x: f64, y: f64) -> ([f64; 2], f64) {
        let mut best = ([x, y], f64::INFINITY);
        for (a, b) in self.edges() {
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let len2 = ex * ex + ey * ey;
            let t = if len2 > 0.0 {
                (((x - a[0]) * ex + (y - a[1]) * ey) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let q = [a[0] + t * ex, a[1] + t * ey];
            let d = (x - q[0]).hypot(y - q[1]);
            if d < best.1 {
                best = (q, d);
            }
        }
        best
    }

    pub fn overlaps_disc(&self, x: f64, y: f64, r: f64) -> bool {
        self.bbox_near(x, y, r) && (self.contains_xy(x, y) || self.nearest_edge(x, y).1 < r - CLEAR * 0.5)
    }

    pub fn disc_exit(&self, x: f64, y: f64, r: f64) -> Option<[f64; 2]> {
        if !self.bbox_near(x, y, r) {
            return None;
        }
        let inside = self.contains_xy(x, y);
        let (q, d) = self.nearest_edge(x, y);
        if !inside && d >= r - CLEAR * 0.5 {
            return None;
        }
        let n = if d > 1e-9 {
            let k = if inside { 1.0 } else { -1.0 } / d;
            [(q[0] - x) * k, (q[1] - y) * k]
        } else {
            self.outward_normal_at(q)
        };
        let shift = if inside { d + r + CLEAR } else { r + CLEAR - d };
        Some([n[0] * shift, n[1] * shift])
    }

    fn outward_normal_at(&self, q: [f64; 2]) -> [f64; 2] {
        let mut best = (f64::INFINITY, [1.0, 0.0]);
        for (a, b) in self.edges() {
            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
            let len = ex.hypot(ey);
            if len == 0.0 {
                continue;
            }
            let t = (((q[0] - a[0]) * ex + (q[1] - a[1]) * ey) / (len * len)).clamp(0.0, 1.0);
            let d = (q[0] - a[0] - t * ex).hypot(q[1] - a[1] - t * ey);
            if d < best.0 {
                best = (d, [ey / len, -ex / len]);
            }
        }
        let n = best.1;
        if self.contains_xy(q[0] + n[0] * 1e-4, q[1] + n[1] * 1e-4) {
            [-n[0], -n[1]]
        } else {
            n
        }
    }
}

impl World {
    pub fn prisms_near(&self, x: f64, y: f64, r: f64) -> impl Iterator<Item = (usize, &Prism)> {
        self.prisms.iter().enumerate().filter(move |(_, p)| p.bbox_near(x, y, r))
    }

    pub fn roof_at(&self, x: f64, y: f64, z_max: f64) -> Option<f64> {
        self.prisms
            .iter()
            .filter(|p| p.top <= z_max && p.contains_xy(x, y))
            .map(|p| p.top)
            .reduce(f64::max)
    }

    pub fn surface_at(&self, x: f64, y: f64, z_max: f64) -> f64 {
        let ground = self.terrain.height(x, y);
        self.roof_at(x, y, z_max).map_or(ground, |roof| roof.max(ground))
    }

    pub fn ground_under(&self, x: f64, y: f64, radius: f64, z_max: f64) -> f64 {
        let ground = self.surface_at(x, y, z_max);
        self.prisms_near(x, y, radius)
            .filter(|(_, p)| p.top <= z_max && p.overlaps_disc(x, y, radius))
            .map(|(_, p)| p.top)
            .fold(ground, f64::max)
    }

    pub fn ceiling_at(&self, x: f64, y: f64, z: f64) -> Option<f64> {
        self.prisms
            .iter()
            .filter(|p| p.base >= z && p.contains_xy(x, y))
            .map(|p| p.base)
            .reduce(f64::min)
    }

    pub fn solid_at(&self, p: Vec3) -> bool {
        p[2] < self.terrain.height(p[0], p[1]) || self.prisms.iter().any(|prism| prism.contains(p))
    }
}
