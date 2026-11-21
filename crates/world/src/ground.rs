//! The surface a mech stands on. "You are a vertical cylinder standing on the
//! surface under your center" (protocol.md §7). Also the disc-against-footprint
//! geometry the movement rules collide with.

use crate::{Prism, Vec3, World};

/// A disc this much past an edge counts as clear of it, so a disc pushed out
/// by [`Prism::disc_exit`] no longer overlaps.
const CLEAR: f64 = 1e-6;

impl Prism {
    /// The point of the footprint's boundary (ring or hole edge) nearest `(x, y)`, and its distance.
    pub fn nearest_edge(&self, x: f64, y: f64) -> ([f64; 2], f64) {
        self.nearest_edge_indexed(x, y)
    }

    /// True when a disc of radius `r` at `(x, y)` overlaps the footprint by more than a touch.
    pub fn overlaps_disc(&self, x: f64, y: f64, r: f64) -> bool {
        self.bbox_near(x, y, r) && (self.contains_xy(x, y) || (self.edges_may_be_within(x, y, r) && self.nearest_edge(x, y).1 < r - CLEAR * 0.5))
    }

    /// The shortest horizontal shift that takes a disc of radius `r` at
    /// `(x, y)` clear of the footprint, or `None` when it already is.
    pub fn disc_exit(&self, x: f64, y: f64, r: f64) -> Option<[f64; 2]> {
        if !self.bbox_near(x, y, r) {
            return None;
        }
        let inside = self.contains_xy(x, y);
        if !inside && !self.edges_may_be_within(x, y, r) {
            return None;
        }
        let (q, d) = self.nearest_edge(x, y);
        if !inside && d >= r - CLEAR * 0.5 {
            return None;
        }
        // Outward unit normal at q: away from the footprint.
        let n = if d > 1e-9 {
            let k = if inside { 1.0 } else { -1.0 } / d;
            [(q[0] - x) * k, (q[1] - y) * k]
        } else {
            self.outward_normal_at(q)
        };
        let shift = if inside { d + r + CLEAR } else { r + CLEAR - d };
        Some([n[0] * shift, n[1] * shift])
    }

    /// Outward normal at a boundary point, for a disc centered exactly on an edge.
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
    /// Prisms whose footprint bounding box comes within `r` of `(x, y)`, with their indices.
    pub fn prisms_near(&self, x: f64, y: f64, r: f64) -> impl Iterator<Item = (usize, &Prism)> {
        self.prisms_near_indexed(x, y, r)
    }

    /// Top of the highest prism over `(x, y)` that is not above `z_max`.
    pub fn roof_at(&self, x: f64, y: f64, z_max: f64) -> Option<f64> {
        self.roof_at_indexed(x, y, z_max)
    }

    /// Height of the surface under the point `(x, y)` reachable from `z_max`:
    /// the terrain, or a prism top at or below `z_max` if one is higher.
    pub fn surface_at(&self, x: f64, y: f64, z_max: f64) -> f64 {
        let ground = self.terrain.height(x, y);
        self.roof_at(x, y, z_max).map_or(ground, |roof| roof.max(ground))
    }

    /// Height a vertical cylinder of `radius` rests at when coming down from `z_max`:
    /// the surface under its center, or a roof its disc still overlaps
    /// ("you land on the edge instead of dropping into the wall below").
    pub fn ground_under(&self, x: f64, y: f64, radius: f64, z_max: f64) -> f64 {
        let ground = self.surface_at(x, y, z_max);
        self.prisms_near(x, y, radius)
            .filter(|(_, p)| p.top <= z_max && p.overlaps_disc(x, y, radius))
            .map(|(_, p)| p.top)
            .fold(ground, f64::max)
    }

    /// Underside of the lowest prism over `(x, y)` whose base is at or above `z`: a bridge deck or an overhang.
    pub fn ceiling_at(&self, x: f64, y: f64, z: f64) -> Option<f64> {
        self.ceiling_at_indexed(x, y, z)
    }

    /// True when the point is under the terrain or inside a prism.
    pub fn solid_at(&self, p: Vec3) -> bool {
        p[2] < self.terrain.height(p[0], p[1]) || self.prisms_at(p[0], p[1]).any(|(_, prism)| prism.contains(p))
    }
}
