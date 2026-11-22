//! A uniform grid over the map that lists, per cell, the prisms and trees that
//! may touch it. `raycast` and `canopy_transmission` walk only the cells a
//! segment crosses instead of scanning every prism and tree. The answers are
//! the same as the full scans in `ray.rs` and `canopy.rs`, which stay as the
//! reference (`*_reference`) and as the fallback.
//!
//! The grid is built on the first query. Tests push prisms and trees into a
//! `World` by hand, so the grid records how many of each it indexed and is
//! ignored when the counts no longer match.

use crate::{Vec3, World};

/// Cell side in meters.
const CELL: f64 = 16.0;
/// Objects are registered in cells this much wider than their footprint, so a
/// rounding slip in the cell walk near a corner cannot miss one.
const MARGIN: f64 = 0.5;

#[derive(Clone, Debug)]
pub(crate) struct Grid {
    min_x: f64,
    min_y: f64,
    nx: usize,
    ny: usize,
    /// Per cell, ascending indices into `World::prisms`.
    prisms: Vec<Vec<u32>>,
    /// Per cell, ascending indices into `World::trees`.
    trees: Vec<Vec<u32>>,
    n_prisms: usize,
    n_trees: usize,
}

/// How far from its trunk a tree can affect sight: the test `canopy.rs` uses.
pub(crate) fn tree_reach(t: &crate::Tree) -> f64 {
    t.crown_r + (t.crown_x - t.x).hypot(t.crown_y - t.y)
}

impl Grid {
    pub(crate) fn build(w: &World) -> Grid {
        let mut lo = [w.bounds.min_x, w.bounds.min_y];
        let mut hi = [w.bounds.max_x, w.bounds.max_y];
        let mut grow = |a: [f64; 2], b: [f64; 2]| {
            lo = [lo[0].min(a[0]), lo[1].min(a[1])];
            hi = [hi[0].max(b[0]), hi[1].max(b[1])];
        };
        let prism_boxes: Vec<[f64; 4]> = w.prisms.iter().map(|p| p.bbox).collect();
        let tree_boxes: Vec<[f64; 4]> = w
            .trees
            .iter()
            .map(|t| {
                let r = tree_reach(t).max(t.trunk_r);
                [t.x - r, t.y - r, t.x + r, t.y + r]
            })
            .collect();
        for b in prism_boxes.iter().chain(&tree_boxes) {
            grow([b[0], b[1]], [b[2], b[3]]);
        }
        let (min_x, min_y) = (lo[0] - CELL, lo[1] - CELL);
        let nx = (((hi[0] + CELL - min_x) / CELL).ceil() as usize).max(1);
        let ny = (((hi[1] + CELL - min_y) / CELL).ceil() as usize).max(1);
        let mut g = Grid {
            min_x,
            min_y,
            nx,
            ny,
            prisms: vec![Vec::new(); nx * ny],
            trees: vec![Vec::new(); nx * ny],
            n_prisms: w.prisms.len(),
            n_trees: w.trees.len(),
        };
        for (i, b) in prism_boxes.iter().enumerate() {
            for c in g.cells_of(b) {
                g.prisms[c].push(i as u32);
            }
        }
        for (i, b) in tree_boxes.iter().enumerate() {
            for c in g.cells_of(b) {
                g.trees[c].push(i as u32);
            }
        }
        g
    }

    pub(crate) fn fits(&self, w: &World) -> bool {
        self.n_prisms == w.prisms.len() && self.n_trees == w.trees.len()
    }

    fn cell_xy(&self, x: f64, y: f64) -> (usize, usize) {
        // Clamped to be non-negative first, so truncating is flooring.
        let i = ((x - self.min_x) / CELL).clamp(0.0, (self.nx - 1) as f64) as usize;
        let j = ((y - self.min_y) / CELL).clamp(0.0, (self.ny - 1) as f64) as usize;
        (i, j)
    }

    fn cells_of(&self, b: &[f64; 4]) -> Vec<usize> {
        let (i0, j0) = self.cell_xy(b[0] - MARGIN, b[1] - MARGIN);
        let (i1, j1) = self.cell_xy(b[2] + MARGIN, b[3] + MARGIN);
        (j0..=j1).flat_map(|j| (i0..=i1).map(move |i| j * self.nx + i)).collect()
    }

    /// Cells the segment's footprint crosses, in order. Every object lies
    /// inside the grid, so the parts of a segment outside it touch nothing and
    /// are clipped off first.
    fn walk(&self, a: Vec3, b: Vec3, mut visit: impl FnMut(usize) -> bool) {
        let (gx1, gy1) = (self.min_x + self.nx as f64 * CELL, self.min_y + self.ny as f64 * CELL);
        let d = [b[0] - a[0], b[1] - a[1]];
        let (mut t0, mut t1) = (0.0f64, 1.0f64);
        for (p, dp, lo, hi) in [(a[0], d[0], self.min_x, gx1), (a[1], d[1], self.min_y, gy1)] {
            if dp.abs() < 1e-12 {
                if p < lo || p > hi {
                    return;
                }
            } else {
                let (ta, tb) = ((lo - p) / dp, (hi - p) / dp);
                t0 = t0.max(ta.min(tb));
                t1 = t1.min(ta.max(tb));
            }
        }
        if t0 > t1 {
            return;
        }
        let start = [a[0] + d[0] * t0, a[1] + d[1] * t0];
        let end = [a[0] + d[0] * t1, a[1] + d[1] * t1];
        let (mut i, mut j) = self.cell_xy(start[0], start[1]);
        let (ie, je) = self.cell_xy(end[0], end[1]);
        let step_i: isize = if d[0] > 0.0 { 1 } else { -1 };
        let step_j: isize = if d[1] > 0.0 { 1 } else { -1 };
        let len = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt();
        let (ux, uy) = if len > 0.0 {
            ((end[0] - start[0]) / len, (end[1] - start[1]) / len)
        } else {
            (0.0, 0.0)
        };
        let next_edge = |c: usize, step: isize, min: f64| min + (c as f64 + if step > 0 { 1.0 } else { 0.0 }) * CELL;
        let mut t_max_x = if ux.abs() > 1e-12 {
            (next_edge(i, step_i, self.min_x) - start[0]) / ux
        } else {
            f64::INFINITY
        };
        let mut t_max_y = if uy.abs() > 1e-12 {
            (next_edge(j, step_j, self.min_y) - start[1]) / uy
        } else {
            f64::INFINITY
        };
        let t_dx = if ux.abs() > 1e-12 { CELL / ux.abs() } else { f64::INFINITY };
        let t_dy = if uy.abs() > 1e-12 { CELL / uy.abs() } else { f64::INFINITY };
        let max_steps = self.nx + self.ny + 2;
        for _ in 0..max_steps {
            if !visit(j * self.nx + i) || (i == ie && j == je) {
                return;
            }
            if t_max_x < t_max_y {
                if t_max_x > len {
                    return;
                }
                let ni = i as isize + step_i;
                if ni < 0 || ni >= self.nx as isize {
                    return;
                }
                i = ni as usize;
                t_max_x += t_dx;
            } else {
                if t_max_y > len {
                    return;
                }
                let nj = j as isize + step_j;
                if nj < 0 || nj >= self.ny as isize {
                    return;
                }
                j = nj as usize;
                t_max_y += t_dy;
            }
        }
    }

    /// Ascending, distinct indices of the prisms registered in the cells the segment crosses.
    pub(crate) fn prisms_on(&self, a: Vec3, b: Vec3, out: &mut Vec<u32>) {
        out.clear();
        self.walk(a, b, |c| {
            out.extend_from_slice(&self.prisms[c]);
            true
        });
        out.sort_unstable();
        out.dedup();
    }

    /// The first prism, registered in a cell the segment crosses, for which
    /// `hit` is true. Cells are visited from `a` toward `b` and the walk stops there.
    pub(crate) fn find_prism_on(&self, a: Vec3, b: Vec3, mut hit: impl FnMut(usize) -> bool) -> Option<usize> {
        // Prisms already tested, kept on the stack; past its size a repeat test is only wasted work.
        let mut tested = [u32::MAX; 48];
        let mut n = 0;
        let mut found = None;
        self.walk(a, b, |c| {
            for &i in &self.prisms[c] {
                if !tested[..n].contains(&i) {
                    if n < tested.len() {
                        tested[n] = i;
                        n += 1;
                    }
                    if hit(i as usize) {
                        found = Some(i as usize);
                        return false;
                    }
                }
            }
            true
        });
        found
    }

    /// Ascending, distinct indices of the trees registered in the cells the segment crosses.
    pub(crate) fn trees_on(&self, a: Vec3, b: Vec3, out: &mut Vec<u32>) {
        out.clear();
        self.walk(a, b, |c| {
            out.extend_from_slice(&self.trees[c]);
            true
        });
        out.sort_unstable();
        out.dedup();
    }
}

impl World {
    /// The grid, built on first use, or `None` when prisms or trees were added since.
    pub(crate) fn grid(&self) -> Option<&Grid> {
        let g = self.grid.get_or_init(|| Grid::build(self));
        g.fits(self).then_some(g)
    }

    /// Prisms that may hold `(x, y)` in plan, ascending: every prism whose
    /// bounding box holds the point is among them (each is registered in the
    /// cells of its box widened by `MARGIN`). All prisms without a grid.
    pub fn prisms_at(&self, x: f64, y: f64) -> PrismsAt<'_> {
        let (ids, all) = match self.grid() {
            Some(g) => {
                let (i, j) = g.cell_xy(x, y);
                (&g.prisms[j * g.nx + i][..], 0..0)
            }
            None => (&[][..], 0..self.prisms.len()),
        };
        PrismsAt {
            prisms: &self.prisms,
            ids: ids.iter(),
            all,
        }
    }

    /// [`World::prisms_near`] through the grid: the prisms whose bounding box
    /// comes within `r` of `(x, y)`, ascending, with their indices.
    pub fn prisms_near_indexed(&self, x: f64, y: f64, r: f64) -> impl Iterator<Item = (usize, &crate::Prism)> {
        let mut ids: Vec<u32> = Vec::new();
        match self.grid() {
            Some(g) => {
                let (i0, j0) = g.cell_xy(x - r, y - r);
                let (i1, j1) = g.cell_xy(x + r, y + r);
                for j in j0..=j1 {
                    for i in i0..=i1 {
                        ids.extend_from_slice(&g.prisms[j * g.nx + i]);
                    }
                }
                if j1 > j0 || i1 > i0 {
                    ids.sort_unstable();
                    ids.dedup();
                }
            }
            None => ids.extend(0..self.prisms.len() as u32),
        }
        ids.into_iter()
            .map(|k| (k as usize, &self.prisms[k as usize]))
            .filter(move |(_, p)| p.bbox_near(x, y, r))
    }

    /// [`World::ceiling_at`] through the grid, with the same answer.
    pub fn ceiling_at_indexed(&self, x: f64, y: f64, z: f64) -> Option<f64> {
        self.prisms_at(x, y)
            .filter(|(_, p)| p.base >= z && p.contains_xy(x, y))
            .map(|(_, p)| p.base)
            .reduce(f64::min)
    }

    /// [`World::roof_at`] through the grid, with the same answer.
    pub fn roof_at_indexed(&self, x: f64, y: f64, z_max: f64) -> Option<f64> {
        self.prisms_at(x, y)
            .filter(|(_, p)| p.top <= z_max && p.contains_xy(x, y))
            .map(|(_, p)| p.top)
            .reduce(f64::max)
    }

    /// [`World::surface_at`] through the grid, with the same answer.
    pub fn surface_at_indexed(&self, x: f64, y: f64, z_max: f64) -> f64 {
        let ground = self.terrain.height(x, y);
        self.roof_at_indexed(x, y, z_max).map_or(ground, |roof| roof.max(ground))
    }

    /// [`World::solid_at`] through the grid, with the same answer.
    pub fn solid_at_indexed(&self, p: Vec3) -> bool {
        p[2] < self.terrain.height(p[0], p[1]) || self.prisms_at(p[0], p[1]).any(|(_, prism)| prism.contains(p))
    }
}

/// Iterator for [`World::prisms_at`].
pub struct PrismsAt<'a> {
    prisms: &'a [crate::Prism],
    ids: std::slice::Iter<'a, u32>,
    /// Every prism, when there is no grid.
    all: std::ops::Range<usize>,
}

impl<'a> Iterator for PrismsAt<'a> {
    type Item = (usize, &'a crate::Prism);

    fn next(&mut self) -> Option<Self::Item> {
        let k = match self.ids.next() {
            Some(&k) => k as usize,
            None => self.all.next()?,
        };
        Some((k, &self.prisms[k]))
    }
}
