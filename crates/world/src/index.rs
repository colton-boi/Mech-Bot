
use crate::{Vec3, World};

const CELL: f64 = 16.0;
const MARGIN: f64 = 0.5;

#[derive(Clone, Debug)]
pub(crate) struct Grid {
    min_x: f64,
    min_y: f64,
    nx: usize,
    ny: usize,
    prisms: Vec<Vec<u32>>,
    trees: Vec<Vec<u32>>,
    n_prisms: usize,
    n_trees: usize,
}

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
        let i = ((x - self.min_x) / CELL).clamp(0.0, (self.nx - 1) as f64) as usize;
        let j = ((y - self.min_y) / CELL).clamp(0.0, (self.ny - 1) as f64) as usize;
        (i, j)
    }

    fn cells_of(&self, b: &[f64; 4]) -> Vec<usize> {
        let (i0, j0) = self.cell_xy(b[0] - MARGIN, b[1] - MARGIN);
        let (i1, j1) = self.cell_xy(b[2] + MARGIN, b[3] + MARGIN);
        (j0..=j1).flat_map(|j| (i0..=i1).map(move |i| j * self.nx + i)).collect()
    }

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

    pub(crate) fn prisms_on(&self, a: Vec3, b: Vec3, out: &mut Vec<u32>) {
        out.clear();
        self.walk(a, b, |c| {
            out.extend_from_slice(&self.prisms[c]);
            true
        });
        out.sort_unstable();
        out.dedup();
    }

    pub(crate) fn find_prism_on(&self, a: Vec3, b: Vec3, mut hit: impl FnMut(usize) -> bool) -> Option<usize> {
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
    pub(crate) fn grid(&self) -> Option<&Grid> {
        let g = self.grid.get_or_init(|| Grid::build(self));
        g.fits(self).then_some(g)
    }
}
