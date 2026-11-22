//! Port of `mech-bot/world.py`: the Python bot's own map model. Terrain
//! heights, prisms rasterized at 2 m, a sampled line of sight, a 4 m
//! walkability grid with cover distances and connected components, a
//! budgeted flow field, and roof candidates.
//!
//! It deliberately does not use `World::los` or `World::raycast`: the bot's
//! choices depend on this coarser model, and parity with the Python bot needs
//! the same answers. Grids that Python keeps in `array("f")` are `f32` here
//! for the same reason.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use world::{PrismKind, World};

pub const NONE_TOP: f64 = -1e9;
const SQRT2: f64 = std::f64::consts::SQRT_2;
/// Neighbour offsets and step costs, in Python's order.
pub const NBR: [(i64, i64, f64); 8] = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, SQRT2),
    (1, -1, SQRT2),
    (-1, 1, SQRT2),
    (-1, -1, SQRT2),
];

pub struct Roof {
    pub c: (f64, f64),
    pub top: f64,
    pub area: f64,
    pub verts: Vec<(f64, f64)>,
}

/// The static part, shared by every mech in a process.
pub struct NavMap {
    // terrain vertex grid
    tx0: f64,
    ty0: f64,
    tstep: f64,
    tnx: usize,
    tny: usize,
    h: Vec<f32>,
    // prism raster
    rs: f64,
    pub rx0: f64,
    pub ry0: f64,
    rw: usize,
    rh: usize,
    rtop: Vec<f32>,
    rbase: Vec<f32>,
    // walk grid
    pub cs: f64,
    pub nw: usize,
    pub nh: usize,
    pub blocked: Vec<u8>,
    pub cover: Vec<u8>,
    pub comp: Vec<i32>,
    pub roofs: Vec<Roof>,
}

/// Mech numbers the grid depends on.
#[derive(Clone, Copy, PartialEq)]
pub struct MechDims {
    pub step_up: f64,
    pub height: f64,
    pub max_slope_deg: f64,
}

impl NavMap {
    /// One map per directory and mech shape per process. Falls back to a flat
    /// empty square when the map does not load, so the bot still plays.
    pub fn shared(dir: &str, dims: MechDims) -> Arc<NavMap> {
        type Cache = Mutex<Vec<(PathBuf, [u64; 3], Arc<NavMap>)>>;
        static CACHE: OnceLock<Cache> = OnceLock::new();
        let key = [
            dims.step_up.to_bits(),
            dims.height.to_bits(),
            dims.max_slope_deg.to_bits(),
        ];
        let path = PathBuf::from(dir);
        let mut cache = CACHE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, nav)) = cache.iter().find(|(p, k, _)| *p == path && *k == key) {
            return nav.clone();
        }
        let world = match World::shared(Path::new(dir)) {
            Ok(world) => world,
            Err(e) => {
                if !dir.is_empty() {
                    eprintln!("policy: no map, navigating a flat square: {e}");
                }
                Arc::new(World::flat(600.0))
            }
        };
        let nav = Arc::new(NavMap::build(&world, dims));
        cache.push((path, key, nav.clone()));
        nav
    }

    pub fn build(world: &World, dims: MechDims) -> NavMap {
        let t = &world.terrain;
        let (tnx, tny) = (t.nx, t.ny);
        let mut h = Vec::with_capacity(tnx * tny);
        for j in 0..tny {
            for i in 0..tnx {
                h.push(t.height(t.min_x + i as f64 * t.step, t.min_y + j as f64 * t.step) as f32);
            }
        }
        let rs = 2.0;
        let rw = (tnx as f64 * t.step / rs).ceil() as usize;
        let rh = (tny as f64 * t.step / rs).ceil() as usize;
        let mut nav = NavMap {
            tx0: t.min_x,
            ty0: t.min_y,
            tstep: t.step,
            tnx,
            tny,
            h,
            rs,
            rx0: t.min_x,
            ry0: t.min_y,
            rw,
            rh,
            rtop: vec![NONE_TOP as f32; rw * rh],
            rbase: vec![1e9; rw * rh],
            cs: 4.0,
            nw: 0,
            nh: 0,
            blocked: Vec::new(),
            cover: Vec::new(),
            comp: Vec::new(),
            roofs: Vec::new(),
        };
        let rwall = nav.rasterize(world, dims);
        nav.roofs = nav.roof_candidates(world);
        nav.nav_grid(&rwall, dims.max_slope_deg);
        nav
    }

    // ---------- terrain ----------
    pub fn ground(&self, x: f64, y: f64) -> f64 {
        let fx = (x - self.tx0) / self.tstep;
        let fy = (y - self.ty0) / self.tstep;
        let (nx, ny) = (self.tnx as i64, self.tny as i64);
        let i = (fx as i64).clamp(0, nx - 2);
        let j = (fy as i64).clamp(0, ny - 2);
        let ax = (fx - i as f64).clamp(0.0, 1.0);
        let ay = (fy - j as f64).clamp(0.0, 1.0);
        let k = (j * nx + i) as usize;
        let n = self.tnx;
        let at = |k: usize| f64::from(self.h[k]);
        let top = at(k) + (at(k + 1) - at(k)) * ax;
        let bot = at(k + n) + (at(k + n + 1) - at(k + n)) * ax;
        top + (bot - top) * ay
    }

    // ---------- prisms ----------
    fn rasterize(&mut self, world: &World, dims: MechDims) -> Vec<u8> {
        let (rs, x0, y0, rw, rh) = (self.rs, self.rx0, self.ry0, self.rw as i64, self.rh as i64);
        let mut rwall = vec![0u8; self.rw * self.rh];
        for p in &world.prisms {
            let mut edges = Vec::new();
            for poly in std::iter::once(&p.ring).chain(p.holes.iter()) {
                let m = poly.len();
                for k in 0..m {
                    let (a, b) = (poly[k], poly[(k + 1) % m]);
                    if a[1] != b[1] {
                        edges.push((a[0], a[1], b[0], b[1]));
                    }
                }
            }
            let ys = p.ring.iter().map(|v| v[1]);
            let ymin = ys.clone().fold(f64::INFINITY, f64::min);
            let ymax = ys.fold(f64::NEG_INFINITY, f64::max);
            let j0 = (((ymin - y0) / rs - 0.5).ceil() as i64).max(0);
            let j1 = (((ymax - y0) / rs - 0.5).floor() as i64).min(rh - 1);
            let bridge = p.kind == PrismKind::Bridge;
            for j in j0..=j1 {
                let yc = y0 + (j as f64 + 0.5) * rs;
                let mut xs: Vec<f64> = edges
                    .iter()
                    .filter(|(_, ay, _, by)| (*ay <= yc) != (*by <= yc))
                    .map(|(ax, ay, bx, by)| ax + (yc - ay) * (bx - ax) / (by - ay))
                    .collect();
                xs.sort_by(f64::total_cmp);
                let mut k = 0;
                while k + 1 < xs.len() {
                    let i0 = (((xs[k] - x0) / rs - 0.5).ceil() as i64).max(0);
                    let i1 = (((xs[k + 1] - x0) / rs - 0.5).floor() as i64).min(rw - 1);
                    for i in i0..=i1 {
                        let c = (j * rw + i) as usize;
                        if p.top > f64::from(self.rtop[c]) {
                            self.rtop[c] = p.top as f32;
                        }
                        if p.base < f64::from(self.rbase[c]) {
                            self.rbase[c] = p.base as f32;
                        }
                        if !bridge {
                            let g = self.ground(x0 + (i as f64 + 0.5) * rs, yc);
                            if p.top - g > dims.step_up && p.base - g < dims.height {
                                rwall[c] = 1;
                            }
                        }
                    }
                    k += 2;
                }
            }
        }
        rwall
    }

    fn rcell(&self, x: f64, y: f64) -> Option<usize> {
        let i = ((x - self.rx0) / self.rs) as i64;
        let j = ((y - self.ry0) / self.rs) as i64;
        (0..self.rw as i64).contains(&i).then_some(())?;
        (0..self.rh as i64)
            .contains(&j)
            .then(|| (j * self.rw as i64 + i) as usize)
    }

    pub fn roof_top(&self, x: f64, y: f64) -> f64 {
        self.rcell(x, y)
            .map_or(NONE_TOP, |c| f64::from(self.rtop[c]))
    }

    /// True if the segment a->b clears every prism and the terrain, sampled every 1.5 m.
    pub fn los(&self, a: [f64; 3], b: [f64; 3]) -> bool {
        let (dx, dy, dz) = (b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        if d < 1e-6 {
            return true;
        }
        let n = (d / 1.5) as i64;
        if n <= 1 {
            return true;
        }
        let (rw, rh, tnx, tny) = (
            self.rw as i64,
            self.rh as i64,
            self.tnx as i64,
            self.tny as i64,
        );
        for k in 1..n {
            let f = k as f64 / n as f64;
            let (x, y, z) = (a[0] + dx * f, a[1] + dy * f, a[2] + dz * f);
            let (i, j) = (
                ((x - self.rx0) / self.rs) as i64,
                ((y - self.ry0) / self.rs) as i64,
            );
            if (0..rw).contains(&i) && (0..rh).contains(&j) {
                let c = (j * rw + i) as usize;
                let top = f64::from(self.rtop[c]);
                if top > NONE_TOP && f64::from(self.rbase[c]) <= z && z <= top {
                    return false;
                }
            }
            let ti = ((x - self.tx0) / self.tstep + 0.5) as i64;
            let tj = ((y - self.ty0) / self.tstep + 0.5) as i64;
            if (0..tnx).contains(&ti)
                && (0..tny).contains(&tj)
                && z < f64::from(self.h[(tj * tnx + ti) as usize]) - 0.3
            {
                return false;
            }
        }
        true
    }

    // ---------- walkability ----------
    fn nav_grid(&mut self, rwall: &[u8], max_slope_deg: f64) {
        let k = ((self.cs / self.rs).round() as usize).max(1);
        let (nw, nh) = (self.rw / k, self.rh / k);
        self.nw = nw;
        self.nh = nh;
        let n = nw * nh;
        let mut blocked = vec![0u8; n];
        let mut gh = vec![0f32; n];
        for j in 0..nh {
            for i in 0..nw {
                let c = j * nw + i;
                let (x, y) = self.ncenter(c);
                gh[c] = self.ground(x, y) as f32;
                if i == 0 || j == 0 || i == nw - 1 || j == nh - 1 {
                    blocked[c] = 1;
                    continue;
                }
                for dj in 0..k {
                    let base = (j * k + dj) * self.rw + i * k;
                    if rwall[base..base + k].iter().any(|&w| w != 0) {
                        blocked[c] = 1;
                        break;
                    }
                }
            }
        }
        let lim = self.cs * max_slope_deg.to_radians().tan();
        for j in 1..nh.saturating_sub(1) {
            for i in 1..nw - 1 {
                let c = j * nw + i;
                let g = f64::from(gh[c]);
                let steep = |o: usize| (f64::from(gh[o]) - g).abs() > lim;
                if steep(c + 1) || steep(c - 1) || steep(c + nw) || steep(c - nw) {
                    blocked[c] = 1;
                }
            }
        }
        // cover[c]: steps to the nearest wall cell, capped at 15
        let mut cover = vec![15u8; n];
        let mut q = VecDeque::new();
        for c in 0..n {
            if blocked[c] != 0 {
                cover[c] = 0;
                q.push_back(c);
            }
        }
        while let Some(c) = q.pop_front() {
            let d = cover[c] + 1;
            if d >= 15 {
                continue;
            }
            let (i, j) = ((c % nw) as i64, (c / nw) as i64);
            for (di, dj, _) in NBR {
                let (ii, jj) = (i + di, j + dj);
                if (0..nw as i64).contains(&ii) && (0..nh as i64).contains(&jj) {
                    let cc = (jj * nw as i64 + ii) as usize;
                    if cover[cc] > d {
                        cover[cc] = d;
                        q.push_back(cc);
                    }
                }
            }
        }
        // comp[c]: 4-connected walkable component id (0 = blocked)
        let mut comp = vec![0i32; n];
        let mut label = 0;
        for s in 0..n {
            if blocked[s] != 0 || comp[s] != 0 {
                continue;
            }
            label += 1;
            comp[s] = label;
            q.push_back(s);
            while let Some(c) = q.pop_front() {
                for o in [1i64, -1, nw as i64, -(nw as i64)] {
                    let cc = c as i64 + o;
                    if (0..n as i64).contains(&cc) {
                        let cc = cc as usize;
                        if blocked[cc] == 0 && comp[cc] == 0 {
                            comp[cc] = label;
                            q.push_back(cc);
                        }
                    }
                }
            }
        }
        self.blocked = blocked;
        self.cover = cover;
        self.comp = comp;
    }

    /// Walk-grid cell index, or -1 off the grid (Python's convention).
    pub fn ncell(&self, x: f64, y: f64) -> i64 {
        let i = ((x - self.rx0) / self.cs) as i64;
        let j = ((y - self.ry0) / self.cs) as i64;
        if (0..self.nw as i64).contains(&i) && (0..self.nh as i64).contains(&j) {
            j * self.nw as i64 + i
        } else {
            -1
        }
    }

    pub fn ncenter(&self, c: usize) -> (f64, f64) {
        (
            self.rx0 + ((c % self.nw) as f64 + 0.5) * self.cs,
            self.ry0 + ((c / self.nw) as f64 + 0.5) * self.cs,
        )
    }

    pub fn is_blocked(&self, c: usize) -> bool {
        self.blocked[c] != 0
    }

    pub fn free(&self, x: f64, y: f64) -> bool {
        let c = self.ncell(x, y);
        c >= 0 && self.blocked[c as usize] == 0
    }

    pub fn component(&self, x: f64, y: f64) -> i32 {
        self.nearest_free(x, y, 3).map_or(0, |c| self.comp[c])
    }

    pub fn clear_ray(&self, x: f64, y: f64, dx: f64, dy: f64, length: f64, step: f64) -> bool {
        let mut d = step;
        while d <= length {
            if !self.free(x + dx * d, y + dy * d) {
                return false;
            }
            d += step;
        }
        true
    }

    pub fn nearest_free(&self, x: f64, y: f64, rings: i64) -> Option<usize> {
        let c = self.ncell(x, y);
        if c >= 0 && self.blocked[c as usize] == 0 {
            return Some(c as usize);
        }
        let (mut best, mut bd) = (None, 1e18);
        let i0 = ((x - self.rx0) / self.cs) as i64;
        let j0 = ((y - self.ry0) / self.cs) as i64;
        for dj in -rings..=rings {
            for di in -rings..=rings {
                let (i, j) = (i0 + di, j0 + dj);
                if (0..self.nw as i64).contains(&i) && (0..self.nh as i64).contains(&j) {
                    let cc = (j * self.nw as i64 + i) as usize;
                    if self.blocked[cc] == 0 {
                        let (cx, cy) = self.ncenter(cc);
                        let d = (cx - x).powi(2) + (cy - y).powi(2);
                        if d < bd {
                            best = Some(cc);
                            bd = d;
                        }
                    }
                }
            }
        }
        best
    }

    // ---------- roofs ----------
    fn roof_candidates(&self, world: &World) -> Vec<Roof> {
        let mut out = Vec::new();
        for p in world
            .prisms
            .iter()
            .filter(|p| p.kind == PrismKind::Building)
        {
            let ring = &p.ring;
            let m = ring.len();
            let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
            for k in 0..m {
                let ([x1, y1], [x2, y2]) = (ring[k], ring[(k + 1) % m]);
                let cr = x1 * y2 - x2 * y1;
                a += cr;
                cx += (x1 + x2) * cr;
                cy += (y1 + y2) * cr;
            }
            if f64::abs(a) < 1e-6 {
                continue;
            }
            let (cx, cy, area) = (cx / (3.0 * a), cy / (3.0 * a), a.abs() / 2.0);
            if (self.roof_top(cx, cy) - p.top).abs() > 0.5 {
                continue;
            }
            out.push(Roof {
                c: (cx, cy),
                top: p.top,
                area,
                verts: ring.iter().map(|v| (v[0], v[1])).collect(),
            });
        }
        out
    }

    /// A free ground point just outside the roof's footprint, nearest to (fx, fy):
    /// `(qx, qy, edge_x, edge_y)`.
    pub fn approach_point(&self, roof: &Roof, fx: f64, fy: f64) -> Option<(f64, f64, f64, f64)> {
        let offset = 5.0;
        let (cx, cy) = roof.c;
        let (mut best, mut bd) = (None, 1e18);
        let vs = &roof.verts;
        for k in 0..vs.len() {
            let (ax, ay) = vs[k];
            let (bx, by) = vs[(k + 1) % vs.len()];
            for t in [0.25, 0.5, 0.75] {
                let (px, py) = (ax + (bx - ax) * t, ay + (by - ay) * t);
                let (ex, ey) = (bx - ax, by - ay);
                let l = ex.hypot(ey);
                if l < 3.0 {
                    continue;
                }
                let (mut nx, mut ny) = (ey / l, -ex / l);
                if (px - cx) * nx + (py - cy) * ny < 0.0 {
                    nx = -nx;
                    ny = -ny;
                }
                let (qx, qy) = (px + nx * offset, py + ny * offset);
                if !self.free(qx, qy) || self.roof_top(qx, qy) > NONE_TOP {
                    continue;
                }
                let d = (qx - fx).powi(2) + (qy - fy).powi(2);
                if d < bd {
                    best = Some((qx, qy, px, py));
                    bd = d;
                }
            }
        }
        best
    }
}

pub type FieldKey = (i64, i64, i64);

struct Job {
    key: FieldKey,
    dist: Vec<f32>,
    q: VecDeque<usize>,
}

/// The flow field toward a goal disc, built a budget of queue pops at a time
/// (Python builds it within a wall-clock budget per decision).
#[derive(Default)]
pub struct Flow {
    pub field_key: Option<FieldKey>,
    field: Option<Arc<Vec<f32>>>,
    fields: Vec<(FieldKey, Arc<Vec<f32>>)>,
    job: Option<Job>,
}

impl Flow {
    pub fn key(nav: &NavMap, x: f64, y: f64, r: f64) -> FieldKey {
        (
            (x / nav.cs).floor() as i64,
            (y / nav.cs).floor() as i64,
            (r / nav.cs).floor() as i64,
        )
    }

    /// Ask for a flow field toward the disc (x, y, r). Work happens in `pump`.
    pub fn request(&mut self, nav: &NavMap, x: f64, y: f64, r: f64) {
        let key = Flow::key(nav, x, y, r);
        if Some(key) == self.field_key || self.job.as_ref().is_some_and(|j| j.key == key) {
            return;
        }
        if let Some((_, f)) = self.fields.iter().find(|(k, _)| *k == key) {
            self.field_key = Some(key);
            self.field = Some(f.clone());
            return;
        }
        let n = nav.nw * nav.nh;
        let mut dist = vec![-1f32; n];
        let mut q = VecDeque::new();
        let rr = r.max(nav.cs * 0.75);
        let ci = ((x - nav.rx0) / nav.cs) as i64;
        let cj = ((y - nav.ry0) / nav.cs) as i64;
        let span = (rr / nav.cs) as i64 + 1;
        for j in (cj - span).max(0)..(cj + span + 1).min(nav.nh as i64) {
            for i in (ci - span).max(0)..(ci + span + 1).min(nav.nw as i64) {
                let c = (j * nav.nw as i64 + i) as usize;
                if nav.blocked[c] != 0 {
                    continue;
                }
                let (cx, cy) = nav.ncenter(c);
                if (cx - x).powi(2) + (cy - y).powi(2) <= rr * rr {
                    dist[c] = 0.0;
                    q.push_back(c);
                }
            }
        }
        if q.is_empty() {
            let Some(c) = nav.nearest_free(x, y, 8) else {
                return;
            };
            dist[c] = 0.0;
            q.push_back(c);
        }
        self.job = Some(Job { key, dist, q });
    }

    /// Advance the pending search by at most `pops` queue pops (label-correcting, octile costs).
    pub fn pump(&mut self, nav: &NavMap, pops: u64) {
        let Some(job) = self.job.as_mut() else { return };
        let (nw, nh) = (nav.nw as i64, nav.nh as i64);
        let blocked = &nav.blocked;
        let mut count = 0u64;
        while !job.q.is_empty() {
            if count >= pops {
                return;
            }
            count += 1;
            let c = job.q.pop_front().expect("queue is not empty");
            let d = f64::from(job.dist[c]);
            let (i, j) = (c as i64 % nw, c as i64 / nw);
            for (di, dj, w) in NBR {
                let (ii, jj) = (i + di, j + dj);
                if ii < 0 || jj < 0 || ii >= nw || jj >= nh {
                    continue;
                }
                let cc = (jj * nw + ii) as usize;
                if blocked[cc] != 0 {
                    continue;
                }
                if di != 0
                    && dj != 0
                    && (blocked[(j * nw + ii) as usize] != 0
                        || blocked[(jj * nw + i) as usize] != 0)
                {
                    continue;
                }
                let nd = d + w;
                let old = f64::from(job.dist[cc]);
                if old < 0.0 || nd < old - 1e-3 {
                    job.dist[cc] = nd as f32;
                    job.q.push_back(cc);
                }
            }
        }
        let job = self.job.take().expect("a job was running");
        let field = Arc::new(job.dist);
        self.field_key = Some(job.key);
        self.field = Some(field.clone());
        self.fields.push((job.key, field));
        if self.fields.len() > 4 {
            self.fields.remove(0);
        }
    }

    /// Unit (dx, dy) along the field. `Some((0, 0))` at the goal, `None` with no usable field here.
    pub fn dir(&self, nav: &NavMap, x: f64, y: f64) -> Option<(f64, f64)> {
        let f = self.field.as_ref()?;
        let c = nav.nearest_free(x, y, 2)?;
        if f[c] < 0.0 {
            return None;
        }
        if f[c] == 0.0 {
            return Some((0.0, 0.0));
        }
        let (nw, nh) = (nav.nw as i64, nav.nh as i64);
        let mut path = vec![c];
        let mut cur = c;
        for _ in 0..6 {
            let (i, j) = (cur as i64 % nw, cur as i64 / nw);
            let (mut best, mut bd) = (None, f64::from(f[cur]));
            for (di, dj, _) in NBR {
                let (ii, jj) = (i + di, j + dj);
                if (0..nw).contains(&ii) && (0..nh).contains(&jj) {
                    let cc = (jj * nw + ii) as usize;
                    let fc = f64::from(f[cc]);
                    if fc >= 0.0 && fc < bd - 1e-3 {
                        if di != 0
                            && dj != 0
                            && (nav.blocked[(j * nw + ii) as usize] != 0
                                || nav.blocked[(jj * nw + i) as usize] != 0)
                        {
                            continue;
                        }
                        best = Some(cc);
                        bd = fc;
                    }
                }
            }
            let Some(b) = best else { break };
            cur = b;
            path.push(cur);
        }
        for &target in path[1..].iter().rev() {
            let (tx, ty) = nav.ncenter(target);
            let (dx, dy) = (tx - x, ty - y);
            let l = dx.hypot(dy);
            if l < 1e-6 {
                continue;
            }
            if target == path[1] || nav.clear_ray(x, y, dx / l, dy / l, l, 1.5) {
                return Some((dx / l, dy / l));
            }
        }
        let (tx, ty) = nav.ncenter(c);
        let (dx, dy) = (tx - x, ty - y);
        let l = dx.hypot(dy);
        (l > 1e-6).then(|| (dx / l, dy / l))
    }
}
