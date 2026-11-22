//! Navigation for the apex bot: a 2 m walkability grid over the static map,
//! built once per process and shared by every seat.
//!
//! The grid knows where a mech can stand (slope, walls, clearance for its
//! disc), which cells belong to the largest connected walkable area (`main`),
//! and how exposed each 4 m patch is to long sight lines. On top of it sit the
//! queries the policy needs: A* paths, straight-walk checks, cover spots,
//! hold spots, and jump plans off roofs and islands.

use super::types::{dist_xy, Rng64, ThreatInfo};
use proto::{DropObs, MechStats, VisionCfg, ZoneObs};
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::f64::consts::{SQRT_2, TAU};
use std::sync::{Arc, Mutex, OnceLock};
use world::World;

/// Side of a walkability cell, meters.
const CELL: f64 = 2.0;
/// Side of an exposure cell, meters (two walk cells).
const EX_CELL: f64 = 4.0;
/// Least clearance from the nearest obstacle a walkable cell must have, meters.
const MIN_CLEAR: f64 = 2.6;
/// Clearance at and beyond which a cell costs nothing extra.
const COMFORT_CLEAR: f64 = 6.0;
/// Extra cost per meter of clearance below `COMFORT_CLEAR`.
const CLEAR_COST: f64 = 0.12;
/// Weight on the A* heuristic: paths may be up to 25% longer than optimal.
const ASTAR_W: f32 = 1.25;
/// Most nodes one A* search expands.
const MAX_EXPANSIONS: usize = 12_000;
/// Rays cast per exposure cell.
const EXPOSURE_RAYS: usize = 16;
/// Longest exposure ray, meters.
const EXPOSURE_RANGE: f64 = 160.0;
/// How far `find_path` looks for a main cell near an off-grid start or goal, meters.
const SNAP_R: f64 = 40.0;
/// Most cells the escape search visits.
const ESCAPE_MAX_CELLS: usize = 3000;
/// Distances past the edge at which an escape jump may land, meters, nearest first.
const ESCAPE_LAND: [f64; 5] = [5.0, 8.0, 12.0, 16.0, 20.0];
/// Margin a cover or hold spot keeps inside the zone edge, meters.
const ZONE_MARGIN: f64 = 3.0;

/// The 8 grid neighbors with their step length in cells.
const DIRS: [(isize, isize, f32); 8] = [
    (1, 0, 1.0),
    (-1, 0, 1.0),
    (0, 1, 1.0),
    (0, -1, 1.0),
    (1, 1, SQRT_2 as f32),
    (1, -1, SQRT_2 as f32),
    (-1, 1, SQRT_2 as f32),
    (-1, -1, SQRT_2 as f32),
];

/// A jump off an isolated roof or island back onto the main walkable area.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EscapePlan {
    /// Where to stand before jumping: a cell at the island's edge.
    pub launch_xy: [f64; 2],
    /// Where the jump should come down, on the main area.
    pub land_xy: [f64; 2],
    /// Height the feet must clear on the way, meters.
    pub need_z: f64,
    /// Unit direction from launch to land.
    pub jump_dir: [f64; 2],
}

/// The walkability grid and its queries. See the module docs.
pub struct ApexNav {
    world: Arc<World>,
    radius: f64,
    height: f64,
    eye_height: f64,
    step_up: f64,
    /// Largest surface change between neighboring cells a mech walks over.
    max_dh: f64,
    min_x: f64,
    min_y: f64,
    nx: usize,
    ny: usize,
    /// Surface walked on at ground level: terrain, or a prism top low enough to step onto.
    surf: Vec<f32>,
    /// Clearance to the nearest obstacle, meters.
    clear: Vec<f32>,
    /// Per-meter path cost, 1 in the open, more near obstacles.
    cost: Vec<f32>,
    /// A mech can stand here.
    walk: Vec<bool>,
    /// Walkable and in the largest 4-connected walkable component.
    main: Vec<bool>,
    /// 4-connected walkable component of each cell, `u32::MAX` where not walkable.
    comp: Vec<u32>,
    /// Cell count of each component.
    comp_size: Vec<u32>,
    /// Id of the largest component (`u32::MAX` when nothing is walkable).
    main_id: u32,
    ex_nx: usize,
    ex_ny: usize,
    /// Fraction of open sight lines per 4 m cell, scaled to 0..=255.
    exposure: Vec<u8>,
}

impl std::fmt::Debug for ApexNav {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApexNav")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("walkable", &self.walk.iter().filter(|&&w| w).count())
            .field("main", &self.main.iter().filter(|&&m| m).count())
            .finish_non_exhaustive()
    }
}

/// Reused A* buffers, one set per thread. A generation stamp marks which
/// entries belong to the current search, so nothing is cleared between searches.
#[derive(Default)]
struct Scratch {
    gen: u32,
    stamp: Vec<u32>,
    closed: Vec<u32>,
    g: Vec<f32>,
    parent: Vec<u32>,
    heap: BinaryHeap<Reverse<(u32, u32)>>,
}

impl Scratch {
    fn begin(&mut self, n: usize) {
        if self.stamp.len() < n {
            self.stamp.resize(n, 0);
            self.closed.resize(n, 0);
            self.g.resize(n, 0.0);
            self.parent.resize(n, u32::MAX);
        }
        self.gen = self.gen.wrapping_add(1);
        if self.gen == 0 {
            self.stamp.fill(0);
            self.closed.fill(0);
            self.gen = 1;
        }
        self.heap.clear();
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Unit vector of `v`, or `None` when it is (nearly) zero.
fn unit(v: [f64; 2]) -> Option<[f64; 2]> {
    let len = v[0].hypot(v[1]);
    (len > 1e-9).then(|| [v[0] / len, v[1] / len])
}

/// True when `(x, y)` is at least `ZONE_MARGIN` inside the current circle; always true with no zone.
fn in_zone(zone: &ZoneObs, x: f64, y: f64) -> bool {
    zone.radius <= 0.0 || dist_xy([x, y], zone.center) <= zone.radius - ZONE_MARGIN
}

/// Up to `k` threats nearest `me`, nearest first.
fn nearest_threats(me: [f64; 3], threats: &[ThreatInfo], k: usize) -> Vec<&ThreatInfo> {
    let mut ts: Vec<&ThreatInfo> = threats.iter().collect();
    let d = |t: &ThreatInfo| (t.pos[0] - me[0]).hypot(t.pos[1] - me[1]);
    ts.sort_by(|a, b| d(a).total_cmp(&d(b)));
    ts.truncate(k);
    ts
}

impl ApexNav {
    /// The grid for map folder `dir` and these mech dimensions, built once per
    /// process. A folder that fails to load gives a flat 1200 m square.
    pub fn shared(dir: &str, mech: &MechStats, vision: &VisionCfg) -> Arc<ApexNav> {
        static CACHE: OnceLock<Mutex<HashMap<String, Arc<ApexNav>>>> = OnceLock::new();
        let key = format!(
            "{dir}|{}|{}|{}|{}|{}|{}",
            mech.radius,
            mech.height,
            mech.eye_height,
            mech.step_up,
            mech.max_slope_deg,
            vision.range
        );
        let mut cache = CACHE
            .get_or_init(Default::default)
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(nav) = cache.get(&key) {
            return nav.clone();
        }
        let world = World::shared(std::path::Path::new(dir))
            .unwrap_or_else(|_| Arc::new(World::flat(600.0)));
        let nav = Arc::new(ApexNav::build(world, mech, vision));
        cache.insert(key, nav.clone());
        nav
    }

    /// Builds the grid over `world.bounds`.
    pub fn build(world: Arc<World>, mech: &MechStats, vision: &VisionCfg) -> ApexNav {
        let b = world.bounds;
        let nx = (((b.max_x - b.min_x) / CELL).ceil() as usize).max(1);
        let ny = (((b.max_y - b.min_y) / CELL).ceil() as usize).max(1);
        let n = nx * ny;
        let (min_x, min_y) = (b.min_x, b.min_y);
        let step_up = mech.step_up;
        let height = mech.height;
        let eye_height = mech.eye_height;
        let slope_dh = CELL * mech.max_slope_deg.to_radians().tan().max(0.0);
        let max_dh = slope_dh.max(step_up);

        let mut ground = vec![0f32; n];
        for j in 0..ny {
            let y = min_y + (j as f64 + 0.5) * CELL;
            for i in 0..nx {
                let x = min_x + (i as f64 + 0.5) * CELL;
                ground[j * nx + i] = world.terrain.height(x, y) as f32;
            }
        }
        let mut surf = ground.clone();
        // Highest solid an eye-level ray from the ground here would meet.
        let mut obs = ground.clone();
        let mut blocked = vec![false; n];

        let cell_of = |x: f64, y: f64| -> Option<usize> {
            let fi = ((x - min_x) / CELL).floor();
            let fj = ((y - min_y) / CELL).floor();
            (fi >= 0.0 && fj >= 0.0 && (fi as usize) < nx && (fj as usize) < ny)
                .then(|| fj as usize * nx + fi as usize)
        };
        for p in &world.prisms {
            let mut apply = |k: usize, inside: bool| {
                let g = f64::from(ground[k]);
                if p.top > g + step_up && p.base < g + height - 0.1 {
                    blocked[k] = true;
                } else if inside && p.top <= g + step_up && p.top as f32 > surf[k] {
                    surf[k] = p.top as f32;
                }
                if p.base < g + eye_height && p.top as f32 > obs[k] {
                    obs[k] = p.top as f32;
                }
            };
            let [x0, y0, x1, y1] = p.bbox;
            let i0 = (((x0 - min_x) / CELL).floor().max(0.0) as usize).min(nx - 1);
            let i1 = (((x1 - min_x) / CELL).floor().max(0.0) as usize).min(nx - 1);
            let j0 = (((y0 - min_y) / CELL).floor().max(0.0) as usize).min(ny - 1);
            let j1 = (((y1 - min_y) / CELL).floor().max(0.0) as usize).min(ny - 1);
            for j in j0..=j1 {
                let y = min_y + (j as f64 + 0.5) * CELL;
                for i in i0..=i1 {
                    let x = min_x + (i as f64 + 0.5) * CELL;
                    if p.contains_xy(x, y) {
                        apply(j * nx + i, true);
                    }
                }
            }
            // Walls thinner than a cell can miss every center: mark the cells their edges cross.
            for (a, e) in p.edges() {
                let len = (e[0] - a[0]).hypot(e[1] - a[1]);
                let steps = (len / (0.5 * CELL)).ceil().max(1.0) as usize;
                for s in 0..=steps {
                    let t = s as f64 / steps as f64;
                    if let Some(k) = cell_of(a[0] + (e[0] - a[0]) * t, a[1] + (e[1] - a[1]) * t) {
                        apply(k, false);
                    }
                }
            }
        }

        // Standable before clearance: not inside a wall and no steep step to a standable neighbor.
        let mut stand = vec![false; n];
        for j in 0..ny {
            for i in 0..nx {
                let k = j * nx + i;
                if blocked[k] {
                    continue;
                }
                let s = surf[k];
                let mut ok = true;
                for (di, dj) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                    let (ni, nj) = (i as isize + di, j as isize + dj);
                    if ni < 0 || nj < 0 || ni as usize >= nx || nj as usize >= ny {
                        continue;
                    }
                    let m = nj as usize * nx + ni as usize;
                    if !blocked[m] && f64::from((surf[m] - s).abs()) > max_dh {
                        ok = false;
                        break;
                    }
                }
                stand[k] = ok;
            }
        }

        let clear = chamfer_clearance(&stand, nx, ny);
        let thr = MIN_CLEAR.max(mech.radius + 0.1) as f32;
        let walk: Vec<bool> = (0..n).map(|k| stand[k] && clear[k] >= thr).collect();
        let cost: Vec<f32> = clear
            .iter()
            .map(|&c| {
                let c = f64::from(c);
                if c < COMFORT_CLEAR {
                    (1.0 + CLEAR_COST * (COMFORT_CLEAR - c)) as f32
                } else {
                    1.0
                }
            })
            .collect();
        let (comp, comp_size, main_id) = components(&walk, &surf, nx, ny, max_dh as f32);
        let main: Vec<bool> = comp
            .iter()
            .map(|&c| c == main_id && c != u32::MAX)
            .collect();

        let range = EXPOSURE_RANGE.min(vision.range.max(40.0));
        let (ex_nx, ex_ny, exposure) =
            exposure_grid(&obs, &surf, &walk, nx, ny, eye_height as f32, range);

        ApexNav {
            world,
            radius: mech.radius,
            height,
            eye_height,
            step_up,
            max_dh,
            min_x,
            min_y,
            nx,
            ny,
            surf,
            clear,
            cost,
            walk,
            main,
            comp,
            comp_size,
            main_id,
            ex_nx,
            ex_ny,
            exposure,
        }
    }

    /// The map.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// The shared handle to the map.
    pub fn world_arc(&self) -> &Arc<World> {
        &self.world
    }

    /// Index of the cell holding `(x, y)`, `None` off the grid.
    fn cell(&self, x: f64, y: f64) -> Option<usize> {
        let (i, j) = self.cell_ij(x, y)?;
        Some(j * self.nx + i)
    }

    fn cell_ij(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let fi = ((x - self.min_x) / CELL).floor();
        let fj = ((y - self.min_y) / CELL).floor();
        (fi >= 0.0 && fj >= 0.0 && (fi as usize) < self.nx && (fj as usize) < self.ny)
            .then_some((fi as usize, fj as usize))
    }

    fn center(&self, k: usize) -> [f64; 2] {
        let (i, j) = (k % self.nx, k / self.nx);
        [
            self.min_x + (i as f64 + 0.5) * CELL,
            self.min_y + (j as f64 + 0.5) * CELL,
        ]
    }

    /// True when a mech can stand at `(x, y)` at ground level.
    pub fn is_walkable_xy(&self, x: f64, y: f64) -> bool {
        self.cell(x, y).is_some_and(|k| self.walk[k])
    }

    /// True when `(x, y)` is walkable and in the main connected area.
    pub fn is_main_xy(&self, x: f64, y: f64) -> bool {
        self.cell(x, y).is_some_and(|k| self.main[k])
    }

    /// Meters from `(x, y)`'s cell to the nearest obstacle, 0 off the grid.
    pub fn clearance_at(&self, x: f64, y: f64) -> f64 {
        self.cell(x, y).map_or(0.0, |k| f64::from(self.clear[k]))
    }

    /// Terrain height at `(x, y)`.
    pub fn ground_at(&self, x: f64, y: f64) -> f64 {
        self.world.terrain.height(x, y)
    }

    /// The surface under `(x, y)` reachable from `z_max`: terrain or a roof.
    pub fn surface_at(&self, x: f64, y: f64, z_max: f64) -> f64 {
        self.world.surface_at(x, y, z_max)
    }

    /// Fraction of long sight lines open from an eye at `(x, y)`, 0..=1.
    /// Unwalkable patches and points off the map read as fully exposed.
    pub fn exposure_at(&self, x: f64, y: f64) -> f64 {
        let fi = ((x - self.min_x) / EX_CELL).floor();
        let fj = ((y - self.min_y) / EX_CELL).floor();
        if fi < 0.0 || fj < 0.0 || fi as usize >= self.ex_nx || fj as usize >= self.ex_ny {
            return 1.0;
        }
        f64::from(self.exposure[fj as usize * self.ex_nx + fi as usize]) / 255.0
    }

    /// Ground-level walk surface of `(x, y)`'s cell, or the terrain off the grid.
    fn walk_surface(&self, x: f64, y: f64) -> f64 {
        self.cell(x, y)
            .map_or_else(|| self.ground_at(x, y), |k| f64::from(self.surf[k]))
    }

    /// The walkable component `(x, y)` belongs to: its own cell's, else a main
    /// cell within 4 m, else the nearest walkable cell within 4 m. `None` when
    /// nothing walkable is that close.
    pub fn area_of(&self, x: f64, y: f64) -> Option<u32> {
        if let Some(k) = self.cell(x, y) {
            if self.walk[k] && self.comp[k] != u32::MAX {
                return Some(self.comp[k]);
            }
        }
        if self.main_id != u32::MAX
            && self
                .nearest_cell(x, y, 2.0 * CELL, |k| self.main[k])
                .is_some()
        {
            return Some(self.main_id);
        }
        let k = self.nearest_cell(x, y, 2.0 * CELL, |k| {
            self.walk[k] && self.comp[k] != u32::MAX
        })?;
        Some(self.comp[k])
    }

    /// `area_of` for a position standing at ground level; `None` when `pos` is
    /// up on a roof or otherwise well above the terrain.
    pub fn ground_area_of(&self, pos: [f64; 3]) -> Option<u32> {
        if pos[2] > self.ground_at(pos[0], pos[1]) + self.step_up + 1.0 {
            return None;
        }
        self.area_of(pos[0], pos[1])
    }

    /// True when a mech can walk straight from `a` to `b`: every cell on the
    /// line is in the main area and no step between them is too tall.
    pub fn walk_line(&self, a: [f64; 2], b: [f64; 2]) -> bool {
        self.walk_line_by(a, b, |k| self.main[k])
    }

    /// `walk_line` within component `area` instead of the main area.
    pub fn walk_line_in(&self, a: [f64; 2], b: [f64; 2], area: u32) -> bool {
        area != u32::MAX && self.walk_line_by(a, b, |k| self.comp[k] == area)
    }

    fn walk_line_by(&self, a: [f64; 2], b: [f64; 2], ok: impl Fn(usize) -> bool) -> bool {
        let steps = (dist_xy(a, b) / (0.5 * CELL)).ceil().max(1.0) as usize;
        let mut prev: Option<usize> = None;
        for s in 0..=steps {
            let t = s as f64 / steps as f64;
            let Some(k) = self.cell(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) else {
                return false;
            };
            if Some(k) == prev {
                continue;
            }
            if !ok(k) {
                return false;
            }
            if let Some(q) = prev {
                if f64::from((self.surf[k] - self.surf[q]).abs()) > self.max_dh {
                    return false;
                }
            }
            prev = Some(k);
        }
        true
    }

    /// `(x, y)` itself when it is in the main area, else the nearest main
    /// cell center within `max_r` meters.
    pub fn snap_to_main(&self, x: f64, y: f64, max_r: f64) -> Option<[f64; 2]> {
        self.snap_to_area(x, y, max_r, self.main_id)
    }

    /// `(x, y)` itself when it is in component `area`, else the nearest cell
    /// center of `area` within `max_r` meters.
    pub fn snap_to_area(&self, x: f64, y: f64, max_r: f64, area: u32) -> Option<[f64; 2]> {
        if area == u32::MAX {
            return None;
        }
        if self.cell(x, y).is_some_and(|k| self.comp[k] == area) {
            return Some([x, y]);
        }
        self.nearest_cell(x, y, max_r, |k| self.comp[k] == area)
            .map(|k| self.center(k))
    }

    /// The cell nearest `(x, y)` (by center distance, within `max_r`) passing `ok`.
    fn nearest_cell(
        &self,
        x: f64,
        y: f64,
        max_r: f64,
        ok: impl Fn(usize) -> bool,
    ) -> Option<usize> {
        let ci = ((x - self.min_x) / CELL)
            .floor()
            .clamp(0.0, (self.nx - 1) as f64) as isize;
        let cj = ((y - self.min_y) / CELL)
            .floor()
            .clamp(0.0, (self.ny - 1) as f64) as isize;
        // A point off the grid starts its rings at the nearest edge cell.
        let off = dist_xy([x, y], self.center(cj as usize * self.nx + ci as usize));
        let rings = ((max_r + off) / CELL).ceil() as isize + 1;
        let mut best: Option<(f64, usize)> = None;
        for r in 0..=rings {
            if let Some((d, _)) = best {
                if (r as f64 - 1.0) * CELL - off > d {
                    break;
                }
            }
            for dj in -r..=r {
                let edge_row = dj.abs() == r;
                let mut di = -r;
                while di <= r {
                    let (i, j) = (ci + di, cj + dj);
                    if i >= 0 && j >= 0 && (i as usize) < self.nx && (j as usize) < self.ny {
                        let k = j as usize * self.nx + i as usize;
                        if ok(k) {
                            let d = dist_xy(self.center(k), [x, y]);
                            if d <= max_r && best.is_none_or(|(bd, _)| d < bd) {
                                best = Some((d, k));
                            }
                        }
                    }
                    di += if edge_row || r == 0 { 1 } else { 2 * r };
                }
            }
        }
        best.map(|(_, k)| k)
    }

    /// Octile distance in meters between two cells.
    fn octile(&self, a: usize, b: usize) -> f32 {
        let dx = (a % self.nx).abs_diff(b % self.nx) as f32;
        let dy = (a / self.nx).abs_diff(b / self.nx) as f32;
        let (lo, hi) = if dx < dy { (dx, dy) } else { (dy, dx) };
        CELL as f32 * (hi - lo + SQRT_2 as f32 * lo)
    }

    /// A walkable route from `start` to `goal` as string-pulled waypoints,
    /// beginning at the (snapped) start and ending at the (snapped) goal.
    ///
    /// Weighted A* (heuristic weight 1.25, octile) over the cells of the
    /// start's component (`area_of`, the main area when none), no corner
    /// cutting, clearance-weighted costs. Start and goal are moved to the
    /// nearest cell of that component within 40 m. A goal outside it (or a
    /// search hitting its expansion cap) yields a route to the reachable cell
    /// nearest the goal instead.
    pub fn find_path(&self, start: [f64; 2], goal: [f64; 2]) -> Option<Vec<[f64; 2]>> {
        let area = self.area_of(start[0], start[1]).unwrap_or(self.main_id);
        let start = self.snap_to_area(start[0], start[1], SNAP_R, area)?;
        let snapped = self.snap_to_area(goal[0], goal[1], SNAP_R, area);
        let goal = snapped.unwrap_or(goal);
        if snapped.is_some() && self.walk_line_in(start, goal, area) {
            return Some(vec![start, goal]);
        }
        let s = self.cell(start[0], start[1])?;
        // A goal off the grid aims at the nearest edge cell.
        let ti = ((goal[0] - self.min_x) / CELL)
            .floor()
            .clamp(0.0, (self.nx - 1) as f64) as usize;
        let tj = ((goal[1] - self.min_y) / CELL)
            .floor()
            .clamp(0.0, (self.ny - 1) as f64) as usize;
        let t = tj * self.nx + ti;
        let nx = self.nx as isize;
        let ny = self.ny as isize;
        let (cells, reached) = SCRATCH.with(|sc| {
            let sc = &mut *sc.borrow_mut();
            sc.begin(self.main.len());
            let gen = sc.gen;
            sc.stamp[s] = gen;
            sc.g[s] = 0.0;
            sc.parent[s] = u32::MAX;
            sc.heap
                .push(Reverse(((ASTAR_W * self.octile(s, t)).to_bits(), s as u32)));
            let mut best = (self.octile(s, t), s);
            let mut expanded = 0usize;
            while let Some(Reverse((_, u))) = sc.heap.pop() {
                let u = u as usize;
                if sc.closed[u] == gen {
                    continue;
                }
                sc.closed[u] = gen;
                if u == t {
                    return (trace(sc, t), true);
                }
                let h = self.octile(u, t);
                if h < best.0 {
                    best = (h, u);
                }
                expanded += 1;
                if expanded >= MAX_EXPANSIONS {
                    break;
                }
                let (ui, uj) = ((u % self.nx) as isize, (u / self.nx) as isize);
                for &(di, dj, len) in &DIRS {
                    let (vi, vj) = (ui + di, uj + dj);
                    if vi < 0 || vj < 0 || vi >= nx || vj >= ny {
                        continue;
                    }
                    let v = (vj * nx + vi) as usize;
                    if self.comp[v] != area || sc.closed[v] == gen {
                        continue;
                    }
                    if di != 0
                        && dj != 0
                        && !(self.comp[(uj * nx + vi) as usize] == area
                            && self.comp[(vj * nx + ui) as usize] == area)
                    {
                        continue;
                    }
                    if f64::from((self.surf[v] - self.surf[u]).abs()) > self.max_dh {
                        continue;
                    }
                    let ng = sc.g[u] + CELL as f32 * len * 0.5 * (self.cost[u] + self.cost[v]);
                    if sc.stamp[v] != gen || ng < sc.g[v] {
                        sc.stamp[v] = gen;
                        sc.g[v] = ng;
                        sc.parent[v] = u as u32;
                        let f = ng + ASTAR_W * self.octile(v, t);
                        sc.heap.push(Reverse((f.to_bits(), v as u32)));
                    }
                }
            }
            (trace(sc, best.1), false)
        });
        if cells.len() < 2 {
            return None;
        }
        let mut pts: Vec<[f64; 2]> = cells.iter().map(|&k| self.center(k)).collect();
        pts[0] = start;
        if reached {
            *pts.last_mut()? = goal;
        }
        Some(self.string_pull(&pts, area))
    }

    /// Greedy line-of-walk smoothing of a cell-center path.
    fn string_pull(&self, pts: &[[f64; 2]], area: u32) -> Vec<[f64; 2]> {
        let Some(&first) = pts.first() else {
            return Vec::new();
        };
        let mut out = vec![first];
        let mut anchor = first;
        let mut k = 1;
        while k < pts.len() {
            let mut last_ok = k;
            let mut m = k + 1;
            while m < pts.len() && self.walk_line_in(anchor, pts[m], area) {
                last_ok = m;
                m += 1;
            }
            out.push(pts[last_ok]);
            anchor = pts[last_ok];
            k = last_ok + 1;
        }
        out
    }

    /// The waypoint to steer at from `pos` along `path`, starting from index
    /// `idx`: the furthest of the next few points that can be walked to in a
    /// straight line (within `pos`'s component), with its index. Falls back to `path[idx]`.
    pub fn string_pull_waypoint(
        &self,
        pos: [f64; 2],
        path: &[[f64; 2]],
        idx: usize,
    ) -> (usize, [f64; 2]) {
        let area = self.area_of(pos[0], pos[1]).unwrap_or(self.main_id);
        self.string_pull_waypoint_in(pos, path, idx, area)
    }

    /// `string_pull_waypoint` with straight walks tested inside component `area`.
    pub fn string_pull_waypoint_in(
        &self,
        pos: [f64; 2],
        path: &[[f64; 2]],
        idx: usize,
        area: u32,
    ) -> (usize, [f64; 2]) {
        if path.is_empty() {
            return (0, pos);
        }
        let mut idx = idx.min(path.len() - 1);
        // Points already reached do not count.
        while idx + 1 < path.len() && dist_xy(pos, path[idx]) < CELL {
            idx += 1;
        }
        let mut best = idx;
        for (j, &p) in path.iter().enumerate().take(idx + 8).skip(idx + 1) {
            if self.walk_line_in(pos, p, area) {
                best = j;
            } else {
                break;
            }
        }
        (best, path[best])
    }

    /// A jump plan back to the main area when standing on a roof or an island
    /// cut off from it; `None` when already on the main area at ground level,
    /// airborne, or no landing is in reach.
    pub fn find_escape(&self, pos: [f64; 3]) -> Option<EscapePlan> {
        let s0 = self.world.surface_at(pos[0], pos[1], pos[2] + 0.5);
        if pos[2] - s0 > 1.5 {
            return None;
        }
        let start = self.cell(pos[0], pos[1])?;
        let at_ground = s0 <= f64::from(self.surf[start]) + self.step_up;
        // At ground level beside a wall the disc sits in an eroded cell; a main cell close by means walk, not jump.
        if at_ground
            && (self.main[start] || self.snap_to_main(pos[0], pos[1], 2.0 * CELL).is_some())
        {
            return None;
        }
        // A big ground basin is a place to live, not a trap: only hop out when
        // the main area is close, and only from launch cells nearby.
        let big_basin = at_ground
            && self
                .area_of(pos[0], pos[1])
                .is_some_and(|c| c != u32::MAX && self.comp_size[c as usize] >= 400);
        if big_basin && self.snap_to_main(pos[0], pos[1], 28.0).is_none() {
            return None;
        }
        // BFS over the cells reachable at this level.
        let mut seen: HashMap<usize, (f64, f64)> = HashMap::new();
        let mut queue = VecDeque::new();
        seen.insert(start, (s0, 0.0));
        queue.push_back(start);
        let mut edges: Vec<(usize, f64, f64)> = Vec::new();
        while let Some(u) = queue.pop_front() {
            let (su, du) = seen[&u];
            let (ui, uj) = ((u % self.nx) as isize, (u / self.nx) as isize);
            let mut at_edge = false;
            for (di, dj) in [(1isize, 0isize), (-1, 0), (0, 1), (0, -1)] {
                let (vi, vj) = (ui + di, uj + dj);
                if vi < 0 || vj < 0 || vi as usize >= self.nx || vj as usize >= self.ny {
                    at_edge = true;
                    continue;
                }
                let v = vj as usize * self.nx + vi as usize;
                if seen.contains_key(&v) {
                    continue;
                }
                let c = self.center(v);
                let sv = self.world.surface_at(c[0], c[1], su + self.step_up + 0.05);
                // No walking up (or down) cliffs taller than a step.
                let level = sv >= su - self.step_up
                    && (sv - su).abs() <= self.max_dh
                    && !self.wall_at(c[0], c[1], sv);
                let main_level = self.main[v] && sv <= f64::from(self.surf[v]) + self.step_up;
                if level && !main_level && seen.len() < ESCAPE_MAX_CELLS {
                    seen.insert(v, (sv, du + CELL));
                    queue.push_back(v);
                } else if !level || main_level {
                    at_edge = true;
                }
            }
            if at_edge {
                edges.push((u, su, du));
            }
        }
        let mut best: Option<(f64, EscapePlan)> = None;
        for &(u, su, du) in &edges {
            let launch = self.center(u);
            if big_basin && dist_xy(launch, [pos[0], pos[1]]) > 16.0 {
                continue;
            }
            for b in 0..8 {
                let ang = b as f64 * TAU / 8.0;
                let dir = [ang.cos(), ang.sin()];
                let edge = [
                    launch[0] + dir[0] * 0.5 * CELL,
                    launch[1] + dir[1] * 0.5 * CELL,
                ];
                let Some(&land_d) = ESCAPE_LAND
                    .iter()
                    .find(|&&d| self.is_main_xy(edge[0] + dir[0] * d, edge[1] + dir[1] * d))
                else {
                    continue;
                };
                let land = [edge[0] + dir[0] * land_d, edge[1] + dir[1] * land_d];
                let need_z = self.max_obstacle(launch, land) + 2.5;
                // Out of jump reach: too high above us or the launch, or too far.
                if need_z > s0 + 28.0 || need_z > su + 28.0 || dist_xy(launch, land) > 30.0 {
                    continue;
                }
                let score = du + 1.5 * land_d + 0.3 * (need_z - su).max(0.0);
                if best.as_ref().is_none_or(|(bs, _)| score < *bs) {
                    best = Some((
                        score,
                        EscapePlan {
                            launch_xy: launch,
                            land_xy: land,
                            need_z,
                            jump_dir: dir,
                        },
                    ));
                }
            }
        }
        best.map(|(_, plan)| plan)
    }

    /// True when a prism rises from below `z + height` to above `z + step_up` at `(x, y)`.
    fn wall_at(&self, x: f64, y: f64, z: f64) -> bool {
        self.world.prisms_at(x, y).any(|(_, p)| {
            p.top > z + self.step_up && p.base < z + self.height - 0.1 && p.contains_xy(x, y)
        })
    }

    /// Highest terrain or prism top on the segment `a -> b`, sampled every meter.
    fn max_obstacle(&self, a: [f64; 2], b: [f64; 2]) -> f64 {
        let steps = dist_xy(a, b).ceil().max(1.0) as usize;
        (0..=steps)
            .map(|s| {
                let t = s as f64 / steps as f64;
                self.world.surface_at(
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    f64::INFINITY,
                )
            })
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// The best spot among `rings` x `bearings` around `me` hidden from the
    /// nearest four threats, scored by distance minus a quarter of the
    /// distance to the closest threat.
    fn cover_search(
        &self,
        me: [f64; 3],
        threats: &[ThreatInfo],
        zone: &ZoneObs,
        require_trees: bool,
        rings: &[f64],
        bearings: &[f64],
    ) -> Option<[f64; 3]> {
        let ts = nearest_threats(me, threats, 4);
        if ts.is_empty() {
            return None;
        }
        let eyes: Vec<[f64; 3]> = ts
            .iter()
            .map(|t| [t.pos[0], t.pos[1], t.pos[2] + self.eye_height])
            .collect();
        let mut best: Option<(f64, [f64; 3])> = None;
        for &r in rings {
            for &ang in bearings {
                let (x, y) = (me[0] + r * ang.cos(), me[1] + r * ang.sin());
                if !self.is_main_xy(x, y) || !in_zone(zone, x, y) {
                    continue;
                }
                let min_threat = ts
                    .iter()
                    .map(|t| (t.pos[0] - x).hypot(t.pos[1] - y))
                    .fold(f64::INFINITY, f64::min);
                let score = r - 0.25 * min_threat;
                if best.is_some_and(|(bs, _)| score >= bs) {
                    continue;
                }
                let z = self.walk_surface(x, y);
                let targets = [[x, y, z + 0.5 * self.height], [x, y, z + 0.9 * self.height]];
                let hidden = eyes.iter().all(|&e| {
                    targets.iter().all(|&p| {
                        if require_trees {
                            !self.world.sight(e, p)
                        } else {
                            !self.world.los(e, p)
                        }
                    })
                });
                if hidden {
                    best = Some((score, [x, y, z]));
                }
            }
        }
        best.map(|(_, p)| p)
    }

    /// A walkable spot within `max_r` that the nearest threats cannot see:
    /// rings at 8, 16, 26, 38, and 52 m by 16 bearings. With `require_trees`
    /// the test is full sight (canopy counts as cover), otherwise solid line of sight.
    pub fn find_cover(
        &self,
        me_pos: [f64; 3],
        threats: &[ThreatInfo],
        zone: &ZoneObs,
        require_trees: bool,
        max_r: f64,
    ) -> Option<[f64; 3]> {
        let rings: Vec<f64> = [8.0, 16.0, 26.0, 38.0, 52.0]
            .into_iter()
            .filter(|&r| r <= max_r)
            .collect();
        let bearings: Vec<f64> = (0..16).map(|b| b as f64 * TAU / 16.0).collect();
        self.cover_search(me_pos, threats, zone, require_trees, &rings, &bearings)
    }

    /// `find_cover` on a small fan: 8, 15, and 24 m by 8 bearings starting
    /// from the bearing away from the nearest threat.
    pub fn find_commit_cover(
        &self,
        me_pos: [f64; 3],
        threats: &[ThreatInfo],
        zone: &ZoneObs,
        require_trees: bool,
    ) -> Option<[f64; 3]> {
        let nearest = nearest_threats(me_pos, threats, 1);
        let t = nearest.first()?;
        let away = (me_pos[1] - t.pos[1]).atan2(me_pos[0] - t.pos[0]);
        // Away first, then alternating sides of it.
        let bearings: Vec<f64> = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0]
            .iter()
            .map(|k| away + k * TAU / 8.0)
            .collect();
        self.cover_search(
            me_pos,
            threats,
            zone,
            require_trees,
            &[8.0, 15.0, 24.0],
            &bearings,
        )
    }

    /// True when `find_commit_cover` finds a spot.
    pub fn has_cover_nearby(
        &self,
        me_pos: [f64; 3],
        threats: &[ThreatInfo],
        zone: &ZoneObs,
        require_trees: bool,
    ) -> bool {
        self.find_commit_cover(me_pos, threats, zone, require_trees)
            .is_some()
    }

    /// A low-exposure spot to hold, sampled in a band at `ring_frac` of the
    /// next zone radius from its center (the current one when the next is unknown). Cost is
    /// `exp_w * exposure + crowd_pen * crowd` plus a small distance term, where
    /// crowd counts nearby threats and supply drops.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_home_spot(
        &self,
        me_pos: [f64; 3],
        zone: &ZoneObs,
        ring_frac: f64,
        exp_w: f64,
        crowd_pen: f64,
        threats: &[ThreatInfo],
        drops: &[DropObs],
        rng: &mut Rng64,
    ) -> Option<[f64; 3]> {
        let (center, radius) = if zone.next_radius > 0.0 {
            (zone.next_center, zone.next_radius)
        } else if zone.radius > 0.0 {
            (zone.center, zone.radius)
        } else {
            ([me_pos[0], me_pos[1]], 60.0)
        };
        let reach = (ring_frac.clamp(0.05, 1.0) * radius).max(CELL);
        let me = [me_pos[0], me_pos[1]];
        let area = self.area_of(me[0], me[1]).unwrap_or(self.main_id);
        let in_area = |p: [f64; 2]| {
            if area == self.main_id {
                self.is_main_xy(p[0], p[1])
            } else {
                self.cell(p[0], p[1]).is_some_and(|k| self.comp[k] == area)
            }
        };
        let cost = |p: [f64; 2]| {
            let crowd: f64 = threats
                .iter()
                .map(|t| (1.0 - (t.pos[0] - p[0]).hypot(t.pos[1] - p[1]) / 100.0).max(0.0))
                .sum::<f64>()
                + drops
                    .iter()
                    .map(|d| 0.5 * (1.0 - (d.pos[0] - p[0]).hypot(d.pos[1] - p[1]) / 80.0).max(0.0))
                    .sum::<f64>();
            exp_w * self.exposure_at(p[0], p[1]) + crowd_pen * crowd + 0.02 * dist_xy(me, p)
        };
        let mut best: Option<(f64, [f64; 2])> = None;
        for _ in 0..40 {
            let ang = rng.range(0.0, TAU);
            // A band around the hold distance, not the whole disc.
            let r = (reach + rng.range(-0.15, 0.15) * radius).max(0.0);
            let p = [center[0] + r * ang.cos(), center[1] + r * ang.sin()];
            if !in_area(p) || !in_zone(zone, p[0], p[1]) {
                continue;
            }
            let c = cost(p);
            if best.is_none_or(|(bc, _)| c < bc) {
                best = Some((c, p));
            }
        }
        let p = match best {
            Some((_, p)) => p,
            None => self
                .snap_to_area(center[0], center[1], reach.max(SNAP_R), area)
                .or_else(|| self.snap_to_main(center[0], center[1], reach.max(SNAP_R)))?,
        };
        Some([p[0], p[1], self.walk_surface(p[0], p[1])])
    }

    /// True when walking from `pos` along `dir` for `look_m` meters runs into a
    /// wall or a step taller than `step_up`, tested across the mech's width.
    pub fn step_blocked_ahead(&self, pos: [f64; 3], dir: [f64; 2], look_m: f64) -> bool {
        let Some(d) = unit(dir) else {
            return false;
        };
        let side = [-d[1], d[0]];
        let mut cur = self.world.surface_at(pos[0], pos[1], pos[2] + 0.5);
        let reach = self.radius + look_m.max(0.0);
        let steps = (reach / 0.5).ceil().max(1.0) as usize;
        for s in 1..=steps {
            let along = reach * s as f64 / steps as f64;
            let c = [pos[0] + d[0] * along, pos[1] + d[1] * along];
            for lat in [-0.8, 0.0, 0.8] {
                let off = lat * self.radius;
                let (x, y) = (c[0] + side[0] * off, c[1] + side[1] * off);
                // Only the leading part of the disc can bump into something.
                if along < self.radius && lat != 0.0 {
                    continue;
                }
                if self.world.terrain.height(x, y) > cur + self.step_up || self.wall_at(x, y, cur) {
                    return true;
                }
            }
            cur = self.world.surface_at(c[0], c[1], cur + self.step_up + 0.05);
        }
        false
    }
}

/// The cells from the search start to `end`, following the parent links.
fn trace(sc: &Scratch, end: usize) -> Vec<usize> {
    let mut cells = vec![end];
    let mut k = end;
    while sc.parent[k] != u32::MAX && cells.len() <= sc.parent.len() {
        k = sc.parent[k] as usize;
        cells.push(k);
    }
    cells.reverse();
    cells
}

/// Distance in meters from each free cell to the nearest non-free cell (or
/// the grid edge), minus half a cell: a two-pass 3x3 chamfer transform.
fn chamfer_clearance(free: &[bool], nx: usize, ny: usize) -> Vec<f32> {
    let (a, d) = (CELL as f32, (CELL * SQRT_2) as f32);
    let mut dist: Vec<f32> = free
        .iter()
        .map(|&f| if f { f32::INFINITY } else { 0.0 })
        .collect();
    let at = |dist: &[f32], i: isize, j: isize| -> f32 {
        if i < 0 || j < 0 || i as usize >= nx || j as usize >= ny {
            0.0
        } else {
            dist[j as usize * nx + i as usize]
        }
    };
    for j in 0..ny as isize {
        for i in 0..nx as isize {
            let k = j as usize * nx + i as usize;
            if dist[k] == 0.0 {
                continue;
            }
            let v = dist[k]
                .min(at(&dist, i - 1, j) + a)
                .min(at(&dist, i - 1, j - 1) + d)
                .min(at(&dist, i, j - 1) + a)
                .min(at(&dist, i + 1, j - 1) + d);
            dist[k] = v;
        }
    }
    for j in (0..ny as isize).rev() {
        for i in (0..nx as isize).rev() {
            let k = j as usize * nx + i as usize;
            if dist[k] == 0.0 {
                continue;
            }
            let v = dist[k]
                .min(at(&dist, i + 1, j) + a)
                .min(at(&dist, i + 1, j + 1) + d)
                .min(at(&dist, i, j + 1) + a)
                .min(at(&dist, i - 1, j + 1) + d);
            dist[k] = v;
        }
    }
    let half = 0.5 * CELL as f32;
    dist.iter()
        .map(|&v| if v > 0.0 { (v - half).max(0.0) } else { 0.0 })
        .collect()
}

/// Labels the 4-connected walkable components (steps up to `max_dh`):
/// `(comp, comp_size, main_id)` with `comp` the per-cell id (`u32::MAX` where
/// not walkable), `comp_size` the cell count per id, and `main_id` the largest.
fn components(
    walk: &[bool],
    surf: &[f32],
    nx: usize,
    ny: usize,
    max_dh: f32,
) -> (Vec<u32>, Vec<u32>, u32) {
    let n = walk.len();
    let mut label = vec![u32::MAX; n];
    let mut best = (0usize, u32::MAX);
    let mut stack = Vec::new();
    let mut sizes: Vec<u32> = Vec::new();
    for seed in 0..n {
        if !walk[seed] || label[seed] != u32::MAX {
            continue;
        }
        let id = sizes.len() as u32;
        label[seed] = id;
        stack.push(seed);
        let mut size = 0usize;
        while let Some(u) = stack.pop() {
            size += 1;
            let (i, j) = (u % nx, u / nx);
            let mut visit = |v: usize| {
                if walk[v] && label[v] == u32::MAX && (surf[v] - surf[u]).abs() <= max_dh {
                    label[v] = id;
                    stack.push(v);
                }
            };
            if i > 0 {
                visit(u - 1);
            }
            if i + 1 < nx {
                visit(u + 1);
            }
            if j > 0 {
                visit(u - nx);
            }
            if j + 1 < ny {
                visit(u + nx);
            }
        }
        sizes.push(size as u32);
        if size > best.0 {
            best = (size, id);
        }
    }
    (label, sizes, best.1)
}

/// The 4 m exposure grid: for each patch with a walkable cell, the fraction
/// of `EXPOSURE_RAYS` horizontal eye-level rays out to `range` meters that
/// meet nothing taller than the eye on the 2 m obstruction raster.
fn exposure_grid(
    obs: &[f32],
    surf: &[f32],
    walk: &[bool],
    nx: usize,
    ny: usize,
    eye_height: f32,
    range: f64,
) -> (usize, usize, Vec<u8>) {
    let steps = (range / CELL).floor().max(1.0) as isize;
    let pad = steps as usize + 1;
    let pnx = nx + 2 * pad;
    let pny = ny + 2 * pad;
    // Padded so every ray offset stays in bounds; the margin blocks nothing.
    let mut raster = vec![f32::NEG_INFINITY; pnx * pny];
    for j in 0..ny {
        let row = (j + pad) * pnx + pad;
        raster[row..row + nx].copy_from_slice(&obs[j * nx..(j + 1) * nx]);
    }
    let rays: Vec<Vec<isize>> = (0..EXPOSURE_RAYS)
        .map(|r| {
            let ang = r as f64 * TAU / EXPOSURE_RAYS as f64;
            let (c, s) = (ang.cos(), ang.sin());
            let mut offs = Vec::with_capacity(steps as usize);
            let mut last = (0isize, 0isize);
            for k in 1..=steps {
                let di = (k as f64 * c).round() as isize;
                let dj = (k as f64 * s).round() as isize;
                if (di, dj) != last {
                    offs.push(dj * pnx as isize + di);
                    last = (di, dj);
                }
            }
            offs
        })
        .collect();
    let ex_nx = nx.div_ceil(2);
    let ex_ny = ny.div_ceil(2);
    let mut exposure = vec![255u8; ex_nx * ex_ny];
    for ej in 0..ex_ny {
        for ei in 0..ex_nx {
            // The first walkable cell of the patch is its eye position.
            let origin = [(0, 0), (1, 0), (0, 1), (1, 1)]
                .iter()
                .find_map(|&(oi, oj)| {
                    let (i, j) = (2 * ei + oi, 2 * ej + oj);
                    (i < nx && j < ny && walk[j * nx + i]).then_some((i, j))
                });
            let Some((i, j)) = origin else {
                continue;
            };
            let eye = surf[j * nx + i] + eye_height;
            let base = ((j + pad) * pnx + i + pad) as isize;
            let open = rays
                .iter()
                .filter(|offs| offs.iter().all(|&o| raster[(base + o) as usize] <= eye))
                .count();
            exposure[ej * ex_nx + ei] = ((open * 255 + EXPOSURE_RAYS / 2) / EXPOSURE_RAYS) as u8;
        }
    }
    (ex_nx, ex_ny, exposure)
}

#[cfg(test)]
mod tests {
    use super::*;
    use world::{Prism, PrismKind};

    fn mech() -> MechStats {
        MechStats::default()
    }

    fn nav_of(world: World) -> ApexNav {
        ApexNav::build(Arc::new(world), &mech(), &VisionCfg::default())
    }

    fn threat(pos: [f64; 3]) -> ThreatInfo {
        ThreatInfo {
            id: 7,
            pos,
            vel: [0.0; 3],
            aim: [0.0; 2],
            hp: 1400.0,
            max_hp: 1400.0,
            cubes: 0,
            weapons: [None, None],
            charging: false,
            dist: 0.0,
            los: true,
            visible_now: true,
            last_seen_t: 0.0,
            last_hit_us_t: f64::NEG_INFINITY,
            last_beam_t: f64::NEG_INFINITY,
            is_ghost: false,
            is_aware: true,
            threat_dps: 100.0,
            target_score: 0.0,
        }
    }

    /// Flat 200 m square with a 4 m thick, 120 m long, 20 m tall wall along x = 0.
    fn walled() -> ApexNav {
        let mut w = World::flat(100.0);
        w.prisms.push(Prism::rect(
            PrismKind::Building,
            [-2.0, -60.0],
            [2.0, 60.0],
            0.0,
            20.0,
        ));
        nav_of(w)
    }

    fn path_len(start: [f64; 2], path: &[[f64; 2]]) -> f64 {
        let mut len = dist_xy(start, path[0]);
        for w in path.windows(2) {
            len += dist_xy(w[0], w[1]);
        }
        len
    }

    #[test]
    fn flat_world_is_open_and_paths_are_straight() {
        let nav = nav_of(World::flat(100.0));
        assert!(nav.is_walkable_xy(0.0, 0.0) && nav.is_main_xy(10.0, -20.0));
        assert!(
            !nav.is_walkable_xy(99.5, 0.0),
            "the map edge counts as an obstacle"
        );
        assert!(nav.clearance_at(0.0, 0.0) > 50.0);
        assert!((nav.exposure_at(0.0, 0.0) - 1.0).abs() < 1e-9);
        assert_eq!(nav.ground_at(3.0, 4.0), 0.0);
        let path = nav.find_path([-50.0, -50.0], [50.0, 40.0]).unwrap();
        assert_eq!(path, vec![[-50.0, -50.0], [50.0, 40.0]]);
        assert!(nav.walk_line([-50.0, 0.0], [50.0, 0.0]));
        assert!(!nav.step_blocked_ahead([0.0, 0.0, 0.0], [1.0, 0.0], 5.0));
        assert!(nav.find_escape([0.0, 0.0, 0.0]).is_none());
        assert_eq!(
            nav.snap_to_main(150.0, 0.0, 60.0).map(|p| p[1].abs() < 2.0),
            Some(true)
        );
        assert!(nav.snap_to_main(150.0, 0.0, 10.0).is_none());
    }

    #[test]
    fn walls_block_walking_and_paths_go_around() {
        let nav = walled();
        assert!(!nav.is_walkable_xy(0.0, 0.0));
        assert!(
            !nav.is_walkable_xy(3.0, 0.0),
            "too close to the wall for the mech's disc"
        );
        assert!(nav.is_main_xy(-10.0, 0.0) && nav.is_main_xy(10.0, 0.0));
        assert!(!nav.walk_line([-20.0, 0.0], [20.0, 0.0]));
        let start = [-20.0, 0.0];
        let path = nav.find_path(start, [20.0, 0.0]).unwrap();
        assert_eq!(*path.last().unwrap(), [20.0, 0.0]);
        for w in path.windows(2) {
            assert!(
                nav.walk_line(w[0], w[1]),
                "leg {:?} -> {:?} is walkable",
                w[0],
                w[1]
            );
        }
        assert!(
            path.iter().any(|p| p[1].abs() > 60.0),
            "the route rounds an end of the wall"
        );
        assert!(path_len(start, &path) > 120.0);
        assert!(
            path.len() < 12,
            "string pulling leaves few waypoints, got {}",
            path.len()
        );
        let (i, wp) = nav.string_pull_waypoint(start, &path, 0);
        assert!(i >= 1 && nav.walk_line(start, wp));
        assert!(nav.step_blocked_ahead([-6.0, 0.0, 0.0], [1.0, 0.0], 3.0));
        assert!(!nav.step_blocked_ahead([-6.0, 0.0, 0.0], [-1.0, 0.0], 3.0));
    }

    #[test]
    fn cover_hides_behind_the_wall() {
        let nav = walled();
        let zone = ZoneObs::default();
        let foe = threat([40.0, 0.0, 0.0]);
        let me = [-6.0, 10.0, 0.0];
        let spot = nav
            .find_cover(me, &[foe], &zone, false, 60.0)
            .expect("cover west of the wall");
        assert!(spot[0] < -2.0 && spot[1].abs() < 55.0, "spot {spot:?}");
        assert!(!nav
            .world()
            .los([40.0, 0.0, 8.0], [spot[0], spot[1], spot[2] + 4.5]));
        assert!(nav.has_cover_nearby(me, &[foe], &zone, false));
        let commit = nav.find_commit_cover(me, &[foe], &zone, false).unwrap();
        assert!(commit[0] < -2.0);
        assert!(nav.find_cover(me, &[], &zone, false, 60.0).is_none());
        // In the open nothing hides.
        let open = nav_of(World::flat(100.0));
        assert!(open.find_cover(me, &[foe], &zone, false, 60.0).is_none());
        assert!(!open.has_cover_nearby(me, &[foe], &zone, false));
        // A zone that excludes everything west of the wall leaves no cover.
        let tight = ZoneObs {
            center: [40.0, 0.0],
            radius: 40.0,
            ..ZoneObs::default()
        };
        assert!(nav.find_cover(me, &[foe], &tight, false, 60.0).is_none());
    }

    #[test]
    fn exposure_drops_next_to_tall_walls() {
        let nav = walled();
        let open = nav.exposure_at(-60.0, 80.0);
        let sheltered = nav.exposure_at(-6.0, 0.0);
        assert!(sheltered < open, "sheltered {sheltered} open {open}");
        assert!(sheltered <= 0.7);
    }

    #[test]
    fn bridges_stay_passable_underneath() {
        let mut w = World::flat(100.0);
        w.prisms.push(Prism::rect(
            PrismKind::Bridge,
            [-30.0, -5.0],
            [30.0, 5.0],
            15.0,
            16.0,
        ));
        let nav = nav_of(w);
        assert!(nav.is_main_xy(0.0, 0.0));
        assert!(nav.walk_line([0.0, -40.0], [0.0, 40.0]));
    }

    #[test]
    fn escape_off_a_roof_lands_on_the_main_area() {
        let mut w = World::flat(100.0);
        w.prisms.push(Prism::rect(
            PrismKind::Building,
            [-10.0, -10.0],
            [10.0, 10.0],
            0.0,
            12.0,
        ));
        let nav = nav_of(w);
        let plan = nav
            .find_escape([0.0, 0.0, 12.0])
            .expect("a way off the roof");
        assert!(nav.is_main_xy(plan.land_xy[0], plan.land_xy[1]));
        assert!(plan.need_z >= 14.5);
        assert!(plan.launch_xy[0].abs() <= 10.0 && plan.launch_xy[1].abs() <= 10.0);
        let to_land = unit([
            plan.land_xy[0] - plan.launch_xy[0],
            plan.land_xy[1] - plan.launch_xy[1],
        ])
        .unwrap();
        assert!(
            (to_land[0] - plan.jump_dir[0]).abs() < 1e-9
                && (to_land[1] - plan.jump_dir[1]).abs() < 1e-9
        );
        assert!(
            nav.find_escape([30.0, 30.0, 0.0]).is_none(),
            "already on the main area"
        );
        assert!(nav.find_escape([0.0, 0.0, 30.0]).is_none(), "airborne");
        assert!(
            nav.find_escape([-12.6, 0.0, 0.0]).is_none(),
            "hugging the wall at ground level"
        );
        assert!(
            nav.find_escape([-98.0, 0.0, 0.0]).is_none(),
            "at the map edge"
        );
    }

    /// A 20 m courtyard walled in on all sides, `wall_h` meters tall.
    fn courtyard(wall_h: f64) -> ApexNav {
        let mut w = World::flat(100.0);
        for (a, b) in [
            ([-14.0, -14.0], [14.0, -10.0]),
            ([-14.0, 10.0], [14.0, 14.0]),
            ([-14.0, -10.0], [-10.0, 10.0]),
            ([10.0, -10.0], [14.0, 10.0]),
        ] {
            w.prisms
                .push(Prism::rect(PrismKind::Building, a, b, 0.0, wall_h));
        }
        nav_of(w)
    }

    #[test]
    fn escape_rejects_cliffs() {
        // Low walls: hop over. Tall walls: out of jump reach, no plan.
        let low = courtyard(8.0);
        let me = [0.0, 0.0, 0.0];
        let a = low.ground_area_of(me).expect("inside the courtyard");
        assert_ne!(a, low.main_id);
        let plan = low.find_escape(me).expect("hop the low wall");
        assert!(low.is_main_xy(plan.land_xy[0], plan.land_xy[1]));
        assert!(plan.need_z <= 28.0 && dist_xy(plan.launch_xy, plan.land_xy) <= 30.0);
        assert!(
            courtyard(40.0).find_escape(me).is_none(),
            "a 40 m wall is no jump"
        );
        // Paths stay inside the courtyard and end at its edge nearest the goal.
        let path = low
            .find_path([0.0, 0.0], [60.0, 0.0])
            .expect("a path toward the goal");
        let end = *path.last().unwrap();
        assert!(end[0] > 4.0, "ends at the near edge, got {end:?}");
        for w in path.windows(2) {
            assert!(low.walk_line_in(w[0], w[1], a));
        }
        assert_eq!(
            low.ground_area_of([0.0, 0.0, 9.0]),
            None,
            "on the roof is not the ground area"
        );
    }

    #[test]
    fn home_spots_prefer_shelter_inside_the_ring() {
        let nav = walled();
        let zone = ZoneObs {
            center: [0.0, 0.0],
            radius: 90.0,
            next_center: [0.0, 0.0],
            next_radius: 60.0,
            ..ZoneObs::default()
        };
        let mut rng = Rng64::new(1, 2);
        let me = [-30.0, 0.0, 0.0];
        let mut exp_sum = 0.0;
        for _ in 0..10 {
            let p = nav
                .sample_home_spot(me, &zone, 0.6, 100.0, 40.0, &[], &[], &mut rng)
                .unwrap();
            assert!(nav.is_main_xy(p[0], p[1]));
            let r = dist_xy([p[0], p[1]], [0.0, 0.0]);
            assert!(
                (27.0 - 1e-9..=45.0 + 1e-9).contains(&r),
                "hold distance {r}"
            );
            exp_sum += nav.exposure_at(p[0], p[1]);
        }
        let open = nav.exposure_at(-60.0, 80.0);
        assert!(
            exp_sum / 10.0 < open,
            "mean exposure {} vs open {open}",
            exp_sum / 10.0
        );
    }

    #[test]
    fn shared_caches_and_falls_back_to_flat() {
        let a = ApexNav::shared("/nonexistent-arena-for-nav", &mech(), &VisionCfg::default());
        let b = ApexNav::shared("/nonexistent-arena-for-nav", &mech(), &VisionCfg::default());
        assert!(Arc::ptr_eq(&a, &b));
        assert!(a.world().prisms.is_empty() && a.is_main_xy(0.0, 0.0));
    }
}
