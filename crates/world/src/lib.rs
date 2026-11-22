//! The static map: terrain, prisms (buildings, stadium, bridges), and trees.
//!
//! The server simulates exactly `v2/world.json` and `v2/terrain.i16`
//! (protocol.md §11). The map is the same every match, so one [`World`] is
//! loaded per process and shared by reference.
//!
//! Queries, by file:
//! - `terrain.rs`: ground height, bilinear.
//! - `prism.rs`: point-in-prism and footprint tests.
//! - `ground.rs`: the surface a cylinder stands on, and ceilings above it.
//! - `ray.rs`: segment against terrain and prisms, for shots and line of sight.
//! - `canopy.rs`: trunks and crowns, which block vision only.
//! - `index.rs`: the grid that lets `ray.rs` and `canopy.rs` skip far objects.

mod canopy;
mod ground;
mod index;
mod load;
mod prism;
mod ray;
mod terrain;

pub use canopy::{CanopyRules, Tree};
pub use index::PrismsAt;
pub use load::WorldError;
pub use prism::{Prism, PrismKind};
pub use ray::{RayHit, Surface, NO_HINT};
pub use terrain::Terrain;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// `[x east, y north, z up]` in meters.
pub type Vec3 = [f64; 3];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl Bounds {
    /// True when `(x, y)` is at least `margin` meters inside the map edge.
    pub fn contains(&self, x: f64, y: f64, margin: f64) -> bool {
        x >= self.min_x + margin && x <= self.max_x - margin && y >= self.min_y + margin && y <= self.max_y - margin
    }
}

#[derive(Clone, Debug)]
pub struct World {
    /// Where it was loaded from, `None` for a synthetic world.
    pub dir: Option<PathBuf>,
    pub bounds: Bounds,
    pub terrain: Terrain,
    pub prisms: Vec<Prism>,
    pub trees: Vec<Tree>,
    pub canopy: CanopyRules,
    /// Prisms and trees by cell, built on first use (`index.rs`).
    grid: OnceLock<index::Grid>,
}

impl World {
    /// A flat, empty square of ground at z = 0, `half` meters from the origin to each edge.
    pub fn flat(half: f64) -> World {
        World {
            dir: None,
            bounds: Bounds {
                min_x: -half,
                max_x: half,
                min_y: -half,
                max_y: half,
            },
            terrain: Terrain::flat(-half, -half, 2.0 * half, 0.0),
            prisms: Vec::new(),
            trees: Vec::new(),
            canopy: CanopyRules { density: 0.35, block_t: 0.25 },
            grid: OnceLock::new(),
        }
    }

    /// Loads the map once per process and directory. A policy calls this from
    /// `on_hello`, so sixteen simulated mechs share one copy.
    pub fn shared(dir: &Path) -> Result<Arc<World>, WorldError> {
        static CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<World>>>> = OnceLock::new();
        let mut cache = CACHE.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
        if let Some(world) = cache.get(dir) {
            return Ok(world.clone());
        }
        let world = Arc::new(World::load(dir)?);
        cache.insert(dir.to_owned(), world.clone());
        Ok(world)
    }
}
