
mod canopy;
mod ground;
mod index;
mod load;
mod prism;
mod ray;
mod terrain;

pub use canopy::{CanopyRules, Tree};
pub use load::WorldError;
pub use prism::{Prism, PrismKind};
pub use ray::{RayHit, Surface, NO_HINT};
pub use terrain::Terrain;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

pub type Vec3 = [f64; 3];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

impl Bounds {
    pub fn contains(&self, x: f64, y: f64, margin: f64) -> bool {
        x >= self.min_x + margin && x <= self.max_x - margin && y >= self.min_y + margin && y <= self.max_y - margin
    }
}

#[derive(Clone, Debug)]
pub struct World {
    pub dir: Option<PathBuf>,
    pub bounds: Bounds,
    pub terrain: Terrain,
    pub prisms: Vec<Prism>,
    pub trees: Vec<Tree>,
    pub canopy: CanopyRules,
    grid: OnceLock<index::Grid>,
}

impl World {
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
