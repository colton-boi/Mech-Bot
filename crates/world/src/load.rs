use crate::{Bounds, CanopyRules, Prism, PrismKind, Terrain, Tree, World};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum WorldError {
    Io(PathBuf, std::io::Error),
    Json(PathBuf, serde_json::Error),
    Format(String),
}

impl std::fmt::Display for WorldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(path, e) => write!(f, "{}: {e}", path.display()),
            Self::Json(path, e) => write!(f, "{}: {e}", path.display()),
            Self::Format(what) => write!(f, "map format: {what}"),
        }
    }
}

impl std::error::Error for WorldError {}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawWorld {
    bounds: RawBounds,
    rules: RawRules,
    terrain: RawTerrain,
    prisms: Vec<RawPrism>,
    trees: RawTable,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawBounds {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawRules {
    canopy_density: f64,
    canopy_block_t: f64,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RawTerrain {
    file: String,
    min_x: f64,
    min_y: f64,
    step: f64,
    nx: usize,
    ny: usize,
    scale: f64,
}

#[derive(Deserialize)]
struct RawPrism {
    kind: PrismKind,
    base: f64,
    top: f64,
    ring: Vec<f64>,
    #[serde(default)]
    holes: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct RawTable {
    fields: Vec<String>,
    data: Vec<f64>,
}

fn points(flat: &[f64]) -> Vec<[f64; 2]> {
    flat.as_chunks::<2>()
        .0
        .iter()
        .map(|p| [p[0], p[1]])
        .collect()
}

impl World {
    /// Reads `v2/world.json` and `v2/terrain.i16` under `dir` (`hello.map.dir`).
    pub fn load(dir: &Path) -> Result<World, WorldError> {
        let json_path = dir.join("v2/world.json");
        let text = std::fs::read(&json_path).map_err(|e| WorldError::Io(json_path.clone(), e))?;
        let raw: RawWorld =
            serde_json::from_slice(&text).map_err(|e| WorldError::Json(json_path, e))?;

        let terrain_path = dir.join("v2").join(&raw.terrain.file);
        let bytes = std::fs::read(&terrain_path).map_err(|e| WorldError::Io(terrain_path, e))?;
        let t = &raw.terrain;
        if bytes.len() != t.nx * t.ny * 2 || t.nx < 2 || t.ny < 2 {
            return Err(WorldError::Format(format!(
                "terrain holds {} bytes, want {} x {} int16",
                bytes.len(),
                t.nx,
                t.ny
            )));
        }
        let heights = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| (f64::from(i16::from_le_bytes([b[0], b[1]])) * t.scale) as f32)
            .collect();
        let terrain = Terrain::new(t.min_x, t.min_y, t.step, t.nx, t.ny, heights);

        let prisms = raw
            .prisms
            .iter()
            .map(|p| {
                Prism::new(
                    p.kind,
                    p.base,
                    p.top,
                    points(&p.ring),
                    p.holes.iter().map(|h| points(h)).collect(),
                )
            })
            .collect();

        let column = |name: &str| {
            raw.trees
                .fields
                .iter()
                .position(|f| f == name)
                .ok_or_else(|| WorldError::Format(format!("trees lack column {name}")))
        };
        let (x, y, z, height) = (column("x")?, column("y")?, column("z")?, column("height")?);
        let (crown_r, crown_base, trunk_r) =
            (column("crownR")?, column("crownBase")?, column("trunkR")?);
        let (crown_x, crown_y) = (column("crownX")?, column("crownY")?);
        let trees = raw
            .trees
            .data
            .chunks_exact(raw.trees.fields.len())
            .map(|row| Tree {
                x: row[x],
                y: row[y],
                z: row[z],
                height: row[height],
                crown_r: row[crown_r],
                crown_base: row[crown_base],
                trunk_r: row[trunk_r],
                crown_x: row[crown_x],
                crown_y: row[crown_y],
            })
            .collect();

        Ok(World {
            dir: Some(dir.to_owned()),
            bounds: Bounds {
                min_x: raw.bounds.min_x,
                max_x: raw.bounds.max_x,
                min_y: raw.bounds.min_y,
                max_y: raw.bounds.max_y,
            },
            terrain,
            prisms,
            trees,
            canopy: CanopyRules {
                density: raw.rules.canopy_density,
                block_t: raw.rules.canopy_block_t,
            },
            grid: std::sync::OnceLock::new(),
        })
    }
}
