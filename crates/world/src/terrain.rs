#[derive(Clone, Debug)]
pub struct Terrain {
    pub min_x: f64,
    pub min_y: f64,
    pub step: f64,
    pub nx: usize,
    pub ny: usize,
    heights: Vec<f32>,
    pub max_height: f64,
    tile_max: Vec<f32>,
    tnx: usize,
}

const TILE: usize = 16;

impl Terrain {
    pub fn new(min_x: f64, min_y: f64, step: f64, nx: usize, ny: usize, heights: Vec<f32>) -> Terrain {
        assert!(
            nx >= 2 && ny >= 2 && heights.len() == nx * ny,
            "terrain grid is {nx} x {ny} with {} heights",
            heights.len()
        );
        let max_height = heights.iter().copied().fold(f32::MIN, f32::max).into();
        let (tnx, tny) = ((nx - 2) / TILE + 1, (ny - 2) / TILE + 1);
        let mut tile_max = vec![f32::MIN; tnx * tny];
        for tj in 0..tny {
            for ti in 0..tnx {
                let mut m = f32::MIN;
                for j in TILE * tj..=(TILE * (tj + 1)).min(ny - 1) {
                    for i in TILE * ti..=(TILE * (ti + 1)).min(nx - 1) {
                        m = m.max(heights[j * nx + i]);
                    }
                }
                tile_max[tj * tnx + ti] = m;
            }
        }
        Terrain {
            min_x,
            min_y,
            step,
            nx,
            ny,
            heights,
            max_height,
            tile_max,
            tnx,
        }
    }

    pub fn flat(min_x: f64, min_y: f64, size: f64, z: f32) -> Terrain {
        Terrain::new(min_x, min_y, size, 2, 2, vec![z; 4])
    }

    fn cell(&self, x: f64, y: f64) -> (usize, usize) {
        let fx = ((x - self.min_x) / self.step).clamp(0.0, (self.nx - 1) as f64);
        let fy = ((y - self.min_y) / self.step).clamp(0.0, (self.ny - 1) as f64);
        ((fx as usize).min(self.nx - 2), (fy as usize).min(self.ny - 2))
    }

    pub fn max_in(&self, x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
        let (i0, j0) = self.cell(x0.min(x1), y0.min(y1));
        let (i1, j1) = self.cell(x0.max(x1), y0.max(y1));
        let mut m = f32::MIN;
        for tj in j0 / TILE..=j1 / TILE {
            for ti in i0 / TILE..=i1 / TILE {
                m = m.max(self.tile_max[tj * self.tnx + ti]);
            }
        }
        f64::from(m)
    }

    pub fn height(&self, x: f64, y: f64) -> f64 {
        let fx = ((x - self.min_x) / self.step).clamp(0.0, (self.nx - 1) as f64);
        let fy = ((y - self.min_y) / self.step).clamp(0.0, (self.ny - 1) as f64);
        let i = (fx as usize).min(self.nx - 2);
        let j = (fy as usize).min(self.ny - 2);
        let (u, v) = (fx - i as f64, fy - j as f64);
        let at = |i: usize, j: usize| f64::from(self.heights[j * self.nx + i]);
        let south = at(i, j) + (at(i + 1, j) - at(i, j)) * u;
        let north = at(i, j + 1) + (at(i + 1, j + 1) - at(i, j + 1)) * u;
        south + (north - south) * v
    }
}
