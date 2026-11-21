use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PrismKind {
    Building,
    Stadium,
    Bridge,
}

#[derive(Clone, Debug)]
pub struct Prism {
    pub kind: PrismKind,
    pub base: f64,
    pub top: f64,
    pub ring: Vec<[f64; 2]>,
    pub holes: Vec<Vec<[f64; 2]>>,
    pub bbox: [f64; 4],
    runs: Vec<Run>,
}

#[derive(Clone, Debug)]
struct Run {
    poly: usize,
    start: usize,
    end: usize,
    bbox: [f64; 4],
}

const RUN: usize = 8;
const RUN_SLACK: f64 = 1e-6;

fn crossings(poly: &[[f64; 2]], range: std::ops::Range<usize>, x: f64, y: f64) -> bool {
    let mut inside = false;
    for i in range {
        let (a, b) = (poly[i], poly[(i + poly.len() - 1) % poly.len()]);
        if (a[1] > y) != (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0] {
            inside = !inside;
        }
    }
    inside
}

fn make_runs(k: usize, poly: &[[f64; 2]], out: &mut Vec<Run>) {
    let n = poly.len();
    for start in (0..n).step_by(RUN) {
        let end = (start + RUN).min(n);
        let mut bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for i in start..=end {
            let p = poly[(i + n - 1) % n];
            bbox = [bbox[0].min(p[0]), bbox[1].min(p[1]), bbox[2].max(p[0]), bbox[3].max(p[1])];
            let q = poly[i % n];
            bbox = [bbox[0].min(q[0]), bbox[1].min(q[1]), bbox[2].max(q[0]), bbox[3].max(q[1])];
        }
        out.push(Run {
            poly: k,
            start,
            end,
            bbox: [bbox[0] - RUN_SLACK, bbox[1] - RUN_SLACK, bbox[2] + RUN_SLACK, bbox[3] + RUN_SLACK],
        });
    }
}

impl Prism {
    pub fn new(kind: PrismKind, base: f64, top: f64, ring: Vec<[f64; 2]>, holes: Vec<Vec<[f64; 2]>>) -> Prism {
        let mut bbox = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
        for p in &ring {
            bbox = [bbox[0].min(p[0]), bbox[1].min(p[1]), bbox[2].max(p[0]), bbox[3].max(p[1])];
        }
        let mut runs = Vec::new();
        for (k, poly) in std::iter::once(&ring).chain(&holes).enumerate() {
            make_runs(k, poly, &mut runs);
        }
        Prism {
            kind,
            base,
            top,
            ring,
            holes,
            bbox,
            runs,
        }
    }

    pub fn rect(kind: PrismKind, min: [f64; 2], max: [f64; 2], base: f64, top: f64) -> Prism {
        Prism::new(kind, base, top, vec![min, [max[0], min[1]], max, [min[0], max[1]]], Vec::new())
    }

    pub fn contains_xy(&self, x: f64, y: f64) -> bool {
        let [min_x, min_y, max_x, max_y] = self.bbox;
        if !(x >= min_x && x <= max_x && y >= min_y && y <= max_y) {
            return false;
        }
        let (mut current, mut parity) = (0, false);
        for r in &self.runs {
            if r.poly != current {
                if parity != (current == 0) {
                    return false;
                }
                (current, parity) = (r.poly, false);
            }
            if y < r.bbox[1] || y > r.bbox[3] || x > r.bbox[2] {
                continue;
            }
            parity ^= crossings(self.poly(r.poly), r.start..r.end, x, y);
        }
        parity == (current == 0)
    }

    fn poly(&self, k: usize) -> &[[f64; 2]] {
        if k == 0 {
            &self.ring
        } else {
            &self.holes[k - 1]
        }
    }

    pub(crate) fn edges_near(&self, a: [f64; 2], b: [f64; 2]) -> impl Iterator<Item = ([f64; 2], [f64; 2])> + '_ {
        let d = [b[0] - a[0], b[1] - a[1]];
        self.runs
            .iter()
            .filter(move |r| {
                let (mut t0, mut t1) = (0.0f64, 1.0f64);
                for (p, dp, lo, hi) in [(a[0], d[0], r.bbox[0], r.bbox[2]), (a[1], d[1], r.bbox[1], r.bbox[3])] {
                    if dp == 0.0 {
                        if p < lo || p > hi {
                            return false;
                        }
                    } else {
                        let (ta, tb) = ((lo - p) / dp, (hi - p) / dp);
                        t0 = t0.max(ta.min(tb));
                        t1 = t1.min(ta.max(tb));
                    }
                }
                t0 <= t1
            })
            .flat_map(move |r| {
                let poly = self.poly(r.poly);
                (r.start..r.end).map(move |i| (poly[i], poly[(i + 1) % poly.len()]))
            })
    }

    pub fn contains(&self, p: [f64; 3]) -> bool {
        p[2] >= self.base && p[2] <= self.top && self.contains_xy(p[0], p[1])
    }

    pub fn edges(&self) -> impl Iterator<Item = ([f64; 2], [f64; 2])> + '_ {
        std::iter::once(&self.ring)
            .chain(&self.holes)
            .flat_map(|poly| (0..poly.len()).map(move |i| (poly[i], poly[(i + 1) % poly.len()])))
    }

    pub fn bbox_near(&self, x: f64, y: f64, r: f64) -> bool {
        x >= self.bbox[0] - r && x <= self.bbox[2] + r && y >= self.bbox[1] - r && y <= self.bbox[3] + r
    }
}
