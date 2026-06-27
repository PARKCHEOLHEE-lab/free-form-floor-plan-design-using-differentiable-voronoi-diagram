//! Self-contained initialization, mirroring `FloorPlanGenerator.__init__`'s
//! site placement and torch-kmeans room assignment in distribution (not
//! bit-stream): a seeded Rust RNG replaces torch's RNG per the PRD — all
//! numeric equivalence claims go through fixtures, never RNG replication.

use geo::{Area, Contains, Point, Polygon};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Initialize Voronoi sites with a polygon-interior **Poisson-disk (blue-noise)**
/// distribution, mirroring the original paper's `poisson_disk_sampling_from_polyloop2`
/// (Wu–Tojo–Umetani, PG2024). Bridson dart-throwing fills the boundary with
/// well-spaced points at a minimum spacing derived from the area, then a
/// farthest-point pass trims to exactly `n_sites`, preserving the blue-noise
/// spacing. Deterministic per `seed`; every returned site lies inside `boundary`.
///
/// This replaces the earlier uniform-in-unit-square placement, which produced a
/// geometry-blind central blob that under-filled concave arms and left the
/// optimizer highly sensitive to the initial guess.
pub fn initialize_sites(boundary: &Polygon<f64>, n_sites: usize, seed: u64) -> Vec<[f32; 2]> {
    if n_sites == 0 {
        return Vec::new();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let pool = poisson_disk_fill(boundary, n_sites, &mut rng);
    let chosen = farthest_point_subsample(&pool, n_sites, &mut rng);
    chosen.iter().map(|p| [p[0] as f32, p[1] as f32]).collect()
}

/// Axis-aligned bounding box of the polygon exterior: `(min_x, min_y, max_x, max_y)`.
fn polygon_bbox(boundary: &Polygon<f64>) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for c in &boundary.exterior().0 {
        min_x = min_x.min(c.x);
        min_y = min_y.min(c.y);
        max_x = max_x.max(c.x);
        max_y = max_y.max(c.y);
    }
    (min_x, min_y, max_x, max_y)
}

/// Fill the boundary with a blue-noise point cloud of at least `n_target` points.
/// Picks a min-spacing `r` from the area (so one Bridson pass yields ~1.5×n_target),
/// shrinking `r` and retrying on the rare shape that packs short.
fn poisson_disk_fill(boundary: &Polygon<f64>, n_target: usize, rng: &mut ChaCha8Rng) -> Vec<[f64; 2]> {
    let area = boundary.unsigned_area().max(f64::MIN_POSITIVE);
    // r = spacing/sqrt(2); a Bridson pass then packs ~1.5x..2x n_target points.
    let mut r = (area / (2.0 * n_target as f64)).sqrt();
    let bbox = polygon_bbox(boundary);
    for _ in 0..8 {
        let pts = bridson(boundary, bbox, r, rng);
        if pts.len() >= n_target {
            return pts;
        }
        r *= 0.8; // smaller spacing packs more points
    }
    bridson(boundary, bbox, r, rng)
}

/// Bridson's fast Poisson-disk sampling (SIGGRAPH 2007) restricted to the polygon
/// interior: grid-accelerated dart throwing, accepting a candidate only when it is
/// inside `boundary` and at least `r` from every existing point.
fn bridson(
    boundary: &Polygon<f64>,
    (min_x, min_y, max_x, max_y): (f64, f64, f64, f64),
    r: f64,
    rng: &mut ChaCha8Rng,
) -> Vec<[f64; 2]> {
    use std::f64::consts::{SQRT_2, TAU};
    let cell = r / SQRT_2;
    let gw = ((max_x - min_x) / cell).ceil() as usize + 1;
    let gh = ((max_y - min_y) / cell).ceil() as usize + 1;
    let mut grid = vec![usize::MAX; gw * gh]; // grid cell -> point index, or MAX
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    let cell_of = |x: f64, y: f64| (((x - min_x) / cell) as usize, ((y - min_y) / cell) as usize);

    // seed the first point with a bounded rejection sample inside the polygon
    let mut seed_pt = None;
    for _ in 0..10_000 {
        let x = min_x + rng.gen::<f64>() * (max_x - min_x);
        let y = min_y + rng.gen::<f64>() * (max_y - min_y);
        if boundary.contains(&Point::new(x, y)) {
            seed_pt = Some([x, y]);
            break;
        }
    }
    let seed_pt = match seed_pt {
        Some(p) => p,
        None => return pts,
    };
    let (sx, sy) = cell_of(seed_pt[0], seed_pt[1]);
    grid[sy * gw + sx] = 0;
    pts.push(seed_pt);
    active.push(0);

    const K: usize = 30; // candidates per active point before it retires
    while !active.is_empty() {
        let a = rng.gen_range(0..active.len());
        let center = pts[active[a]];
        let mut placed = false;
        for _ in 0..K {
            let ang = rng.gen::<f64>() * TAU;
            let rad = r * (1.0 + rng.gen::<f64>()); // annulus [r, 2r]
            let x = center[0] + rad * ang.cos();
            let y = center[1] + rad * ang.sin();
            if x < min_x || x > max_x || y < min_y || y > max_y {
                continue;
            }
            if !boundary.contains(&Point::new(x, y)) {
                continue;
            }
            let (cx, cy) = cell_of(x, y);
            let mut ok = true;
            let x0 = cx.saturating_sub(2);
            let y0 = cy.saturating_sub(2);
            let x1 = (cx + 2).min(gw - 1);
            let y1 = (cy + 2).min(gh - 1);
            'scan: for gy in y0..=y1 {
                for gx in x0..=x1 {
                    let pi = grid[gy * gw + gx];
                    if pi != usize::MAX {
                        let dx = pts[pi][0] - x;
                        let dy = pts[pi][1] - y;
                        if dx * dx + dy * dy < r * r {
                            ok = false;
                            break 'scan;
                        }
                    }
                }
            }
            if ok {
                let pi = pts.len();
                pts.push([x, y]);
                grid[cy * gw + cx] = pi;
                active.push(pi);
                placed = true;
                break;
            }
        }
        if !placed {
            active.swap_remove(a);
        }
    }
    pts
}

/// Greedily keep the `n` most spread-out points (farthest-point sampling): start
/// from one point, then repeatedly add the point farthest from those already kept.
/// Trims the over-produced blue-noise pool to exactly `n` while improving spacing.
fn farthest_point_subsample(pts: &[[f64; 2]], n: usize, rng: &mut ChaCha8Rng) -> Vec<[f64; 2]> {
    if pts.len() <= n {
        return pts.to_vec();
    }
    let start = rng.gen_range(0..pts.len());
    let mut chosen = vec![start];
    let mut min_d2: Vec<f64> = pts
        .iter()
        .map(|p| {
            let dx = p[0] - pts[start][0];
            let dy = p[1] - pts[start][1];
            dx * dx + dy * dy
        })
        .collect();
    while chosen.len() < n {
        let mut best = 0usize;
        let mut best_d2 = -1.0;
        for (i, &d2) in min_d2.iter().enumerate() {
            if d2 > best_d2 {
                best_d2 = d2;
                best = i;
            }
        }
        chosen.push(best);
        for (i, p) in pts.iter().enumerate() {
            let dx = p[0] - pts[best][0];
            let dy = p[1] - pts[best][1];
            let d2 = dx * dx + dy * dy;
            if d2 < min_d2[i] {
                min_d2[i] = d2;
            }
        }
    }
    chosen.into_iter().map(|i| pts[i]).collect()
}

/// Lloyd's k-means with seeded random-site init (torch-kmeans's default
/// "rnd" scheme in spirit). An emptied cluster is reseeded at the site
/// farthest from the first cluster's center, so all k clusters end nonempty.
pub fn kmeans_labels(sites: &[[f32; 2]], k: usize, seed: u64) -> Vec<usize> {
    assert!(k <= sites.len(), "more clusters than sites");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // pick k distinct sites as initial centers
    let mut indices: Vec<usize> = (0..sites.len()).collect();
    for i in 0..k {
        let j = rng.gen_range(i..indices.len());
        indices.swap(i, j);
    }
    let mut centers: Vec<[f32; 2]> = indices[..k].iter().map(|&i| sites[i]).collect();

    let dist2 = |a: &[f32; 2], b: &[f32; 2]| {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        dx * dx + dy * dy
    };

    let mut labels = vec![0usize; sites.len()];
    for _ in 0..100 {
        let new_labels: Vec<usize> = sites
            .iter()
            .map(|s| {
                (0..k)
                    .min_by(|&a, &b| dist2(s, &centers[a]).total_cmp(&dist2(s, &centers[b])))
                    .unwrap()
            })
            .collect();

        let mut sums = vec![[0.0f32; 2]; k];
        let mut counts = vec![0usize; k];
        for (s, &l) in sites.iter().zip(&new_labels) {
            sums[l][0] += s[0];
            sums[l][1] += s[1];
            counts[l] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                centers[c] = [sums[c][0] / counts[c] as f32, sums[c][1] / counts[c] as f32];
            } else {
                let far = sites
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        dist2(a, &centers[0]).total_cmp(&dist2(b, &centers[0]))
                    })
                    .map(|(i, _)| i)
                    .unwrap();
                centers[c] = sites[far];
            }
        }

        let converged = new_labels == labels;
        labels = new_labels;
        if converged {
            break;
        }
    }
    labels
}
