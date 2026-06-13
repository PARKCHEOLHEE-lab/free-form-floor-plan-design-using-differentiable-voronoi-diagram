//! Self-contained initialization, mirroring `FloorPlanGenerator.__init__`'s
//! site placement and torch-kmeans room assignment in distribution (not
//! bit-stream): a seeded Rust RNG replaces torch's RNG per the PRD — all
//! numeric equivalence claims go through fixtures, never RNG replication.

use geo::{Contains, InteriorPoint, Point, Polygon};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// `_initialize_parameters`: sample sites uniformly in the unit square,
/// recenter their mean on an interior representative point, then pull each
/// outside site toward that point and 0.05..0.15 past it. Panics (like
/// Python's assert) if a site still lands outside.
pub fn initialize_sites(boundary: &Polygon<f64>, n_sites: usize, seed: u64) -> Vec<[f32; 2]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let rep = boundary
        .interior_point()
        .expect("boundary polygon has an interior point");
    let (cx, cy) = (rep.x() as f32, rep.y() as f32);

    let mut sites: Vec<[f32; 2]> = (0..n_sites)
        .map(|_| [rng.gen::<f32>(), rng.gen::<f32>()])
        .collect();

    let mean_x: f32 = sites.iter().map(|s| s[0]).sum::<f32>() / n_sites as f32;
    let mean_y: f32 = sites.iter().map(|s| s[1]).sum::<f32>() / n_sites as f32;
    for s in sites.iter_mut() {
        s[0] -= mean_x - cx;
        s[1] -= mean_y - cy;
    }

    for s in sites.iter_mut() {
        let p = Point::new(s[0] as f64, s[1] as f64);
        if !boundary.contains(&p) {
            let vx = cx - s[0];
            let vy = cy - s[1];
            let norm = (vx * vx + vy * vy).sqrt();
            let t: f32 = rng.gen();
            let scale = (norm + ((1.0 - t) * 0.05 + t * 0.15)) / norm;
            s[0] += vx * scale;
            s[1] += vy * scale;
        }
    }

    for s in &sites {
        let p = Point::new(s[0] as f64, s[1] as f64);
        assert!(
            boundary.contains(&p),
            "site {s:?} fell outside the boundary after the inward pull"
        );
    }

    sites
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
