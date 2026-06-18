//! KR7: self-contained initialization — seeded site placement inside the
//! boundary plus k-means room assignment with k nonempty clusters,
//! deterministic per seed.

use geo::{Area, Contains, Point};
use voronoi_floorplan::{init, shapes};

#[test]
fn init_produces_deterministic_sites_inside_boundary_and_full_clusters() {
    for name in shapes::SHAPE_NAMES {
        let boundary = shapes::by_name(name).unwrap().polygon();

        let sites = init::initialize_sites(&boundary, 40, 777);
        assert_eq!(sites.len(), 40, "{name}: site count");
        assert_eq!(
            sites,
            init::initialize_sites(&boundary, 40, 777),
            "{name}: same seed must reproduce identical sites"
        );
        assert_ne!(
            sites,
            init::initialize_sites(&boundary, 40, 778),
            "{name}: different seed must change sites"
        );
        for (i, s) in sites.iter().enumerate() {
            let p = Point::new(s[0] as f64, s[1] as f64);
            assert!(
                boundary.contains(&p),
                "{name}: site {i} {s:?} is outside the boundary"
            );
        }

        let k = 5;
        let labels = init::kmeans_labels(&sites, k, 777);
        assert_eq!(labels.len(), sites.len(), "{name}: one label per site");
        assert_eq!(
            labels,
            init::kmeans_labels(&sites, k, 777),
            "{name}: same seed must reproduce identical labels"
        );
        let mut counts = vec![0usize; k];
        for &l in &labels {
            counts[l] += 1;
        }
        assert!(
            counts.iter().all(|&c| c > 0),
            "{name}: every cluster must be nonempty, got {counts:?}"
        );
    }
}

/// Poisson-disk init must place sites with a blue-noise minimum spacing: every
/// pair is at least ~0.5*sqrt(area/n) apart. Uniform-random placement clusters
/// points and fails this (its closest pair scales like sqrt(area)/n).
#[test]
fn init_sites_are_blue_noise_spread() {
    for name in shapes::SHAPE_NAMES {
        let boundary = shapes::by_name(name).unwrap().polygon();
        let n = 40usize;
        let sites = init::initialize_sites(&boundary, n, 777);

        let mut min_d2 = f64::INFINITY;
        for i in 0..sites.len() {
            for j in (i + 1)..sites.len() {
                let dx = (sites[i][0] - sites[j][0]) as f64;
                let dy = (sites[i][1] - sites[j][1]) as f64;
                let d2 = dx * dx + dy * dy;
                if d2 < min_d2 {
                    min_d2 = d2;
                }
            }
        }
        let min_d = min_d2.sqrt();
        let spacing = (boundary.unsigned_area() / n as f64).sqrt();
        let threshold = 0.5 * spacing;
        assert!(
            min_d >= threshold,
            "{name}: sites are not blue-noise spread — min pairwise distance {min_d:.4} < {threshold:.4} (blue-noise spacing {spacing:.4})"
        );
    }
}
