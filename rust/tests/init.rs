//! KR7: self-contained initialization — seeded site placement inside the
//! boundary plus k-means room assignment with k nonempty clusters,
//! deterministic per seed.

use geo::{Contains, Point};
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
