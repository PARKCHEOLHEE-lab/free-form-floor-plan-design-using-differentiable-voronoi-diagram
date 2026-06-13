//! Regression test for the standalone (no-hint) cell-to-site pairing.
//! Each site is the generator of its own Voronoi cell, so the cell assigned
//! to site i must contain site i. The order-dependent containment re-pairing
//! violated this for shape_b (10/40 sites mis-assigned) because of the
//! MultiPolygon split; this test pins the invariant for all examples.

mod common;

use common::load_checkpoint;
use geo::{Contains, Point};
use voronoi_floorplan::{shapes, voronoi};

#[test]
fn no_hint_pairing_assigns_each_site_to_a_cell_containing_it() {
    for name in shapes::SHAPE_NAMES {
        let fx = load_checkpoint(name);
        let boundary = shapes::by_name(name).unwrap().polygon();
        // standalone path (no fixture hint), as used by the GIF/CLI/outcome loop
        let geom = voronoi::compute_cells(&fx.initial_sites, &boundary, None);

        let mut mismatches = 0;
        for (i, cell) in geom.cells_sorted.iter().enumerate() {
            if voronoi::is_empty_cell(cell) {
                continue;
            }
            let p = Point::new(fx.initial_sites[i][0] as f64, fx.initial_sites[i][1] as f64);
            if !cell.contains(&p) {
                mismatches += 1;
            }
        }
        assert_eq!(
            mismatches, 0,
            "{name}: {mismatches} site(s) were assigned a cell that does not contain them \
             (no-hint pairing mis-assignment)"
        );
    }
}

#[test]
fn no_hint_pairing_stays_healthy_during_optimization() {
    use voronoi_floorplan::loss::LossWeights;
    use voronoi_floorplan::{grad, optim::AdamW};
    let w = LossWeights { w_wall: 2.5, w_area: 20.0, w_lloyd: 2.1, w_topo: 1.5, w_bb: 0.0, w_cell: 0.0 };
    for name in shapes::SHAPE_NAMES {
        let fx = load_checkpoint(name);
        let boundary = shapes::by_name(name).unwrap().polygon();
        let mut sites = fx.initial_sites.clone();
        let mut opt = AdamW::new(sites.len(), 1e-2);
        // Transient MultiPolygon splits form mid-run (e.g. shape_duck ~iter 12),
        // which is what the old zip-pop mishandled. The direct mapping must keep
        // every site still INSIDE the boundary inside its assigned cell. Sites
        // can legitimately drift OUTSIDE the boundary during optimization (the
        // loss does not constrain them — true in Python too); their
        // boundary-clipped cell then does not contain them, so they are exempt.
        for it in 1..=25usize {
            let g = grad::finite_difference_grads(&sites, &boundary, &fx.target_areas, &fx.room_indices, &w, None);
            opt.step(&mut sites, &g);
            let geom = voronoi::compute_cells(&sites, &boundary, None);
            let mism = geom.cells_sorted.iter().enumerate().filter(|(i, c)| {
                let p = Point::new(sites[*i][0] as f64, sites[*i][1] as f64);
                !voronoi::is_empty_cell(c) && boundary.contains(&p) && !c.contains(&p)
            }).count();
            assert_eq!(
                mism, 0,
                "{name}: iter {it} has {mism} inside-boundary site(s) assigned the wrong cell"
            );
        }
    }
}

#[test]
fn no_hint_cells_tile_the_boundary() {
    use geo::Area;
    use voronoi_floorplan::loss::LossWeights;
    use voronoi_floorplan::{grad, optim::AdamW};
    let w = LossWeights { w_wall: 2.5, w_area: 20.0, w_lloyd: 2.1, w_topo: 1.5, w_bb: 0.0, w_cell: 0.0 };
    for name in shapes::SHAPE_NAMES {
        let fx = load_checkpoint(name);
        let boundary = shapes::by_name(name).unwrap().polygon();
        let b_area = boundary.unsigned_area();
        let mut sites = fx.initial_sites.clone();
        let mut opt = AdamW::new(sites.len(), 1e-2);
        // The Voronoi cells partition the boundary, so the assigned cells must
        // cover essentially all of it; only tiny split slivers are dropped.
        // A whole cell going empty (e.g. an outside-boundary site at a concave
        // split — shape_duck ~iter 11) leaves a ~cell-sized hole.
        for it in 1..=25usize {
            let g = grad::finite_difference_grads(&sites, &boundary, &fx.target_areas, &fx.room_indices, &w, None);
            opt.step(&mut sites, &g);
            let geom = voronoi::compute_cells(&sites, &boundary, None);
            let covered: f64 = geom.cells_sorted.iter().map(|c| c.unsigned_area()).sum();
            let frac = covered / b_area;
            assert!(
                frac >= 0.99,
                "{name}: iter {it} cells cover only {:.3} of the boundary (a cell went empty)",
                frac
            );
        }
    }
}
