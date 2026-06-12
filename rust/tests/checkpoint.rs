//! Checkpoint equivalence tests: frozen Python-exported inputs, numeric
//! comparison of the Rust port's outputs against the golden fixtures.

mod common;

use common::{load_checkpoint, rel_err};
use geo::{Area, Centroid};
use voronoi_floorplan::loss;
use voronoi_floorplan::shapes;
use voronoi_floorplan::voronoi::{self, GeosOrderHint};

/// The example weights shared by all four configurations.
const W_WALL: f64 = 2.5;
const W_AREA: f64 = 20.0;

/// KR1: the embedded boundary polygons reproduce the Python shapes exactly —
/// vertex count equal, area and perimeter within 1e-12 relative.
#[test]
fn shapes_match_python_fixture_geometry() {
    for name in shapes::SHAPE_NAMES {
        let fx = load_checkpoint(name);
        let shape = shapes::by_name(name).expect("known shape");

        assert_eq!(
            shape.boundary.len(),
            fx.shape.boundary_coords.len(),
            "{name}: boundary vertex count differs from Python"
        );

        let poly = shape.polygon();
        let area = poly.unsigned_area();
        assert!(
            rel_err(area, fx.shape.area) <= 1e-12,
            "{name}: area {area} vs Python {} (rel err {})",
            fx.shape.area,
            rel_err(area, fx.shape.area)
        );

        let perimeter: f64 = poly
            .exterior()
            .lines()
            .map(|l| (l.dx() * l.dx() + l.dy() * l.dy()).sqrt())
            .sum();
        assert!(
            rel_err(perimeter, fx.shape.perimeter) <= 1e-12,
            "{name}: perimeter {perimeter} vs Python {} (rel err {})",
            fx.shape.perimeter,
            rel_err(perimeter, fx.shape.perimeter)
        );
    }
}

/// KR2: with the GEOS orderings injected, the Rust forward geometry
/// reproduces Python's clipped-and-paired cells at iteration 0 — counts and
/// split structure equal, per-cell areas and centroids within 1e-8 relative
/// (+1e-12 absolute floor).
///
/// Tolerance calibration: voronoice/delaunator and GEOS compute Delaunay
/// circumcenters and overlay vertices with different arithmetic; measured
/// cross-stack noise on cell areas reaches ~3e-9 relative. A pairing or
/// clipping bug shows up at ~1e-1, eight orders above this tolerance, and
/// 1e-8 f64 noise is ~20x below one f32 ulp, so the downstream f32 loss
/// comparisons (1e-6 rel) are unaffected.
#[test]
fn forward_geometry_matches_python_cells() {
    for name in shapes::SHAPE_NAMES {
        let fx = load_checkpoint(name);
        let boundary = shapes::by_name(name).unwrap().polygon();
        let hint = GeosOrderHint {
            cell_order: &fx.iter0.geos_cell_order,
            split_pieces: &fx.iter0.split_pieces,
        };

        let geom = voronoi::compute_cells(&fx.initial_sites, &boundary, Some(&hint));

        assert_eq!(geom.n_raw_cells, fx.iter0.n_raw_cells, "{name}: raw cell count");
        assert_eq!(geom.n_pieces, fx.iter0.n_pieces, "{name}: clipped piece count");
        assert_eq!(
            geom.split_positions, fx.iter0.split_positions,
            "{name}: split positions"
        );
        assert_eq!(
            geom.cells_sorted.len(),
            fx.iter0.cells_sorted.len(),
            "{name}: paired cell count"
        );

        for (i, (cell, fxc)) in geom.cells_sorted.iter().zip(&fx.iter0.cells_sorted).enumerate() {
            assert_eq!(
                voronoi::is_empty_cell(cell),
                fxc.is_empty,
                "{name}: cell {i} emptiness"
            );
            let area = cell.unsigned_area();
            assert!(
                (area - fxc.area).abs() <= 1e-8 + 1e-8 * fxc.area.abs(),
                "{name}: cell {i} area {area} vs Python {} (abs err {})",
                fxc.area,
                (area - fxc.area).abs()
            );
            // Coordinates are O(1) (normalized shapes) and the cross-stack
            // vertex noise is absolute (~1e-9), so centroid comparison needs
            // an absolute floor — a relative-only band collapses near zero.
            if let Some([cx, cy]) = fxc.centroid {
                let c = cell.centroid().expect("nonempty cell has centroid");
                assert!(
                    (c.x() - cx).abs() <= 1e-8 + 1e-8 * cx.abs(),
                    "{name}: cell {i} centroid x {} vs Python {cx}",
                    c.x()
                );
                assert!(
                    (c.y() - cy).abs() <= 1e-8 + 1e-8 * cy.abs(),
                    "{name}: cell {i} centroid y {} vs Python {cy}",
                    c.y()
                );
            }
        }
    }
}

/// KR3: the wall and area losses, computed with the exact f32/f64 casting
/// boundaries of the torch implementation, match the Python iteration-0
/// values within 1e-6 relative on every example.
#[test]
fn wall_and_area_losses_match_python() {
    for name in shapes::SHAPE_NAMES {
        let fx = load_checkpoint(name);
        let boundary = shapes::by_name(name).unwrap().polygon();
        let hint = GeosOrderHint {
            cell_order: &fx.iter0.geos_cell_order,
            split_pieces: &fx.iter0.split_pieces,
        };
        let geom = voronoi::compute_cells(&fx.initial_sites, &boundary, Some(&hint));
        let groups = loss::rooms_group(&geom.cells_sorted, &fx.room_indices);

        let wall = loss::compute_wall_loss(&groups, W_WALL) as f64;
        assert!(
            rel_err(wall, fx.iter0.losses.wall) <= 1e-6,
            "{name}: loss_wall {wall} vs Python {} (rel err {})",
            fx.iter0.losses.wall,
            rel_err(wall, fx.iter0.losses.wall)
        );

        let area = loss::compute_area_loss(
            &geom.cells_sorted,
            &fx.target_areas,
            &fx.room_indices,
            W_AREA,
        ) as f64;
        assert!(
            rel_err(area, fx.iter0.losses.area) <= 1e-6,
            "{name}: loss_area {area} vs Python {} (rel err {})",
            fx.iter0.losses.area,
            rel_err(area, fx.iter0.losses.area)
        );
    }
}
