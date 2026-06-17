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
        let unions: Vec<_> = groups.iter().map(|g| loss::union_group(g)).collect();

        let wall = loss::compute_wall_loss(&unions, W_WALL) as f64;
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

/// KR4: the remaining loss components (lloyd, topology, bb, cell_area) and
/// the f32-ordered total match Python at iteration 0 within 1e-6 relative
/// (bb and cell_area against the nonzero-weight fixture variant, since every
/// example config pins w_bb = w_cell = 0).
#[test]
fn remaining_losses_and_total_match_python() {
    for name in shapes::SHAPE_NAMES {
        let fx = load_checkpoint(name);
        let boundary = shapes::by_name(name).unwrap().polygon();
        let hint = GeosOrderHint {
            cell_order: &fx.iter0.geos_cell_order,
            split_pieces: &fx.iter0.split_pieces,
        };
        let geom = voronoi::compute_cells(&fx.initial_sites, &boundary, Some(&hint));
        let groups = loss::rooms_group(&geom.cells_sorted, &fx.room_indices);
        let unions: Vec<_> = groups.iter().map(|g| loss::union_group(g)).collect();

        let lloyd = loss::compute_lloyd_loss(&geom.cells_sorted, &fx.initial_sites, 2.1) as f64;
        assert!(
            rel_err(lloyd, fx.iter0.losses.lloyd) <= 1e-6,
            "{name}: loss_lloyd {lloyd} vs Python {} (rel err {})",
            fx.iter0.losses.lloyd,
            rel_err(lloyd, fx.iter0.losses.lloyd)
        );

        let topo = loss::compute_topology_loss(&groups, &unions, 1.5) as f64;
        assert!(
            rel_err(topo, fx.iter0.losses.topo) <= 1e-6,
            "{name}: loss_topo {topo} vs Python {} (rel err {})",
            fx.iter0.losses.topo,
            rel_err(topo, fx.iter0.losses.topo)
        );

        let bb = loss::compute_bb_loss(&unions, 1.0) as f64;
        assert!(
            rel_err(bb, fx.iter0.losses_bbcell_w1.bb) <= 1e-6,
            "{name}: loss_bb {bb} vs Python {} (rel err {})",
            fx.iter0.losses_bbcell_w1.bb,
            rel_err(bb, fx.iter0.losses_bbcell_w1.bb)
        );

        let cell = loss::compute_cell_area_loss(&geom.cells_sorted, 1.0) as f64;
        assert!(
            rel_err(cell, fx.iter0.losses_bbcell_w1.cell) <= 1e-6,
            "{name}: loss_cell {cell} vs Python {} (rel err {})",
            fx.iter0.losses_bbcell_w1.cell,
            rel_err(cell, fx.iter0.losses_bbcell_w1.cell)
        );

        let w = loss::LossWeights {
            w_wall: 2.5,
            w_area: 20.0,
            w_lloyd: 2.1,
            w_topo: 1.5,
            w_bb: 0.0,
            w_cell: 0.0,
            w_wall_local: 0.0,
            ..Default::default()
        };
        let breakdown = loss::floor_plan_loss(
            &fx.initial_sites,
            &boundary,
            &fx.target_areas,
            &fx.room_indices,
            &w,
            Some(&hint),
        );
        assert!(
            rel_err(breakdown.total as f64, fx.iter0.losses.total) <= 1e-6,
            "{name}: total loss {} vs Python {} (rel err {})",
            breakdown.total,
            fx.iter0.losses.total,
            rel_err(breakdown.total as f64, fx.iter0.losses.total)
        );
    }
}

/// KR5: the finite-difference gradient matrix matches Python entry-by-entry.
/// An entry passes strictly when |g_rs - g_py| <= 1e-6 + 1e-3|g_py|. The
/// losses are f32, so a last-bit rounding flip in either of the two forward
/// evaluations shifts the gradient by exactly k * ulp(loss)/2e-6 for a small
/// integer k — entries failing the strict band must match that quantum
/// signature (the PRD's documented discrete-branch exception). PRD metric 2
/// was amended with user approval to this "stair rule": every entry must
/// pass strictly OR be an exact quantum multiple (k <= 8) — the measured
/// strict ceiling for pure-Rust geometry is 45-51%, and a genuine porting
/// bug produces a NON-quantum difference, which still fails. Zero
/// unexplained divergences are tolerated.
#[test]
fn finite_difference_gradients_match_python() {
    use voronoi_floorplan::grad;
    for name in shapes::SHAPE_NAMES {
        let fx = load_checkpoint(name);
        let boundary = shapes::by_name(name).unwrap().polygon();
        let hint = GeosOrderHint {
            cell_order: &fx.iter0.geos_cell_order,
            split_pieces: &fx.iter0.split_pieces,
        };
        let w = loss::LossWeights {
            w_wall: 2.5,
            w_area: 20.0,
            w_lloyd: 2.1,
            w_topo: 1.5,
            w_bb: 0.0,
            w_cell: 0.0,
            w_wall_local: 0.0,
            ..Default::default()
        };

        let grads = grad::finite_difference_grads(
            &fx.initial_sites,
            &boundary,
            &fx.target_areas,
            &fx.room_indices,
            &w,
            Some(&hint),
        );

        // gradient quantum: one f32 ulp of the loss, propagated through /2e-6
        let quantum = common::ulp_f32(fx.iter0.losses.total as f32) / 2e-6;

        for (i, (g, gpy)) in grads.iter().zip(&fx.iter0.grads).enumerate() {
            for j in 0..2 {
                let diff = (g[j] as f64 - gpy[j]).abs();
                if diff <= 1e-6 + 1e-3 * gpy[j].abs() {
                    continue; // strict match
                }
                let r = diff / quantum;
                let k = r.round();
                assert!(
                    (1.0..=8.0).contains(&k) && (r - k).abs() <= 1e-3,
                    "{name}: grad[{i}][{j}] = {} vs Python {} — diff {diff} is \
                     neither within tolerance nor an f32 quantum multiple \
                     (quantum {quantum}, ratio {r})",
                    g[j],
                    gpy[j]
                );
            }
        }
    }
}

/// KR6: starting from the fixture sites, the Rust loop (forward, FD
/// backward, AdamW step) tracks Python's loss trace for iterations 1-4
/// within 1% relative, with each iteration's GEOS orderings injected.
/// Iterations 5-10 are computed and printed informatively but not gated
/// (PRD metric 3, amended with user approval): Adam normalizes each step to
/// +-lr from the gradient sign, so a single f32-quantum gradient flip
/// reverses a coordinate's step; trajectories separate after ~4 iterations
/// and the faithfully-ported split-misalignment loss spikes fire at
/// trajectory-dependent iterations.
#[test]
fn optimization_trace_matches_python() {
    use voronoi_floorplan::{grad, optim::AdamW};
    for name in shapes::SHAPE_NAMES {
        let fx = load_checkpoint(name);
        let boundary = shapes::by_name(name).unwrap().polygon();
        let w = loss::LossWeights {
            w_wall: 2.5,
            w_area: 20.0,
            w_lloyd: 2.1,
            w_topo: 1.5,
            w_bb: 0.0,
            w_cell: 0.0,
            w_wall_local: 0.0,
            ..Default::default()
        };

        let mut sites = fx.initial_sites.clone();
        let mut opt = AdamW::new(sites.len(), 1e-2);

        for entry in &fx.trace {
            let hint = GeosOrderHint {
                cell_order: &entry.geos_cell_order,
                split_pieces: &entry.split_pieces,
            };
            let total = loss::floor_plan_loss(
                &sites,
                &boundary,
                &fx.target_areas,
                &fx.room_indices,
                &w,
                Some(&hint),
            )
            .total as f64;
            if entry.iteration <= 4 {
                assert!(
                    rel_err(total, entry.loss) <= 0.01,
                    "{name}: iteration {} loss {total} vs Python {} (rel err {})",
                    entry.iteration,
                    entry.loss,
                    rel_err(total, entry.loss)
                );
            } else {
                eprintln!(
                    "{name}: iteration {} (informative) loss {total} vs Python {} (rel err {:.4})",
                    entry.iteration,
                    entry.loss,
                    rel_err(total, entry.loss)
                );
            }

            let grads = grad::finite_difference_grads(
                &sites,
                &boundary,
                &fx.target_areas,
                &fx.room_indices,
                &w,
                Some(&hint),
            );
            opt.step(&mut sites, &grads);
        }
    }
}
