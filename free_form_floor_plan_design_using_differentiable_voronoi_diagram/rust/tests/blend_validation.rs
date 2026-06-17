//! GATE (Codex #1+#6 validation): the experimental `WallLocalMode::Blend`, used
//! with a NORMAL weight mix on a diagonal shape, must CONVERGE — not diverge as
//! the earlier pure-alignment-only attempt did when isolated. This is the check
//! the unit tests missed: green unit tests did not catch the demo regression, so
//! here we run the REAL optimization end to end and compare the two modes.
//!
//! Heavy (full finite-difference optimization); ignored by default, run in
//! release:
//!   cargo test --release --test blend_validation -- --ignored --nocapture

use geo::Area;
use voronoi_floorplan::loss::{floor_plan_loss, LossWeights, WallLocalMode};
use voronoi_floorplan::{grad, init, optim::AdamW, shapes};

/// Run a bounded AdamW + central-FD optimization on a diagonal shape with a
/// normal weight mix plus the given wall_local mode/weight. Returns
/// (initial_loss, final_loss, worst_seen_loss).
fn run_opt(mode: WallLocalMode, w_wall_local: f64) -> (f32, f32, f32) {
    let boundary = shapes::by_name("shape_e").unwrap().polygon(); // has diagonal walls
    let area = boundary.unsigned_area();
    let ratios = [0.3, 0.25, 0.2, 0.15, 0.1];
    let target_areas: Vec<f64> = ratios.iter().map(|r| area * r).collect();
    let mut sites = init::initialize_sites(&boundary, 40, 7);
    let room_indices = init::kmeans_labels(&sites, ratios.len(), 7);
    let w = LossWeights {
        w_wall: 2.5,
        w_area: 20.0,
        w_lloyd: 2.1,
        w_topo: 1.5,
        w_bb: 0.0,
        w_cell: 0.0,
        w_wall_local,
        wall_local_mode: mode,
    };
    let mut opt = AdamW::new(sites.len(), 1e-2);
    let initial = floor_plan_loss(&sites, &boundary, &target_areas, &room_indices, &w, None).total;
    let mut worst = initial;
    for it in 1..=200 {
        if it == 100 {
            opt.set_lr(8e-3);
        }
        let g = grad::finite_difference_grads(&sites, &boundary, &target_areas, &room_indices, &w, None);
        opt.step(&mut sites, &g);
        let cur = floor_plan_loss(&sites, &boundary, &target_areas, &room_indices, &w, None).total;
        worst = worst.max(cur);
    }
    let final_loss = floor_plan_loss(&sites, &boundary, &target_areas, &room_indices, &w, None).total;
    println!(
        "mode {:?}  w_wall_local {}: initial {:.4} -> final {:.4}  (worst {:.4})",
        mode, w_wall_local, initial, final_loss, worst
    );
    (initial, final_loss, worst)
}

#[test]
#[ignore = "heavy: real optimization, run in release"]
fn blend_mode_converges_with_normal_weight_mix() {
    let (n_i, n_f, _n_w) = run_opt(WallLocalMode::Nearest, 2.5);
    let (b_i, b_f, b_w) = run_opt(WallLocalMode::Blend, 2.5);

    assert!(
        n_f.is_finite() && b_f.is_finite(),
        "loss must stay finite (nearest {n_f}, blend {b_f})"
    );
    assert!(n_f < n_i, "Nearest must converge: {n_i} -> {n_f}");
    // The gate: Blend must CONVERGE (reduce loss), not diverge like the isolated
    // pure-alignment attempt. It also must not blow up mid-run beyond its start.
    assert!(b_f < b_i, "Blend must converge (not diverge): {b_i} -> {b_f}");
    assert!(
        b_w < b_i * 2.0,
        "Blend must not blow up mid-run: worst {b_w} vs initial {b_i}"
    );
}
