//! Regression for the GIF frame-ordering bug (PR #1, review round 1).
//!
//! The Python example steps the optimizer BEFORE capturing the frame:
//!     loss.backward(); optimizer.step(); generator.log(... captures frame ...)
//! (`free_form_floor_plan_design_using_differentiable_voronoi_diagram/python/examples/shape_a.py:79-81`),
//! so `optimization.gif` frame i shows the
//! sites *after* the i-th step. `run_example` used to render the frame before the
//! step, leaving the Rust GIF one iteration behind. This test pins the first GIF
//! frame to the POST-step geometry through the public CLI entry point.

use geo::Area;
use std::path::Path;
use voronoi_floorplan::{config, grad, init, loss, optim, render, run, shapes, voronoi};

fn decode_first_frame(path: &Path) -> Vec<u8> {
    let mut opts = gif::DecodeOptions::new();
    opts.set_color_output(gif::ColorOutput::RGBA);
    let mut decoder = opts.read_info(std::fs::File::open(path).unwrap()).unwrap();
    decoder
        .read_next_frame()
        .unwrap()
        .expect("gif has at least one frame")
        .buffer
        .to_vec()
}

/// Round-trip an RGBA frame through the exact same GIF encoder `run_example`
/// uses, so the decoded bytes are comparable to the run's decoded frame.
fn rgba_through_gif(rgba: &[u8], path: &Path) -> Vec<u8> {
    render::save_gif(&[rgba.to_vec()], path).unwrap();
    decode_first_frame(path)
}

#[test]
fn first_gif_frame_is_post_step_geometry_like_python() {
    let dir = std::env::temp_dir().join("voronoi_floorplan_gif_frame_order");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    let cfg = config::by_name("shape_a").unwrap();
    let seed = 777u64;

    // One full iteration through the public CLI pipeline.
    run::run_example(&cfg, seed, Some(1), &dir).unwrap();
    let actual = decode_first_frame(&dir.join("optimization.gif"));

    // Reconstruct that single iteration exactly as run_example does.
    let boundary = shapes::by_name(cfg.name).unwrap().polygon();
    let target_areas: Vec<f64> = cfg
        .area_ratio
        .iter()
        .map(|r| boundary.unsigned_area() * r)
        .collect();
    let mut sites = init::initialize_sites(&boundary, cfg.num_sites, seed);
    let room_indices = init::kmeans_labels(&sites, cfg.area_ratio.len(), seed);
    let weights = loss::LossWeights {
        w_wall: cfg.w_wall,
        w_area: cfg.w_area,
        w_lloyd: cfg.w_lloyd,
        w_topo: cfg.w_topo,
        w_bb: cfg.w_bb,
        w_cell: cfg.w_cell,
    };
    let n_rooms = cfg.area_ratio.len();

    // Frame at the initial (pre-step) sites.
    let pre_geom = voronoi::compute_cells(&sites, &boundary, None);
    let pre_rgba = render::render_frame(&boundary, &pre_geom.cells_sorted, &room_indices, n_rooms, &sites);
    let pre = rgba_through_gif(&pre_rgba, &dir.join("pre.gif"));

    // One AdamW step, identical to run_example's first iteration.
    let grads =
        grad::finite_difference_grads(&sites, &boundary, &target_areas, &room_indices, &weights, None);
    let mut opt = optim::AdamW::new(sites.len(), cfg.lr_initial);
    opt.step(&mut sites, &grads);

    // Frame at the post-step sites.
    let post_geom = voronoi::compute_cells(&sites, &boundary, None);
    let post_rgba = render::render_frame(&boundary, &post_geom.cells_sorted, &room_indices, n_rooms, &sites);
    let post = rgba_through_gif(&post_rgba, &dir.join("post.gif"));

    // Guard the reconstruction is discriminating: stepping must move the geometry.
    assert_ne!(pre, post, "one AdamW step must change the rendered geometry");

    assert_ne!(
        actual, pre,
        "first GIF frame must NOT be the pre-step (initial) geometry"
    );
    assert_eq!(
        actual, post,
        "first GIF frame must be the POST-step geometry (Python steps before capturing the frame)"
    );
}
