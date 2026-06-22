//! The optimization run pipeline: init -> loop(forward, log, render,
//! backward, step) -> gif, mirroring the Python example scripts end to end.

use crate::config::ExampleConfig;
use crate::loss::LossWeights;
use crate::render::GifWriter;
use crate::tfevents::TfEventsWriter;
use crate::{grad, init, loss, render, shapes, voronoi};
use geo::Area;
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn run_example(
    config: &ExampleConfig,
    seed: u64,
    iterations_override: Option<usize>,
    out_dir: &Path,
) -> io::Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let iterations = iterations_override.unwrap_or(config.iterations);

    let boundary = shapes::by_name(config.name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "unknown shape"))?
        .polygon();
    let target_areas: Vec<f64> = config
        .area_ratio
        .iter()
        .map(|r| boundary.unsigned_area() * r)
        .collect();

    let mut sites = init::initialize_sites(&boundary, config.num_sites, seed);
    let room_indices = init::kmeans_labels(&sites, config.area_ratio.len(), seed);

    let configs_json = serde_json::json!({
        "shape": config.name,
        "num_sites": config.num_sites,
        "area_ratio": config.area_ratio,
        "w_wall": config.w_wall,
        "w_area": config.w_area,
        "w_lloyd": config.w_lloyd,
        "w_topo": config.w_topo,
        "w_bb": config.w_bb,
        "w_cell": config.w_cell,
        "init_with_kmeans": config.init_with_kmeans,
        "iterations": iterations,
        "iteration_to_modify_lr": config.iteration_to_modify_lr,
        "lr_initial": config.lr_initial,
        "lr_modified": config.lr_modified,
        "seed": seed,
        "log_dir": out_dir.to_string_lossy(),
    });
    std::fs::write(
        out_dir.join("configs.json"),
        serde_json::to_string_pretty(&configs_json)?,
    )?;

    let now = || {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_secs_f64()
    };
    let mut events = TfEventsWriter::create(out_dir, now())?;
    let mut gif = GifWriter::create(&out_dir.join("optimization.gif"))?;

    let weights = LossWeights {
        w_wall: config.w_wall,
        w_area: config.w_area,
        w_lloyd: config.w_lloyd,
        w_topo: config.w_topo,
        w_bb: config.w_bb,
        w_cell: config.w_cell,
    };
    let mut optimizer = crate::optim::AdamW::new(sites.len(), config.lr_initial);

    for iteration in 1..=iterations {
        if iteration == config.iteration_to_modify_lr {
            optimizer.set_lr(config.lr_modified);
        }

        let b = loss::floor_plan_loss(
            &sites,
            &boundary,
            &target_areas,
            &room_indices,
            &weights,
            None,
        );

        let step = iteration as i64;
        let t = now();
        events.add_scalar("loss", b.total, step, t)?;
        events.add_scalar("loss_wall", b.wall, step, t)?;
        events.add_scalar("loss_area", b.area, step, t)?;
        events.add_scalar("loss_lloyd", b.lloyd, step, t)?;
        events.add_scalar("loss_topo", b.topo, step, t)?;
        events.add_scalar("loss_bb", b.bb, step, t)?;
        events.add_scalar("loss_cell_area", b.cell, step, t)?;

        let grads = grad::finite_difference_grads(
            &sites,
            &boundary,
            &target_areas,
            &room_indices,
            &weights,
            None,
        );
        optimizer.step(&mut sites, &grads);

        // Capture the frame AFTER the optimizer step, mirroring the Python
        // example (`optimizer.step()` then `generator.log(...)`, which captures
        // the frame): optimization.gif frame i shows the post-step sites.
        let geom = voronoi::compute_cells(&sites, &boundary, None);
        let frame = render::render_frame(
            &boundary,
            &geom.cells_sorted,
            &room_indices,
            config.area_ratio.len(),
            &sites,
        );
        gif.add_frame(&frame)?;

        println!("Iteration {iteration}, Loss: {}", b.total);
    }

    events.flush()?;
    Ok(())
}
