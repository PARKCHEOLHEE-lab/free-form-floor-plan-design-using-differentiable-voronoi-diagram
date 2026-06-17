//! Rust side of the Python-vs-Rust comparison: load an example's golden-fixture
//! start, run N optimization iterations timing only the compute (forward loss +
//! finite-difference backward + AdamW step, excluding rendering/IO), render one
//! frame per iteration, and emit `rust_<name>.gif` + `rust_timing_<name>.json`.
//! Pair with `bench_python.py` and `make_html.py` in the sibling `comparison/`
//! dir (`free_form_.../comparison`).
//!
//!   cargo run --release --example bench_compare -- <example> <iters> <out_dir>

use std::path::PathBuf;
use std::time::Instant;

use voronoi_floorplan::loss::LossWeights;
use voronoi_floorplan::render::{render_frame, GifWriter};
use voronoi_floorplan::{grad, loss, optim::AdamW, shapes, voronoi};

fn main() {
    let mut args = std::env::args().skip(1);
    let name = args.next().expect("usage: <example> <iters> <out_dir>");
    let iters: usize = args.next().expect("iters").parse().expect("iters is a number");
    let out_dir = PathBuf::from(args.next().expect("out_dir"));
    std::fs::create_dir_all(&out_dir).unwrap();

    let fx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(format!("{name}.checkpoint.json"));
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&fx_path).unwrap()).unwrap();
    let mut sites: Vec<[f32; 2]> = serde_json::from_value(v["initial_sites"].clone()).unwrap();
    let rooms: Vec<usize> = serde_json::from_value(v["room_indices"].clone()).unwrap();
    let targets: Vec<f64> = serde_json::from_value(v["target_areas"].clone()).unwrap();
    let n_rooms = targets.len();
    let boundary = shapes::by_name(&name).unwrap().polygon();
    let w = LossWeights { w_wall: 2.5, w_area: 20.0, w_lloyd: 2.1, w_topo: 1.5, w_bb: 0.0, w_cell: 0.0, w_wall_local: 0.0, ..Default::default() };

    let mut opt = AdamW::new(sites.len(), 1e-2);
    let mut gif = GifWriter::create(&out_dir.join(format!("rust_{name}.gif"))).unwrap();
    let mut per_iter_ms: Vec<f64> = Vec::with_capacity(iters);

    for it in 1..=iters {
        if it == 300 {
            opt.set_lr(8e-3);
        }
        // timed: one optimization iteration (forward + backward + step)
        let t = Instant::now();
        let _ = loss::floor_plan_loss(&sites, &boundary, &targets, &rooms, &w, None);
        let grads = grad::finite_difference_grads(&sites, &boundary, &targets, &rooms, &w, None);
        opt.step(&mut sites, &grads);
        per_iter_ms.push(t.elapsed().as_secs_f64() * 1e3);

        // untimed: render the post-step state
        let geom = voronoi::compute_cells(&sites, &boundary, None);
        gif.add_frame(&render_frame(&boundary, &geom.cells_sorted, &rooms, n_rooms, &sites)).unwrap();
    }

    let total_ms: f64 = per_iter_ms.iter().sum();
    let timing = serde_json::json!({
        "impl": "rust", "example": name, "iterations": iters,
        "total_compute_s": total_ms / 1e3, "mean_iter_ms": total_ms / iters as f64,
        "per_iter_ms": per_iter_ms,
        "threads": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
    });
    std::fs::write(
        out_dir.join(format!("rust_timing_{name}.json")),
        serde_json::to_string_pretty(&timing).unwrap(),
    ).unwrap();
    println!("rust {name}: {iters} iters, total {:.3}s, mean {:.3} ms/iter", total_ms / 1e3, total_ms / iters as f64);
}
