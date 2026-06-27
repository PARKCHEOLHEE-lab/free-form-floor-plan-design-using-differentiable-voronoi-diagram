//! KR9: CLI arg parsing and a 2-iteration smoke run producing all artifacts.

use clap::Parser;
use voronoi_floorplan::{cli::Args, config, run};

#[test]
fn args_parse_example_name_and_iterations_override() {
    let args = Args::try_parse_from(["vf", "shape_duck", "--iterations", "2"]).unwrap();
    assert_eq!(args.example, "shape_duck");
    assert_eq!(args.iterations, Some(2));
    assert_eq!(args.seed, 777);

    assert!(Args::try_parse_from(["vf"]).is_err(), "example name is required");
    assert!(config::by_name(&args.example).is_some());
    assert!(config::by_name("shape_z").is_none());
}

#[test]
fn smoke_run_produces_configs_tfevents_and_gif() {
    let dir = std::env::temp_dir().join("voronoi_floorplan_smoke_run");
    let _ = std::fs::remove_dir_all(&dir);

    let config = config::by_name("shape_a").unwrap();
    run::run_example(&config, 777, Some(2), &dir).unwrap();

    assert!(dir.join("configs.json").is_file(), "configs.json must exist");
    let has_events = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(Result::ok)
        .any(|e| e.file_name().to_string_lossy().starts_with("events.out.tfevents"));
    assert!(has_events, "an events.out.tfevents* file must exist");

    let gif_path = dir.join("optimization.gif");
    assert!(gif_path.is_file(), "optimization.gif must exist");
    let mut opts = gif::DecodeOptions::new();
    opts.set_color_output(gif::ColorOutput::RGBA);
    let mut decoder = opts.read_info(std::fs::File::open(&gif_path).unwrap()).unwrap();
    let mut n = 0;
    while decoder.read_next_frame().unwrap().is_some() {
        n += 1;
    }
    assert_eq!(n, 2, "one gif frame per iteration");
}
