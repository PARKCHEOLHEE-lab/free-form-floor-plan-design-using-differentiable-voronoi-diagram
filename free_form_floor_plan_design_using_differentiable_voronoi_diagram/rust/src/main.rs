use clap::Parser;
use std::time::{SystemTime, UNIX_EPOCH};
use voronoi_floorplan::{cli::Args, config, run};

fn main() {
    let args = Args::parse();
    let config = config::by_name(&args.example).unwrap_or_else(|| {
        eprintln!(
            "unknown example '{}'; expected one of shape_a, shape_b, shape_c, shape_d, shape_e, shape_duck",
            args.example
        );
        std::process::exit(2);
    });

    let out_dir = args.out_dir.unwrap_or_else(|| {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_secs();
        std::path::PathBuf::from("runs").join(&args.example).join(ts.to_string())
    });

    if let Err(e) = run::run_example(&config, args.seed, args.iterations, &out_dir) {
        eprintln!("run failed: {e}");
        std::process::exit(1);
    }
    println!("artifacts written to {}", out_dir.display());
}
