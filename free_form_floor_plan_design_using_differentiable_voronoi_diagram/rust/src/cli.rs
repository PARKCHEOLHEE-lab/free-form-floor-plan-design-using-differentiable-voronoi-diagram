//! Command-line interface: an example name selects the run configuration,
//! mirroring the role of the four Python example scripts.

use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "voronoi-floorplan", about = "Free-form floor plan generation with a differentiable Voronoi diagram (Rust port)")]
pub struct Args {
    /// Example configuration: shape_a | shape_b | shape_c | shape_duck
    pub example: String,

    /// Override the number of optimization iterations
    #[arg(long)]
    pub iterations: Option<usize>,

    /// RNG seed for site initialization and k-means
    #[arg(long, default_value_t = 777)]
    pub seed: u64,

    /// Output directory (default: runs/<example>/<timestamp>)
    #[arg(long)]
    pub out_dir: Option<std::path::PathBuf>,
}
