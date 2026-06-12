//! Fixture loading shared by the checkpoint tests. Fields are added as the
//! KRs that consume them land.

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Deserialize)]
pub struct CheckpointFixture {
    pub shape: ShapeFixture,
    pub target_areas: Vec<f64>,
    pub initial_sites: Vec<[f32; 2]>,
    pub room_indices: Vec<usize>,
    pub iter0: Iter0Fixture,
}

#[derive(Deserialize)]
pub struct Iter0Fixture {
    pub geos_cell_order: Vec<usize>,
    pub n_raw_cells: usize,
    pub n_pieces: usize,
    pub split_positions: Vec<(usize, usize)>,
    #[serde(default)]
    pub split_pieces: Vec<(usize, Vec<f64>)>,
    pub cells_sorted: Vec<CellFixture>,
    pub losses: LossesFixture,
}

#[derive(Deserialize)]
pub struct LossesFixture {
    pub wall: f64,
    pub area: f64,
    pub lloyd: f64,
    pub topo: f64,
    pub bb: f64,
    pub cell: f64,
    pub total: f64,
}

#[derive(Deserialize)]
pub struct CellFixture {
    pub area: f64,
    pub centroid: Option<[f64; 2]>,
    pub is_empty: bool,
}

#[derive(Deserialize)]
pub struct ShapeFixture {
    pub boundary_coords: Vec<[f64; 2]>,
    pub area: f64,
    pub perimeter: f64,
}

pub fn fixture_path(file: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(file)
}

pub fn load_checkpoint(name: &str) -> CheckpointFixture {
    let path = fixture_path(&format!("{name}.checkpoint.json"));
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture {}: {e}", path.display()));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("cannot parse fixture {}: {e}", path.display()))
}

pub fn rel_err(actual: f64, expected: f64) -> f64 {
    if expected == 0.0 {
        actual.abs()
    } else {
        ((actual - expected) / expected).abs()
    }
}
