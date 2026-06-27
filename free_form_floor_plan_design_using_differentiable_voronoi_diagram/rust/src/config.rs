//! Per-example run configurations, mirroring the `configs` dicts in the four
//! example scripts (which differ only in shape and area_ratio).

pub struct ExampleConfig {
    pub name: &'static str,
    pub num_sites: usize,
    pub area_ratio: &'static [f64],
    pub w_wall: f64,
    pub w_area: f64,
    pub w_lloyd: f64,
    pub w_topo: f64,
    pub w_bb: f64,
    pub w_cell: f64,
    pub init_with_kmeans: bool,
    pub iterations: usize,
    pub iteration_to_modify_lr: usize,
    pub lr_initial: f64,
    pub lr_modified: f64,
}

const COMMON: ExampleConfig = ExampleConfig {
    name: "",
    num_sites: 40,
    area_ratio: &[],
    w_wall: 2.5,
    w_area: 20.0,
    w_lloyd: 2.1,
    w_topo: 1.5,
    w_bb: 0.0,
    w_cell: 0.0,
    init_with_kmeans: true,
    iterations: 800,
    iteration_to_modify_lr: 300,
    lr_initial: 1e-2,
    lr_modified: 8e-3,
};

pub fn by_name(name: &str) -> Option<ExampleConfig> {
    let (name, area_ratio): (&'static str, &'static [f64]) = match name {
        "shape_a" => ("shape_a", &[0.5, 0.3, 0.1, 0.1]),
        "shape_b" => ("shape_b", &[0.5, 0.2, 0.1, 0.1, 0.1]),
        "shape_c" => ("shape_c", &[0.4, 0.3, 0.2, 0.1]),
        "shape_duck" => ("shape_duck", &[0.2, 0.2, 0.2, 0.2, 0.2]),
        "shape_d" => ("shape_d", &[0.3, 0.25, 0.2, 0.15, 0.1]),
        "shape_e" => ("shape_e", &[0.3, 0.25, 0.2, 0.15, 0.1]),
        _ => return None,
    };
    Some(ExampleConfig {
        name,
        area_ratio,
        ..COMMON
    })
}
