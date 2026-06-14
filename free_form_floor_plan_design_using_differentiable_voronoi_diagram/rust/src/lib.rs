// Geometry + optimization core — compiles for both native and wasm32.
pub mod config;
pub mod grad;
pub mod grad_local;
pub mod init;
pub mod loss;
pub mod optim;
pub mod shapes;
pub mod voronoi;

// Native-only: GIF/tfevents IO, the CLI, and the run pipeline that ties them
// together. Excluded from the wasm build (the browser demo renders to a
// <canvas> and has no filesystem).
#[cfg(not(target_arch = "wasm32"))]
pub mod cli;
#[cfg(not(target_arch = "wasm32"))]
pub mod render;
#[cfg(not(target_arch = "wasm32"))]
pub mod run;
#[cfg(not(target_arch = "wasm32"))]
pub mod tfevents;
