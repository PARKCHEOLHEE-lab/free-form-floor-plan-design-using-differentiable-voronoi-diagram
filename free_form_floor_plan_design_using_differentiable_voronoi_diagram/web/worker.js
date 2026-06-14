// Module worker: owns the wasm optimizer and runs every step OFF the main
// thread, streaming each frame back for rendering. The main thread imports
// nothing from the wasm pkg — it only talks to this worker via postMessage.
// This keeps the demo a static, build-step-free site (the worker imports the
// same `--target web` ES module that index.html used to import directly).
//
// Protocol
//   main -> worker : {cmd:'build', kind, mode|boundary, sites, ratios, w[6], seed, lr, iters, then:'preview'|'start'}
//                    {cmd:'dispose'}
//   worker -> main : {type:'ready'} | {type:'built'} | {type:'error', msg}
//                    {type:'frame', frame, mode:'preview'|'step'} | {type:'done'}
import init, { WasmOpt } from './pkg/voronoi_floorplan_web.js';

let opt = null;
let maxIters = 0;
let running = false;
let gen = 0; // generation of the current opt; echoed on frames so the main
             // thread can drop frames from a superseded build (staleness guard)
let readyPromise = null;

// Initialize wasm exactly once, even if several messages race in before it
// resolves — they all await the same promise.
function ensureReady() {
  if (!readyPromise) readyPromise = init();
  return readyPromise;
}

// (Re)construct the optimizer from a build message. Returns false (and posts an
// error) for an unknown preset shape.
function build(msg) {
  if (opt) { opt.free(); opt = null; }
  gen = msg.gen;
  maxIters = msg.iters;
  const ratios = Float64Array.from(msg.ratios);
  const w = msg.w; // [w_wall, w_area, w_lloyd, w_topo, w_bb, w_cell]
  if (msg.kind === 'custom') {
    const flat = Float64Array.from(msg.boundary);
    opt = new WasmOpt(flat, msg.sites, ratios, w[0], w[1], w[2], w[3], w[4], w[5], msg.seed, msg.lr);
  } else {
    opt = WasmOpt.from_shape(msg.mode, msg.sites, ratios, w[0], w[1], w[2], w[3], w[4], w[5], msg.seed, msg.lr);
    if (!opt) {
      postMessage({ type: 'error', msg: 'unknown shape' });
      return false;
    }
  }
  return true;
}

// One optimization step, then reschedule. setTimeout(0) yields so the worker's
// own message queue can be processed between steps (needed for pause/reset).
function tick() {
  if (!running || !opt) return;
  const frame = opt.step();
  postMessage({ type: 'frame', frame, mode: 'step', gen });
  if (frame.iteration >= maxIters) {
    running = false;
    postMessage({ type: 'done', gen });
    return;
  }
  setTimeout(tick, 0);
}

onmessage = async (e) => {
  const msg = e.data;
  await ensureReady();
  switch (msg.cmd) {
    case 'build':
      if (build(msg)) {
        postMessage({ type: 'built', gen });
        if (msg.then === 'preview' && opt) {
          postMessage({ type: 'frame', frame: opt.current(), mode: 'preview', gen });
        } else if (msg.then === 'start') {
          running = true;
          tick();
        }
      }
      break;
    case 'pause':
      running = false; // the scheduled tick() sees this and stops
      break;
    case 'resume':
      if (opt && !running) { running = true; tick(); } // continues from the current iteration
      break;
    case 'dispose':
      running = false;
      if (opt) { opt.free(); opt = null; }
      break;
    default:
      break;
  }
};

// Initialize wasm up front and announce readiness so the UI can enable input.
(async () => {
  await ensureReady();
  postMessage({ type: 'ready' });
})();
