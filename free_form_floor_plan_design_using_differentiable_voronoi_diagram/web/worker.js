// Module worker: owns the wasm optimizer and runs every step OFF the main
// thread, streaming each frame back for rendering. The main thread imports
// nothing from the wasm pkg — it only talks to this worker via postMessage.
// This keeps the demo a static, build-step-free site (the worker imports the
// same `--target web` ES module that index.html used to import directly).
//
// Protocol
//   main -> worker : {cmd:'build', gen, mode, sites, ratios, w[6], seed, lr, iters,
//                    kind?: 'shape'|'custom', boundary?: [[x,y],...],
//                    cosineWeight?: number,
//                    program?: {rooms:[{center:[x,y],radius}],links:[[a,b]],doorWidth}, then:'preview'|'start'}
//                    {cmd:'dispose'}
//   worker -> main : {type:'ready'} | {type:'built', gen, boundary} | {type:'error', gen?, msg}
//                    {type:'frame', frame, mode:'preview'|'step'} | {type:'done'}
import init, { WasmOpt } from './pkg/voronoi_floorplan_web.js';
import { isSimpleBoundary } from './plan/geom.js';

let opt = null;
let maxIters = 0;
let running = false;
let gen = 0; // generation of the current opt; echoed on frames so the main
             // thread can drop frames from a superseded build (staleness guard)
let readyPromise = null;
let tickTimer = null;

function stop() {
  running = false;
  if (tickTimer !== null) clearTimeout(tickTimer);
  tickTimer = null;
}

function fail(error, requestGen) {
  stop();
  postMessage({type:'error', ...(requestGen === undefined ? {} : {gen:requestGen}), msg:error?.message ?? String(error)});
}

// Check before typed-array conversion, which would silently truncate fractional
// room indices. Geometry and engine-precision checks stay authoritative in Rust.
function programInput(program, siteCount) {
  const {rooms,links,doorWidth} = program ?? {};
  if (!Array.isArray(rooms) || !rooms.length || !Array.isArray(links)) throw new Error('Provide room circles and connection pairs.');
  if (!Number.isInteger(siteCount) || siteCount < Math.max(3,rooms.length)) throw new Error('Use at least three sites and one per room.');
  const circles = rooms.flatMap(room => {
    if (!Array.isArray(room?.center) || room.center.length!==2 || !room.center.every(Number.isFinite) || !Number.isFinite(room.radius)) throw new Error('Each room needs a finite x/y center and radius.');
    return [...room.center,room.radius];
  });
  if (!links.every(pair => Array.isArray(pair) && pair.length===2 && pair.every(i => Number.isInteger(i) && i >= 0 && i < rooms.length))) throw new Error('Connections must use existing integer room indices.');
  if (!Number.isFinite(doorWidth) || doorWidth<=0) throw new Error('Door width must be positive and finite.');
  return {circles:Float64Array.from(circles),links:Uint32Array.from(links.flat()),doorWidth};
}

// Initialize wasm exactly once, even if several messages race in before it
// resolves — they all await the same promise.
function ensureReady() {
  if (!readyPromise) readyPromise = init();
  return readyPromise;
}

// Build the replacement before releasing the last valid optimizer.
function build(msg) {
  stop();
  const program = msg.program === undefined ? null : programInput(msg.program,msg.sites);
  if (!Number.isInteger(msg.iters) || msg.iters<1) throw new Error('Iterations must be a positive integer.');
  const ratios = Float64Array.from(msg.ratios);
  const [wall, area, lloyd, topo, bb, cell] = msg.w;
  const args = [msg.sites, ratios, wall, area, lloyd, topo, bb, cell, msg.seed, msg.lr];
  const custom = msg.kind === 'custom';
  if (custom && !isSimpleBoundary(msg.boundary)) throw new Error('Boundary must enclose an area without crossing or retracing itself.');
  const next = custom
    ? new WasmOpt(Float64Array.from(msg.boundary.flat()), ...args)
    : WasmOpt.from_shape(msg.mode, ...args);
  if (!next) throw new Error('Unknown boundary.');
  try {
    if (msg.cosineWeight !== undefined) next.set_cosine_weight(msg.cosineWeight);
    if (program) next.set_program(program.circles,program.links,program.doorWidth);
    else next.fit_preset_areas();
  }
  catch(error) { next.free(); throw error; }
  if (opt) opt.free();
  opt = next;
  gen = msg.gen;
  maxIters = msg.iters;
  return true;
}

// One optimization step, then reschedule. setTimeout(0) yields so the worker's
// own message queue can be processed between steps (needed for pause/reset).
function tick() {
  tickTimer = null;
  if (!running || !opt) return;
  let frame;
  try { frame = opt.step(); }
  catch(error) { fail(error,gen); return; }
  postMessage({ type: 'frame', frame, mode: 'step', gen });
  if (frame.iteration >= maxIters) {
    running = false;
    postMessage({ type: 'done', gen });
    return;
  }
  tickTimer = setTimeout(tick, 0);
}

onmessage = async (e) => {
  const msg = e.data;
  try {
    await ensureReady();
    switch (msg.cmd) {
    case 'build':
      if (build(msg)) {
        postMessage({ type: 'built', gen, boundary: opt.boundary() });
        if (msg.then === 'preview' && opt) {
          postMessage({ type: 'frame', frame: opt.current(), mode: 'preview', gen });
        } else if (msg.then === 'start') {
          running = true;
          tick();
        }
      }
      break;
    case 'pause':
      stop();
      break;
    case 'resume':
      if (opt && !running) { running = true; tick(); } // continues from the current iteration
      break;
    case 'dispose':
      stop();
      if (opt) { opt.free(); opt = null; }
      break;
    default:
      break;
    }
  } catch(error) { fail(error,msg.gen); }
};

// Initialize wasm up front and announce readiness so the UI can enable input.
(async () => {
  try { await ensureReady(); postMessage({ type: 'ready' }); }
  catch(error) { fail(error); }
})();
