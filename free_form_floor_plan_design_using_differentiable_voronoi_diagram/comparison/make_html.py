"""Builds comparison.html from the benchmark outputs.

Reads python_timing_<name>.json + rust_timing_<name>.json and references the
python_<name>.gif / rust_<name>.gif files (relative paths) to produce a
single self-contained report: a timing table, per-example speedup bars, and
side-by-side evolution GIFs.

Usage:  python comparison/make_html.py [out_dir]   (default: comparison/output)
"""

import os
import sys
import json
import html
import datetime

ORDER = ["shape_a", "shape_b", "shape_c", "shape_duck"]


def load(out_dir, impl, name):
    path = os.path.join(out_dir, f"{impl}_timing_{name}.json")
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def fmt_ms(x):
    return f"{x:,.1f}" if x >= 1 else f"{x:.3f}"


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "output")
    rows = []
    for name in ORDER:
        py = load(out_dir, "python", name)
        rs = load(out_dir, "rust", name)
        if py and rs:
            rows.append((name, py, rs))

    if not rows:
        print("no paired timing data found in", out_dir)
        sys.exit(1)

    speedups = [py["mean_iter_ms"] / rs["mean_iter_ms"] for _, py, rs in rows]
    avg_speedup = sum(speedups) / len(speedups)
    max_mean = max(max(py["mean_iter_ms"], rs["mean_iter_ms"]) for _, py, rs in rows)

    # --- timing table rows ---
    table_rows = ""
    for (name, py, rs), sp in zip(rows, speedups):
        table_rows += f"""
      <tr>
        <td class="ex">{html.escape(name)}</td>
        <td>{py['iterations']}</td>
        <td>{py['total_compute_s']:.2f}</td>
        <td>{fmt_ms(py['mean_iter_ms'])}</td>
        <td>{rs['total_compute_s']:.2f}</td>
        <td>{fmt_ms(rs['mean_iter_ms'])}</td>
        <td class="speed">{sp:.1f}&times;</td>
      </tr>"""

    # --- per-example mean-iteration bars (linear; rust is the tiny sliver) ---
    bars = ""
    for (name, py, rs), sp in zip(rows, speedups):
        pw = py["mean_iter_ms"] / max_mean * 100
        rw = max(rs["mean_iter_ms"] / max_mean * 100, 0.4)
        bars += f"""
      <div class="barblock">
        <div class="barlabel">{html.escape(name)} &middot; <b>{sp:.1f}&times; faster</b></div>
        <div class="bar"><div class="fill py" style="width:{pw:.2f}%"></div>
          <span class="bartext">Python {fmt_ms(py['mean_iter_ms'])} ms/iter</span></div>
        <div class="bar"><div class="fill rs" style="width:{rw:.2f}%"></div>
          <span class="bartext">Rust {fmt_ms(rs['mean_iter_ms'])} ms/iter</span></div>
      </div>"""

    # --- side-by-side GIFs ---
    gifs = ""
    for name, py, rs in rows:
        py_gif = f"python_{name}.gif"
        rs_gif = f"rust_{name}.gif"
        fixed_note = (
            ' <span class="fixtag">regenerated after the standalone cell-pairing fix'
            ' (see note below)</span>'
        )
        gifs += f"""
      <section class="gifrow">
        <h3>{html.escape(name)}
          <span class="sub">{py['iterations']} iterations from the identical fixture start</span>{fixed_note}</h3>
        <div class="pair">
          <figure><figcaption>Python (matplotlib &middot; torch FD backward)</figcaption>
            <img src="{py_gif}" alt="python {name}"></figure>
          <figure><figcaption>Rust (tiny-skia &middot; rayon FD backward)</figcaption>
            <img src="{rs_gif}" alt="rust {name}"></figure>
        </div>
      </section>"""

    iters = rows[0][1]["iterations"]
    threads = rows[0][2].get("threads", "?")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Differentiable Voronoi floor plan — Python vs Rust</title>
<style>
  :root {{ --py:#3776ab; --rs:#dea584; --ink:#1b1f24; --mut:#5a6573; --line:#e3e8ee; }}
  * {{ box-sizing: border-box; }}
  body {{ font: 15px/1.55 -apple-system, "Segoe UI", Roboto, sans-serif; color: var(--ink);
         margin: 0; background: #f6f8fa; }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 36px 22px 80px; }}
  h1 {{ font-size: 26px; margin: 0 0 4px; }}
  .lede {{ color: var(--mut); margin: 0 0 28px; }}
  .head {{ display:flex; gap:18px; flex-wrap:wrap; align-items:center; margin-bottom:26px; }}
  .badge {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 18px; }}
  .badge .n {{ font-size:30px; font-weight:700; }}
  .badge .l {{ color:var(--mut); font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line);
          border-radius:12px; overflow:hidden; margin-bottom:14px; }}
  th, td {{ padding:10px 12px; text-align:right; border-bottom:1px solid var(--line); }}
  th:first-child, td:first-child {{ text-align:left; }}
  thead th {{ background:#fbfcfd; font-size:12px; text-transform:uppercase; letter-spacing:.04em;
             color:var(--mut); }}
  tbody tr:last-child td {{ border-bottom:none; }}
  td.ex {{ font-weight:600; }}
  td.speed {{ font-weight:700; color:#0a7b34; }}
  h2 {{ font-size:18px; margin:34px 0 14px; }}
  .barblock {{ background:#fff; border:1px solid var(--line); border-radius:12px;
              padding:12px 16px; margin-bottom:10px; }}
  .barlabel {{ font-size:13px; color:var(--mut); margin-bottom:7px; }}
  .barlabel b {{ color:#0a7b34; }}
  .bar {{ position:relative; height:22px; background:#f0f2f5; border-radius:5px;
         margin:4px 0; overflow:hidden; }}
  .fill {{ height:100%; border-radius:5px; }}
  .fill.py {{ background:var(--py); }}
  .fill.rs {{ background:var(--rs); }}
  .bartext {{ position:absolute; left:9px; top:2px; font-size:12px; color:var(--ink);
             mix-blend-mode:luminosity; }}
  .gifrow {{ margin-top:26px; }}
  .gifrow h3 {{ font-size:16px; margin:0 0 10px; }}
  .gifrow h3 .sub {{ font-weight:400; color:var(--mut); font-size:13px; margin-left:8px; }}
  .pair {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  figure {{ margin:0; background:#fff; border:1px solid var(--line); border-radius:12px; padding:10px; }}
  figcaption {{ font-size:12px; color:var(--mut); margin-bottom:8px; }}
  img {{ width:100%; height:auto; display:block; border-radius:6px; background:#fff; }}
  .note {{ background:#fff8e6; border:1px solid #f3e2b0; border-radius:12px; padding:14px 18px;
          margin-top:30px; color:#5d4b16; font-size:14px; }}
  .note b {{ color:#7a5f10; }}
  .note.fix {{ background:#eaf7ee; border-color:#bfe3c9; color:#1f5130; }}
  .note.fix b {{ color:#13683a; }}
  .fixtag {{ display:inline-block; margin-left:8px; font-size:12px; font-weight:600;
            color:#13683a; background:#eaf7ee; border:1px solid #bfe3c9; border-radius:6px;
            padding:1px 7px; }}
  footer {{ color:var(--mut); font-size:12px; margin-top:40px; }}
  code {{ background:#eef1f4; padding:1px 5px; border-radius:4px; font-size:13px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Free-form floor plan via differentiable Voronoi — Python vs Rust</h1>
  <p class="lede">Same algorithm, same fixture start, same {iters} iterations per example.
     Timing covers the optimization compute only (forward loss + finite-difference
     backward + AdamW step); rendering and IO are excluded.</p>

  <div class="head">
    <div class="badge"><div class="n">{avg_speedup:.0f}&times;</div>
      <div class="l">mean speedup<br>(Rust vs Python)</div></div>
    <div class="badge"><div class="n">{iters}</div>
      <div class="l">iterations<br>per example</div></div>
    <div class="badge"><div class="n">{threads}</div>
      <div class="l">CPU threads<br>(both implementations)</div></div>
  </div>

  <h2>Per-iteration timing</h2>
  <table>
    <thead><tr>
      <th>Example</th><th>Iters</th>
      <th>Python total (s)</th><th>Python ms/iter</th>
      <th>Rust total (s)</th><th>Rust ms/iter</th>
      <th>Speedup</th>
    </tr></thead>
    <tbody>{table_rows}
    </tbody>
  </table>

  <h2>Mean iteration time</h2>
  {bars}

  <h2>Result evolution (identical start)</h2>
  {gifs}

  <div class="note fix">
    <b>Standalone cell-pairing fix (all four GIFs regenerated).</b> The earlier standalone
    (hint-free) run re-paired clipped cells to sites with an order-dependent containment
    search. Whenever a Voronoi cell, clipped to the boundary, split into a MultiPolygon,
    that search mis-assigned cells and rooms shattered. shape_b has such a split at
    iteration&nbsp;0 (10 of 40 sites wrong &rarr; the fragmentation + white gap around
    iter&nbsp;6); the others form <i>transient</i> splits mid-optimization (e.g. shape_duck
    ~iter&nbsp;12, shape_a ~iter&nbsp;4), so they shattered later in the run. The fix uses
    voronoice's <b>direct site&rarr;cell mapping</b> (each site is the generator of its own
    cell), which is order-independent. The checkpoint (hint) path that proves 1e-6
    equivalence was left untouched; two regression tests (<code>tests/pairing.rs</code>)
    now pin that every inside-boundary site lands in its assigned cell, at iteration&nbsp;0
    and across the first 25 iterations. All four Rust GIFs above are the fixed runs.
  </div>

  <div class="note">
    <b>Why the two animations still drift apart.</b> The backward pass is central finite
    differences over <b>float32</b> losses (it divides f32 loss differences by 2&times;10<sup>-6</sup>,
    amplifying last-bit float noise ~5&times;10<sup>5</sup>&times;), and AdamW normalizes each step to
    &plusmn;lr from the gradient <i>sign</i>. So a single last-bit difference between the
    pure-Rust geometry and Shapely/GEOS flips a step and the trajectories separate after
    ~4 iterations — a chaotic system with several near-optimal basins, not a porting bug
    (Python-vs-Python perturbation controls diverge the same way). The frozen-input
    checkpoint tests confirm the losses match to 1e-6 and the gradients match the f32
    quantum signature; here you are watching expected long-horizon chaos. Renderers also
    differ (matplotlib vs tiny-skia), styled to match.
  </div>

  <footer>Generated {stamp}. Timing is wall-clock of the optimization loop body on this
    machine; Python re-spawns a multiprocessing pool each iteration (as the original does),
    Rust uses a persistent rayon pool.</footer>
</div>
</body>
</html>"""

    out_path = os.path.join(out_dir, "comparison.html")
    with open(out_path, "w") as f:
        f.write(doc)
    print("wrote", out_path)
    print(f"avg speedup {avg_speedup:.1f}x over {len(rows)} examples")


if __name__ == "__main__":
    main()
