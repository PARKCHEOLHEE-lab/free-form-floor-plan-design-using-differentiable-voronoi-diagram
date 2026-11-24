// Wall alignment on the five sample runs (final frames in fixtures/shape_*.json).
// Run: node --test web-tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { alignPlan, alignedShare, cellWidth } from "../web/plan/align.js";
import { fromCells, roomRings } from "../web/plan/graph.js";
import { openRing, ringArea, signedArea2 } from "../web/plan/geom.js";

const load = (f) => JSON.parse(readFileSync(new URL(`./fixtures/${f}`, import.meta.url)));
// after: the earlier Python version on its own shapely graph; same_graph_after: its merge + align run
// on this corner graph (the fair comparison: shapely's union adds stub edges, 9 extra corners on
// shape C, that this model does not have)
const python = load("python-align.json");

// Same thresholds as the earlier Python version, in the optimizer's unit-radius space.
const TOL = 0.02;
const SIMPLIFY = 0.004;

for (const name of ["shape_a", "shape_b", "shape_c", "shape_d", "shape_e"]) {
    test(`${name} align keeps every room, covers the boundary, and aligns at least as much wall as the Python version`, () => {
        const run = load(`${name}.json`);
        const g = fromCells(run.cells, run.boundary, { roomCount: run.ratios.length, simplifyTol: SIMPLIFY });
        const before = alignedShare(g, run.boundary);
        alignPlan(g, run.boundary, { tol: TOL });
        const after = alignedShare(g, run.boundary);
        const py = python[name];

        run.ratios.forEach((_, r) => {
            const rings = roomRings(g, r);
            assert.ok(rings.length > 0, `room ${r} kept`);
            assert.ok(rings.reduce((s, ring) => s + signedArea2(ring), 0) > 0, `room ${r} not collapsed`);
        });
        const total = run.ratios.reduce((s, _, r) => s + roomRings(g, r).reduce((t, ring) => t + signedArea2(ring) / 2, 0), 0);
        assert.ok(Math.abs(total - ringArea(openRing(run.boundary))) < 1e-9, `rooms cover the boundary (${total})`);

        assert.ok(after > before, `aligned share rose (${before} -> ${after})`);
        assert.ok(after >= py.same_graph_after - 1e-4, `${after} >= Python version on this graph ${py.same_graph_after}`);
        assert.ok(after >= py.after - 0.01, `${after} >= Python version ${py.after} - 0.01`);
    });
}

test("point merge joins only the two ends of a wall piece shorter than the value", () => {
    const merged = (cells, boundary) => {
        const g = fromCells(cells, boundary, { roomCount: cells.length, simplifyTol: 0 });
        const before = g.corners.size;
        alignPlan(g, boundary, { mergeTol: 0.02, snapTol: 0 });
        return before - g.corners.size;
    };
    const unit = [[0, 0], [1, 0], [1, 1], [0, 1]];
    const step = [
        { room: 0, ring: [[0, 0], [1, 0], [1, 0.5], [0.5, 0.5], [0.5, 0.515], [0, 0.515]] },
        { room: 1, ring: [[0, 0.515], [0.5, 0.515], [0.5, 0.5], [1, 0.5], [1, 1], [0, 1]] },
    ];
    assert.equal(merged(step, unit), 1, "a 0.015 step riser becomes one point");
    const apart = [
        { room: 0, ring: [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]] },
        { room: 1, ring: [[0.51, 0], [1, 0], [1, 0.5], [0.51, 0.5]] },
    ];
    assert.equal(merged(apart, unit), 0, "corners 0.01 apart with no wall between them stay apart");
    const wide = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    const diagonal = [{ room: 0, ring: [[0, 0], [0.6, 0], [0.6, 0.6], [0.0177, 0.6177], [0, 0.6]] }];
    assert.equal(merged(diagonal, wide), 0, "a diagonal piece 0.025 long stays, though x and y differ by 0.0177");
});

test("point merging and wall snapping use independent tolerances", () => {
    const boundary = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    const polygon = [[0, 0], [0.6, 0.012], [0.6, 0.6], [0.014, 0.61], [0, 0.6]];
    const makeGraph = () => fromCells([{ room: 0, ring: polygon }], boundary, {
        roomCount: 1,
        simplifyTol: 0,
    });

    const merge = makeGraph();
    alignPlan(merge, boundary, { mergeTol: 0.02, snapTol: 0.005 });
    assert.equal(merge.corners.size, 4, "the close pair merges at the higher point tolerance");
    assert.equal(merge.corners.get(1).y - merge.corners.get(0).y, 0.012,
        "the bottom wall stays tilted at the lower snap tolerance");

    const snap = makeGraph();
    alignPlan(snap, boundary, { mergeTol: 0.005, snapTol: 0.02 });
    assert.equal(snap.corners.size, 5, "the lower point tolerance keeps both corners");
    assert.equal(snap.corners.get(1).y, snap.corners.get(0).y,
        "the bottom wall straightens at the higher snap tolerance");
});

test("cell width: the typical Voronoi cell width is the square root of the boundary area over the site count", () => {
    // a unit square split among 4 sites gives cells 0.5 wide; a closing point repeated at the end changes nothing
    assert.equal(cellWidth([[0, 0], [1, 0], [1, 1], [0, 1]], 4), 0.5);
    assert.equal(cellWidth([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], 4), 0.5);
    // preset E (area 1.328, measured with the shoelace formula in Python) with 80 sites
    const e = load("shape_e.json").boundary;
    assert.ok(Math.abs(cellWidth(e, 80) - 0.1288) < 5e-4, `preset E, 80 sites: ${cellWidth(e, 80)}`);
    assert.ok(Math.abs(cellWidth(e, 8) - cellWidth(e, 80) * Math.sqrt(10)) < 1e-12, "ten times fewer sites make cells sqrt(10) times wider");
});
