// Dragging one interior wall of an aligned plan (web/plan/wall-edit.js).
// Run: node --test web-tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { moveRun, roomShares, snapOffset, wallRun } from "../web/plan/wall-edit.js";
import { SNAP_LABELS } from "../web/plan/boundary-editor.js";
import { alignPlan } from "../web/plan/align.js";
import { fromCells, roomRings, wallEdges } from "../web/plan/graph.js";
import { isSimpleBoundary, openRing, segmentDistance, signedArea2 } from "../web/plan/geom.js";

const square = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
// R0 the left half; R1 and R2 the top and bottom right quarters; one T junction at the origin
const threeRooms = () => fromCells([
    { room: 0, ring: [[-1, -1], [0, -1], [0, 0], [0, 1], [-1, 1]] },
    { room: 1, ring: [[0, 0], [1, 0], [1, 1], [0, 1]] },
    { room: 2, ring: [[0, -1], [1, -1], [1, 0], [0, 0]] },
], square, { roomCount: 3, simplifyTol: 0 });
const idAt = (g, x, y) => [...g.corners].find(([, c]) => Math.abs(c.x - x) < 1e-12 && Math.abs(c.y - y) < 1e-12)[0];
const at = (g, id) => [g.corners.get(id).x, g.corners.get(id).y];
const areas = (g) => g.rooms.map((_, r) => roomRings(g, r).reduce((s, ring) => s + signedArea2(ring) / 2, 0));
const onBoundary = (p, boundary) => boundary.some((a, i) => segmentDistance(p, a, boundary[(i + 1) % boundary.length]) < 1e-9);

test("dragging a wall moves its straight run, changes only the rooms beside it, and stops before a wall vanishes", () => {
    // the stem between R1 and R2 slides up; its T junction slides along the long wall, its end along the boundary
    const g = threeRooms();
    const stem = wallRun(g, idAt(g, 0, 0), idAt(g, 1, 0));
    assert.ok(stem && !stem.locked, "the stem can move");
    assert.deepEqual(stem.edges.length, 1, "the stem is one edge");
    const up = moveRun(g, stem, 0.2);
    assert.ok(Math.abs(up.offset - 0.2) < 1e-12);
    assert.deepEqual(at(up.graph, idAt(g, 0, 0)).map((v) => +v.toFixed(12)), [0, 0.2], "the T junction slides along the long wall");
    assert.deepEqual(at(up.graph, idAt(g, 1, 0)).map((v) => +v.toFixed(12)), [1, 0.2], "the end slides along the boundary");
    const [a0, a1, a2] = areas(up.graph);
    assert.ok(Math.abs(a0 - 2) < 1e-12 && Math.abs(a1 - 0.8) < 1e-12 && Math.abs(a2 - 1.2) < 1e-12, `areas ${a0} ${a1} ${a2}`);
    assert.deepEqual(roomShares(up.graph).map((v) => +v.toFixed(12)), [0.5, 0.2, 0.3]);
    assert.deepEqual(areas(g), [2, 1, 1], "the original plan is left as it was");

    // the long wall is one run through the T junction; moving it moves the junction along the stem
    const long = wallRun(g, idAt(g, 0, 0), idAt(g, 0, 1));
    assert.equal(long.edges.length, 2, "both halves of the straight wall move together");
    const right = moveRun(g, long, 0.25);
    assert.deepEqual(at(right.graph, idAt(g, 0, 0)).map((v) => +v.toFixed(12)), [0.25, 0]);
    assert.deepEqual(areas(right.graph).map((v) => +v.toFixed(12)), [2.5, 0.75, 0.75]);

    // a drag past the top stops before R1 disappears
    const far = moveRun(g, stem, 2);
    assert.ok(far.offset > 0.9 && far.offset < 1, `stopped at ${far.offset}`);
    assert.ok(areas(far.graph).every((a) => a > 0), "every room keeps an area");

    // a wall that ends at a bend of the boundary cannot move
    const L = [[-1, -1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, 1]];
    const bend = fromCells([
        { room: 0, ring: [[-1, -1], [0, -1], [0, 0], [0, 1], [-1, 1]] },
        { room: 1, ring: [[0, -1], [1, -1], [1, 0], [0, 0]] },
    ], L, { roomCount: 2, simplifyTol: 0 });
    const locked = wallRun(bend, idAt(bend, 0, -1), idAt(bend, 0, 0));
    assert.equal(locked.locked, true);
    assert.equal(moveRun(bend, locked, 0.1), null);
    assert.equal(wallRun(g, idAt(g, 1, 0), idAt(g, 1, 1)), null, "a boundary edge is not a movable wall");

    // alignment leaves small kinks (a few degrees) where walls meet; such a wall still counts as straight
    const kinked = fromCells([
        { room: 0, ring: [[-1, -1], [0, -1], [0, 0], [0.05, 1], [-1, 1]] },
        { room: 1, ring: [[0, 0], [1, 0], [1, 1], [0.05, 1]] },
        { room: 2, ring: [[0, -1], [1, -1], [1, 0], [0, 0]] },
    ], square, { roomCount: 3, simplifyTol: 0 });
    const bent = wallRun(kinked, idAt(kinked, 0, 0), idAt(kinked, 0.05, 1));
    assert.equal(bent.locked, false, "a wall with a 3 degree kink can move");
    assert.equal(bent.edges.length, 2, "both pieces of the kinked wall move together");
    const kinkedStem = moveRun(kinked, wallRun(kinked, idAt(kinked, 0, 0), idAt(kinked, 1, 0)), 0.2);
    assert.ok(kinkedStem, "the stem that meets the kinked wall can move");
    const [k0, k1] = areas(kinkedStem.graph);
    assert.ok(Math.abs(k1 - (0.975 - 0.2)) < 0.01 && Math.abs(k0 - 2.025) < 0.01, `areas after the move ${k0} ${k1}`);
    for (const name of ["a", "b", "c", "d", "e"]) {
        const fixture = JSON.parse(readFileSync(new URL(`./fixtures/shape_${name}.json`, import.meta.url)));
        const p = fromCells(fixture.cells, fixture.boundary, { roomCount: fixture.ratios.length, simplifyTol: 0.004 });
        alignPlan(p, fixture.boundary, { tol: 0.02 });
        const walls = wallEdges(p).filter((w) => w.rooms.length === 2);
        const free = walls.filter((w) => !wallRun(p, w.a, w.b).locked).length;
        assert.ok(free >= walls.length / 2, `preset ${name.toUpperCase()}: ${free} of ${walls.length} walls can move`);
    }

    // on a real aligned plan, every movable wall keeps the plan valid when moved a little
    const run = JSON.parse(readFileSync(new URL("./fixtures/shape_e.json", import.meta.url)));
    const plan = fromCells(run.cells, run.boundary, { roomCount: run.ratios.length, simplifyTol: 0.004 });
    alignPlan(plan, run.boundary, { tol: 0.02 });
    const B = openRing(run.boundary), before = areas(plan);
    let moved = 0;
    for (const e of wallEdges(plan).filter((w) => w.rooms.length === 2)) {
        const r = wallRun(plan, e.a, e.b);
        if (r.locked) continue;
        const result = moveRun(plan, r, 0.03);
        moved++;
        const after = areas(result.graph);
        assert.ok(Math.abs(after.reduce((s, a) => s + a) - before.reduce((s, a) => s + a)) < 1e-9, "the total area stays");
        after.forEach((a, k) => { if (!r.rooms.includes(k)) assert.ok(Math.abs(a - before[k]) < 1e-12, `room ${k} is not beside the wall and keeps its area`); });
        assert.ok(r.rooms.some((k) => Math.abs(after[k] - before[k]) > 1e-6), "the rooms beside the wall change");
        for (const [, c] of result.graph.corners) if (c.fixed) assert.ok(onBoundary([c.x, c.y], B), "a boundary corner stays on the boundary");
        result.graph.rooms.forEach((rings, k) => roomRings(result.graph, k).forEach((ring) => assert.ok(isSimpleBoundary(ring), `room ${k} stays a simple shape`)));
    }
    assert.ok(moved >= 5, `${moved} walls of preset E can move`);
});

test("a dragged wall snaps to a corner or a wall midpoint within 10 screen pixels", () => {
    // a wall at y 0.5 on the left meets the long wall; the stem between R2 and R3 is dragged up
    const g = fromCells([
        { room: 0, ring: [[-1, 0.5], [0, 0.5], [0, 1], [-1, 1]] },
        { room: 1, ring: [[-1, -1], [0, -1], [0, 0], [0, 0.5], [-1, 0.5]] },
        { room: 2, ring: [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0.5]] },
        { room: 3, ring: [[0, -1], [1, -1], [1, 0], [0, 0]] },
    ], square, { roomCount: 4, simplifyTol: 0 });
    const stem = wallRun(g, idAt(g, 0, 0), idAt(g, 1, 0));
    const screen = (scale) => ([x, y]) => [400 + x * scale, 300 - y * scale];

    const corner = snapOffset(g, stem, 0.47, screen(100));
    assert.equal(corner.offset, 0.5, "3 px from the corner at y 0.5: the wall snaps onto it");
    assert.equal(SNAP_LABELS[corner.kind], "Endpoint");
    const middle = snapOffset(g, stem, 0.72, screen(100));
    assert.equal(middle.offset, 0.75, "3 px from the middle of the wall above: it snaps there");
    assert.equal(SNAP_LABELS[middle.kind], "Midpoint");
    assert.equal(middle.target[1], 0.75, "the snap target is the middle of a wall at y 0.75");
    const free = snapOffset(g, stem, 0.62, screen(100));
    assert.deepEqual([free.offset, free.kind], [0.62, "free"], "12 px away from every target: no snap");
    // the radius is in screen pixels: zoomed out, the same plan distance is close enough
    assert.equal(snapOffset(g, stem, 0.3, screen(100)).kind, "free", "20 px away at 100 px per unit");
    assert.equal(snapOffset(g, stem, 0.3, screen(20)).offset, 0.5, "4 px away at 20 px per unit");
});
