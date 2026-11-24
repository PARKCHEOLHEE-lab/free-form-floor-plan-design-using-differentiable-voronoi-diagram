// Exporting the state on screen as JSON (web/plan/export.js).
// Run: node --test web-tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { exportName, exportPlan } from "../web/plan/export.js";
import { fromCells } from "../web/plan/graph.js";

const square = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
const cells = [
    { room: 0, ring: [[-1, -1], [0, -1], [0, 0], [0, 1], [-1, 1]] },
    { room: 1, ring: [[0, 0], [1, 0], [1, 1], [0, 1]] },
    { room: 2, ring: [[0, -1], [1, -1], [1, 0], [0, 0]] },
];
const settings = {
    sites: 60, weights: { l1: 1, cos: 1, area: 0.5, lloyd: 0.12, topo: 0.06 }, iterations: 800, lr: 0.005, seed: 777,
    wallWidth: 0.02, pointMerge: 0.02, wallSnap: 0.02, walls: "aligned", doors: true, bubbles: false,
};
// the shape of RoomProgram.snapshot(): links hold room indices, ratios sit beside the rooms
const program = {
    rooms: [{ id: 4, center: [-0.5, 0], radius: 0.7 }, { id: 9, center: [0.5, 0.5], radius: 0.5 }, { id: 2, center: [0.5, -0.5], radius: 0.5 }],
    links: [[0, 1], [0, 2]],
    ratios: [0.5, 0.25, 0.25],
};
const now = new Date(2026, 10, 24, 9, 5, 7);
const state = (over = {}) => ({
    mode: "shape_e", boundary: square, program, settings, finished: false, edited: true, now,
    frame: { iteration: 12, cells, sites: new Float64Array([0.1, 0.2, -0.3, 0.4]) },
    plan: fromCells(cells, square, { roomCount: 3, simplifyTol: 0 }),
    ...over,
});

test("the export holds the boundary, the room program by id, the settings, the frame and the plan on screen, as plain JSON", () => {
    const out = exportPlan(state());
    assert.deepEqual(Object.keys(out), ["format", "version", "exportedAt", "boundary", "program", "settings", "frame", "plan"]);
    assert.deepEqual([out.format, out.version, out.exportedAt], ["voronoi-plan", 1, now.toISOString()]);
    assert.deepEqual(out.boundary, { mode: "shape_e", ring: square });
    assert.deepEqual(out.program, {
        rooms: [
            { id: 4, center: [-0.5, 0], radius: 0.7, ratio: 0.5 },
            { id: 9, center: [0.5, 0.5], radius: 0.5, ratio: 0.25 },
            { id: 2, center: [0.5, -0.5], radius: 0.5, ratio: 0.25 },
        ],
        links: [[4, 9], [4, 2]],
    }, "links name rooms by id, not by their place in the list");
    assert.deepEqual(out.settings, settings);
    assert.deepEqual(out.frame, { iteration: 12, finished: false, sites: [0.1, 0.2, -0.3, 0.4], cells });

    const g = state().plan;
    assert.equal(out.plan.source, "edited");
    assert.deepEqual(out.plan.corners, [...g.corners].sort(([a], [b]) => a - b).map(([id, c]) => ({ id, x: c.x, y: c.y, fixed: c.fixed })));
    assert.deepEqual(out.plan.rooms, g.rooms.map((rings, index) => ({ index, id: program.rooms[index].id, rings })), "rings list corner ids");
    assert.deepEqual(out.plan.areaShares, [0.5, 0.25, 0.25]);

    assert.deepEqual(JSON.parse(JSON.stringify(out)), out, "nothing is lost in JSON (no Map, no typed array)");
    out.boundary.ring[0][0] = 5;
    out.frame.cells[0].ring[0][0] = 5;
    assert.deepEqual([square[0][0], cells[0].ring[0][0]], [-1, -1], "the export shares no arrays with the page state");
});

test("without a room program or wall edits the export says so, and the file name carries the boundary and the local time", () => {
    const out = exportPlan(state({ program: null, edited: false, finished: true }));
    assert.equal(out.program, null);
    assert.equal(out.plan.source, "aligned");
    assert.equal(out.frame.finished, true);
    assert.deepEqual(out.plan.rooms.map((r) => r.id), [null, null, null]);
    assert.equal(exportName("shape_e", now), "plan-shape_e-20261124-090507.json");
    assert.equal(exportName("custom", new Date(2027, 0, 2, 23, 59, 0)), "plan-custom-20270102-235900.json");
});
