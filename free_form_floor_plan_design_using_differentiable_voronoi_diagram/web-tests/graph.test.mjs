// Corner model: room outlines from optimizer cells. Run: node --test web-tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { fromCells, roomRings } from "../web/plan/graph.js";
import { ringArea } from "../web/plan/geom.js";

// Four unit squares in a 2 x 2 block: the left column is room 0, the right column room 1.
const sq = (x, y) => [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]];
const cells = [
    { room: 0, ring: sq(0, 0) },
    { room: 0, ring: sq(0, 1) },
    { room: 1, ring: sq(1, 0) },
    { room: 1, ring: sq(1, 1) },
];
const boundary = [[0, 0], [2, 0], [2, 2], [0, 2]];
const key = (p) => `${p[0]},${p[1]}`;

test("room outlines come from cells, with shared corner ids and kinks dropped between junctions", () => {
    const g = fromCells(cells, boundary, { roomCount: 2, simplifyTol: 0.01 });
    const [left = []] = roomRings(g, 0);
    const [right = []] = roomRings(g, 1);

    // (1, 1) sits on the shared wall between two junctions, so it goes; (0, 1) and (2, 1)
    // lie on the boundary, which keeps every corner.
    assert.deepEqual(left.map(key).sort(), ["0,0", "0,1", "0,2", "1,0", "1,2"]);
    assert.deepEqual(right.map(key).sort(), ["1,0", "1,2", "2,0", "2,1", "2,2"]);

    // the two rooms hold the same corner ids for the shared wall's ends
    const ids = (r) => new Set(g.rooms[r][0]);
    const shared = [...ids(0)].filter((id) => ids(1).has(id)).map((id) => key([g.corners.get(id).x, g.corners.get(id).y]));
    assert.deepEqual(shared.sort(), ["1,0", "1,2"]);

    assert.ok(Math.abs(ringArea(left) - 2) < 1e-12);
    assert.ok(Math.abs(ringArea(right) - 2) < 1e-12);
});
