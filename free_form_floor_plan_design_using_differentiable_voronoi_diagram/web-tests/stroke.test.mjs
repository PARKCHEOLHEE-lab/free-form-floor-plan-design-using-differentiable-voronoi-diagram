// Wall band: a closed outline stroked to a band of half width d, as the 2D canvas stroked it.
// Run: node --test web-tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { strokeRing, strokeTriangles } from "../web/plan/stroke.js";

const triArea = ([a, b, c]) => Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;
const has = (pts, [x, y]) => pts.some((p) => Math.abs(p[0] - x) < 1e-12 && Math.abs(p[1] - y) < 1e-12);

test("a unit square stroked at half width 0.1 has miter corners at +-0.1 and band area 0.8", () => {
    const pairs = strokeRing([[0, 0], [1, 0], [1, 1], [0, 1]], 0.1);
    const pts = pairs.flat();
    for (const corner of [[-0.1, -0.1], [1.1, -0.1], [1.1, 1.1], [-0.1, 1.1], [0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        assert.ok(has(pts, corner), `corner ${corner} is on the band`);
    const area = strokeTriangles(pairs).reduce((s, t) => s + triArea(t), 0);
    assert.ok(Math.abs(area - 0.8) < 1e-12, `band area ${area}`);
});

test("a join sharper than the miter limit is bevelled, so no band point runs far past its corner", () => {
    const d = 0.1;
    // a 40 degree spike sits just past limit 2 (its miter would reach 2.9 d); a 5.7 degree one far past
    for (const ring of [[[0, 0], [3, 1.092], [0, 2.184]], [[0, 0], [10, 0.5], [0, 1]]]) {
        const pairs = strokeRing(ring, d, { miterLimit: 2 });
        for (const p of pairs.flat()) {
            const near = Math.min(...ring.map((v) => Math.hypot(p[0] - v[0], p[1] - v[1])));
            assert.ok(near <= 2 * d + 1e-12, `band point ${p} is ${near} from the nearest corner`);
        }
    }
});
