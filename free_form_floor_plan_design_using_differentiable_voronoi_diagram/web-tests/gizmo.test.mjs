// Gizmo layout: where the plan axes' ends sit on screen for a camera, and which end is nearest.
// Run: node --test web-tests/gizmo.test.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { axisTips } from "../web/plan/gizmo.js";

const near = (a, b) => Math.abs(a - b) < 1e-9;
const byAxis = (tips) => Object.fromEntries(tips.map((t) => [t.axis, t]));

test("axis ends: the top view shows X to the right, Y up and Z at the centre in front; isometric shows X right-up, Y left-up and Z up", () => {
    // three.js world: plan point (x, y) at height h is (x, h, -y)
    const top = byAxis(axisTips([1, 0, 0], [0, 0, -1], [0, 1, 0], 10));
    assert.deepEqual(Object.keys(top).sort(), ["+x", "+y", "+z", "-x", "-y"]);
    assert.ok(near(top["+x"].x, 10) && near(top["+x"].y, 0), `top view +X at ${top["+x"].x}, ${top["+x"].y}`);
    assert.ok(near(top["+y"].x, 0) && near(top["+y"].y, -10), `top view +Y at ${top["+y"].x}, ${top["+y"].y}`);
    assert.ok(near(top["-y"].x, 0) && near(top["-y"].y, 10), `top view -Y at ${top["-y"].x}, ${top["-y"].y}`);
    assert.ok(near(top["+z"].x, 0) && near(top["+z"].y, 0) && near(top["+z"].depth, 1), "top view +Z points at the viewer");

    // the isometric camera, south-west of the plan at elevation 35.26 degrees
    const el = Math.atan(1 / Math.SQRT2);
    const [c, s, h] = [Math.cos(el), Math.sin(el), 1 / Math.SQRT2];
    const tips = axisTips([h, 0, h], [s * h, c, -s * h], [-c * h, s, c * h], 10);
    const iso = byAxis(tips);
    assert.ok(near(iso["+x"].x, 10 * h) && near(iso["+x"].y, -10 * s * h), `iso +X at ${iso["+x"].x}, ${iso["+x"].y}`);
    assert.ok(near(iso["+y"].x, -10 * h) && near(iso["+y"].y, -10 * s * h), `iso +Y at ${iso["+y"].x}, ${iso["+y"].y}`);
    assert.ok(near(iso["+z"].x, 0) && near(iso["+z"].y, -10 * c), `iso +Z at ${iso["+z"].x}, ${iso["+z"].y}`);
    // drawn far to near, so +X and +Y (pointing away) come before the ends facing the camera
    for (let i = 1; i < tips.length; i++) assert.ok(tips[i - 1].depth <= tips[i].depth, "ends are ordered far to near");
    assert.deepEqual(tips.slice(0, 2).map((t) => t.axis).sort(), ["+x", "+y"]);
});
