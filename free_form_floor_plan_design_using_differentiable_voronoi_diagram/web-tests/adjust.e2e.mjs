// The Adjustment panel in a real browser: drag one wall of the aligned plan by hand.
// Run: cd web-tests && npm install && npm run e2e
import { test } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { serve } from "./serve.mjs";

/** Open the page from the top, wait for the first preview, and run `fn(page)`. */
async function onPage(fn) {
    const { url, proc } = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width: 1145, height: 820 } });
        await page.goto(url);
        await page.waitForFunction(() => document.getElementById("status").textContent.includes("ready"), null, { timeout: 30_000 });
        await page.click('.gizmo [data-axis="+z"]'); // look straight down, so a sideways drag moves an upright wall
        return await fn(page);
    } finally {
        await browser.close();
        proc.kill();
    }
}
const areasNow = (page) => page.$$eval("#adjust-areas [data-now]", (es) => es.map((e) => e.textContent));
const pixels = (page) => page.$eval("#plan", (c) => c.toDataURL());
/** Move across the plan until a movable wall is under the pointer; returns that screen point. */
async function findWall(page) {
    for (const y of [410, 380, 440, 350, 470]) {
        for (let x = 380; x <= 900; x += 3) {
            await page.mouse.move(x, y);
            if (await page.$eval("#plan", (c) => "wallHover" in c.dataset)) return [x, y];
        }
    }
    return null;
}

test("adjustment: dragging a wall moves it, updates the room areas beside it, shows a snap label, and undoes", { timeout: 120_000 }, async () => {
    await onPage(async (page) => {
        assert.deepEqual(await page.$$eval(".toolbar .tool", (ts) => ts.map((t) => t.getAttribute("aria-label"))), ["Rooms", "Optimization", "Adjustment"]);
        await page.click("button[data-open=adjust]");
        assert.equal(await page.locator(".panel[data-panel=adjust] .label").first().textContent(), "Adjustment");
        assert.equal(await page.locator("[data-walls=aligned].active").count(), 1, "opening Adjustment shows the Aligned walls");
        assert.equal(await page.locator("#bubble-editing").isVisible(), true, "the editing bar shows");
        assert.equal(await page.locator("#bubble-editing [role=status]").textContent(), "Editing Walls");

        const before = await areasNow(page), shot = await pixels(page);
        assert.ok(before.length >= 3 && before.every((t) => /^\d+\.\d%$/.test(t)), `area list: ${before}`);
        const hit = await findWall(page);
        assert.ok(hit, "a wall can be found under the pointer");
        const labels = new Set();
        await page.mouse.down();
        for (let dx = 2; dx <= 60; dx += 2) {
            await page.mouse.move(hit[0] + dx, hit[1]);
            const label = await page.$eval("#wall-sketch text", (t) => (t.hasAttribute("hidden") ? "" : t.textContent));
            if (label) labels.add(label);
        }
        await page.mouse.up();
        const after = await areasNow(page), moved = await pixels(page);
        assert.notEqual(moved, shot, "the drawn walls change");
        const changed = after.filter((t, i) => t !== before[i]).length;
        assert.ok(changed >= 2, `the rooms beside the wall change: ${before} -> ${after}`);
        assert.ok(labels.has("Endpoint") || labels.has("Midpoint"), `a snap label shows while dragging: ${[...labels]}`);

        await page.click("#undo");
        assert.deepEqual(await areasNow(page), before, "Undo puts the wall back");
        assert.equal(await pixels(page), shot);
        await page.click("#redo");
        assert.deepEqual(await areasNow(page), after, "Redo moves it again");
        await page.mouse.move(1100, 700);
        await page.keyboard.press("Meta+z");
        assert.deepEqual(await areasNow(page), before, "Cmd+Z undoes the wall edit");
        await page.keyboard.press("Meta+Shift+z");
        assert.deepEqual(await areasNow(page), after, "Shift+Cmd+Z redoes it");
    });
});

/** Drag the first wall found sideways by `dx` css px. */
async function dragWall(page, dx = 40) {
    const hit = await findWall(page);
    assert.ok(hit, "a wall can be found under the pointer");
    await page.mouse.down();
    for (let x = 4; x <= dx; x += 4) await page.mouse.move(hit[0] + x, hit[1]);
    await page.mouse.up();
    await page.mouse.move(1100, 700);
}
const dialogOpen = (page) => page.$eval("#discard-walls", (d) => d.open);
const hasEdits = (page) => page.$eval("#plan", (c) => c.hasAttribute("data-wall-edits"));

test("adjustment: with wall edits, a control that rebuilds the walls asks first; Cancel keeps them, Discard clears them and goes on", { timeout: 120_000 }, async () => {
    await onPage(async (page) => {
        await page.click("button[data-open=adjust]");
        assert.equal(await hasEdits(page), false);
        await dragWall(page);
        assert.equal(await hasEdits(page), true, "the drag left a wall edit");
        const edited = await areasNow(page);

        // display controls and the wall width never rebuild the walls, so they never ask
        await page.click("button[data-open=rooms]");
        await page.click("[data-walls=raw]");
        await page.click("[data-walls=aligned]");
        await page.click(".toggle:has(#show-doors)");
        await page.locator("#wall").focus();
        await page.keyboard.press("ArrowLeft");
        assert.equal(await dialogOpen(page), false, "display controls do not ask");
        assert.equal(await hasEdits(page), true);

        // a room edit on the plan asks; Cancel draws nothing
        const rooms = await page.locator("#rooms_v").textContent();
        await page.mouse.move(560, 300);
        await page.mouse.down();
        assert.equal(await dialogOpen(page), true, "pressing on the plan in Rooms asks first");
        await page.mouse.up();
        await page.click("#discard-walls button[value=cancel]");
        assert.equal(await page.locator("#rooms_v").textContent(), rooms, "Cancel draws no room");

        // a slider that rebuilds asks, by keyboard and by mouse; Cancel keeps its value and the edits
        await page.click("button[data-open=optimize]");
        const iters = await page.locator("#iters").inputValue();
        await page.locator("#iters").focus();
        await page.keyboard.press("ArrowRight");
        assert.equal(await dialogOpen(page), true, "a slider key asks first");
        await page.click("#discard-walls button[value=cancel]");
        await page.locator("#iters").click({ position: { x: 5, y: 2 } });
        assert.equal(await dialogOpen(page), true, "a slider press asks first");
        await page.click("#discard-walls button[value=cancel]");
        assert.equal(await page.locator("#iters").inputValue(), iters, "Cancel keeps the slider value");
        assert.deepEqual(await areasNow(page), edited, "Cancel keeps the wall edits");

        // Run asks; Cancel does not run; Discard clears the edits and runs
        await page.click("#run");
        assert.equal(await dialogOpen(page), true, "Run asks first");
        await page.click("#discard-walls button[value=cancel]");
        assert.equal(await page.locator("#run").getAttribute("aria-label"), "Run", "Cancel does not run");
        assert.equal(await hasEdits(page), true);
        await page.click("#run");
        await page.click("#discard-walls button[value=discard]");
        await page.waitForFunction(() => !document.getElementById("discard-walls").open); // the dialog closes in the next task
        assert.equal(await hasEdits(page), false, "Discard clears the wall edits");
        await page.waitForFunction(() => document.getElementById("run").getAttribute("aria-label") === "Pause");
        await page.click("#run");
    });
});

test("adjustment: walls cannot be dragged while the optimizer runs, and can after Pause", { timeout: 180_000 }, async () => {
    await onPage(async (page) => {
        const message = () => page.locator("#adjust-message").textContent();
        await page.click("button[data-open=optimize]");
        await page.locator("#iters").focus();
        await page.keyboard.press("End"); // the longest run, so it is still going while the test looks
        await page.waitForFunction(() => document.getElementById("status").textContent.includes("ready"));
        await page.click("#run");
        await page.waitForFunction(() => document.getElementById("run").getAttribute("aria-label") === "Pause");

        await page.click("button[data-open=adjust]");
        assert.match(await message(), /Pause the run/, `the panel says to pause first: ${await message()}`);
        assert.equal(await findWall(page), null, "no wall can be picked while the plan is changing");
        assert.equal(await hasEdits(page), false);

        await page.click("button[data-open=optimize]");
        await page.click("#run"); // pause
        await page.waitForFunction(() => document.getElementById("run").getAttribute("aria-label") === "Resume");
        await page.click("button[data-open=adjust]");
        assert.doesNotMatch(await message(), /Pause the run/, "after Pause the panel explains dragging again");
        await dragWall(page);
        assert.equal(await hasEdits(page), true, "after Pause a wall drags");
    });
});

/** Sweep the plan until an inner corner (not on the boundary) is under the pointer; returns its screen point and id. */
async function findCorner(page) {
    for (let y = 250; y <= 620; y += 10) {
        for (let x = 430; x <= 860; x += 10) {
            await page.mouse.move(x, y);
            const hit = await page.$eval("#plan", (c) => ("cornerHover" in c.dataset && !("cornerBoundary" in c.dataset) ? c.dataset.cornerHover : null));
            if (hit !== null) return { at: [x, y], id: hit };
        }
    }
    return null;
}
const hovered = (page) => page.$eval("#plan", (c) => ({ corner: c.dataset.cornerHover ?? null, wall: "wallHover" in c.dataset }));

test("adjustment: a corner is picked before a wall within 8 px, dragging it reshapes the rooms with a snap label, and one drag is one undo step", { timeout: 180_000 }, async () => {
    await onPage(async (page) => {
        await page.click("button[data-open=adjust]");
        const found = await findCorner(page);
        assert.ok(found, "an inner corner can be found under the pointer");
        const [x, y] = found.at;
        assert.deepEqual(await hovered(page), { corner: found.id, wall: false }, "on a corner the corner is picked, not the walls through it");
        await page.mouse.move(x + 6, y);
        assert.equal((await hovered(page)).corner, found.id, "6 px away the same corner is still picked");
        await page.mouse.move(x + 11, y);
        assert.notEqual((await hovered(page)).corner, found.id, "11 px away it is not");

        const before = await areasNow(page), shot = await pixels(page);
        await page.mouse.move(x, y);
        await page.mouse.down();
        for (let k = 1; k <= 8; k++) await page.mouse.move(x + 3 * k, y + 2 * k);
        await page.mouse.up();
        await page.mouse.move(1100, 700);
        const after = await areasNow(page);
        assert.notEqual(await pixels(page), shot, "the drawn walls change");
        assert.ok(after.filter((t, i) => t !== before[i]).length >= 2, `the rooms around the corner change: ${before} -> ${after}`);
        await page.click("#undo");
        assert.deepEqual(await areasNow(page), before, "one Undo puts the corner back");
        assert.equal(await pixels(page), shot);

        // with Shift the corner moves only sideways and the X/Y label shows
        const labels = new Set();
        await page.mouse.move(x, y);
        await page.keyboard.down("Shift");
        await page.mouse.down();
        for (let k = 1; k <= 10; k++) {
            await page.mouse.move(x + 3 * k, y + k);
            const label = await page.$eval("#wall-sketch text", (t) => (t.hasAttribute("hidden") ? "" : t.textContent));
            if (label) labels.add(label);
        }
        await page.mouse.up();
        await page.keyboard.up("Shift");
        assert.ok(labels.has("X/Y"), `the X/Y snap label shows while Shift is held: ${[...labels]}`);
    });
});

const cameraPose = (page) => page.$eval("#plan", (c) => [c.dataset.azimuth, c.dataset.elevation, c.dataset.zoom]);
const selectedCorner = (page) => page.$eval("#plan", (c) => c.dataset.selectedCorner ?? null);
/** Move off and back onto `at`, so the hover is picked again there. */
async function hoverAt(page, [x, y]) {
    await page.mouse.move(1100, 700);
    await page.mouse.move(x, y);
    return hovered(page);
}

test("adjustment: a click selects a corner, Delete removes it, a double click on a wall adds a corner that drags at once, Undo restores each step, and a refused delete says why", { timeout: 240_000 }, async () => {
    await onPage(async (page) => {
        await page.click("button[data-open=adjust]");
        const pose = await cameraPose(page);
        const wall = await findWall(page);
        assert.ok(wall, "a wall can be found under the pointer");
        const before = await areasNow(page);

        // a double click on a wall adds a corner there, selected, without turning the camera home
        await page.mouse.dblclick(...wall);
        assert.deepEqual(await cameraPose(page), pose, "the double click on a wall does not reset the view");
        const added = await selectedCorner(page);
        assert.ok(added, "the new corner is selected");
        assert.equal((await hoverAt(page, wall)).corner, added, "the new corner sits under the pointer");
        assert.deepEqual(await areasNow(page), before, "adding a corner changes no area");

        // it drags at once (areas show one decimal, so a small bump may round away in one of the two rooms)
        const flat = await pixels(page);
        await page.mouse.down();
        for (let k = 1; k <= 8; k++) await page.mouse.move(wall[0] + 4 * k, wall[1] + 3 * k);
        await page.mouse.up();
        const dragged = await areasNow(page);
        assert.notEqual(await pixels(page), flat, "the walls bend at the new corner");
        assert.ok(dragged.some((t, i) => t !== before[i]), `the rooms beside the wall change: ${before} -> ${dragged}`);

        // Undo takes back the drag, then the new corner
        await page.click("#undo");
        assert.deepEqual(await areasNow(page), before, "Undo puts the corner back on the wall");
        assert.equal((await hoverAt(page, wall)).corner, added);
        await page.click("#undo");
        assert.deepEqual(await hoverAt(page, wall), { corner: null, wall: true }, "a second Undo removes the added corner");

        // Delete removes the selected corner; Undo brings it back
        await page.mouse.dblclick(...wall);
        const again = await selectedCorner(page);
        assert.ok(again);
        await page.keyboard.press("Delete");
        assert.equal(await selectedCorner(page), null, "nothing is selected after the delete");
        assert.deepEqual(await hoverAt(page, wall), { corner: null, wall: true }, "Delete removes the selected corner");
        await page.keyboard.press("Meta+z");
        assert.equal((await hoverAt(page, wall)).corner, again, "Undo brings the corner back");

        // a click selects a corner; a corner where three walls meet cannot be deleted, and the help line says why
        const tried = new Set([again]);
        let refused = null;
        for (let y = 250; y <= 620 && !refused; y += 10) {
            for (let x = 430; x <= 860 && !refused; x += 10) {
                await page.mouse.move(x, y);
                const { corner } = await hovered(page);
                if (corner === null || tried.has(corner)) continue;
                tried.add(corner);
                await page.mouse.down();
                await page.mouse.up();
                assert.equal(await selectedCorner(page), corner, "a click selects the corner under the pointer");
                await page.keyboard.press("Backspace");
                if (/cannot be deleted/.test(await page.locator("#adjust-message").textContent())) refused = corner;
                else await page.keyboard.press("Meta+z");
            }
        }
        assert.ok(refused, "some corner of the preset refuses a delete and the help line says why");
        assert.equal(await selectedCorner(page), refused, "a refused delete keeps the corner selected");
    });
});

const handle = (page) => page.$eval("#wall-sketch [data-handle]", (h) => (h.hasAttribute("hidden") || h.closest("svg").hasAttribute("hidden") ? null : [+h.getAttribute("cx"), +h.getAttribute("cy")]));
const selectedWall = (page) => page.$eval("#plan", (c) => c.dataset.selectedWall ?? null);
/** The screen segments of the selected wall's highlight. */
const chosenSegments = (page) => page.$eval("#wall-sketch [data-chosen]", (p) => [...p.getAttribute("d").matchAll(/M([-\d.e]+),([-\d.e]+)L([-\d.e]+),([-\d.e]+)/g)].map((m) => m.slice(1, 5).map(Number)));
function footOn(segments, [x, y]) {
    let best = null;
    for (const [ax, ay, bx, by] of segments) {
        const dx = bx - ax, dy = by - ay, t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)));
        const foot = [ax + t * dx, ay + t * dy], d = Math.hypot(x - foot[0], y - foot[1]);
        if (!best || d < best.d) best = { foot, d };
    }
    return best;
}

test("adjustment: clicking a wall shows a round rotate handle, dragging it turns the wall with Parallel or Perpendicular labels, Escape clears it, and a locked wall shows none", { timeout: 240_000 }, async () => {
    await onPage(async (page) => {
        await page.click("button[data-open=adjust]");
        assert.match(await page.locator("#adjust-message").textContent(), /drag.*wall.*corner.*click.*rotate.*double.click.*Delete/i, "the help line names the four actions");
        const wall = await findWall(page);
        assert.ok(wall);
        await page.mouse.down();
        await page.mouse.up();
        assert.ok(await selectedWall(page), "a click selects the wall");
        const h = await handle(page);
        assert.ok(h, "the rotate handle shows");
        const { foot, d } = footOn(await chosenSegments(page), h);
        assert.ok(Math.abs(d - 28) < 2, `the handle sits about 28 px from the wall: ${d}`);
        assert.equal(await page.$eval("#wall-sketch [data-handle]", (c) => getComputedStyle(c).fill), "rgb(255, 255, 255)", "a white dot like the room resize handle");

        // swing the handle sideways in 2 px steps (about 4 degrees each): some angle lines up with the boundary
        const before = await areasNow(page), shot = await pixels(page);
        const heading = async () => { const [[ax, ay, bx, by]] = await chosenSegments(page); return Math.atan2(by - ay, bx - ax); };
        const was = await heading();
        const n = [(h[0] - foot[0]) / d, (h[1] - foot[1]) / d], t = [-n[1], n[0]];
        const labels = new Set();
        await page.mouse.move(...h);
        await page.mouse.down();
        const swing = [];
        for (let s = -4; s >= -28; s -= 2) swing.push(s);
        for (let s = -26; s <= 14; s += 2) swing.push(s);
        for (const s of swing) {
            await page.mouse.move(h[0] + t[0] * s, h[1] + t[1] * s);
            const label = await page.$eval("#wall-sketch text", (x) => (x.hasAttribute("hidden") ? "" : x.textContent));
            if (label) labels.add(label);
        }
        await page.mouse.up();
        assert.ok(labels.has("Parallel") || labels.has("Perpendicular"), `a lined-up turn shows its label: ${[...labels]}`);
        assert.notEqual(await pixels(page), shot, "the drawn walls change");
        // a turn about the middle moves one triangle out of a room and an equal one in, so the areas
        // may stay the same (exactly, when the two end walls are parallel); the wall's direction must not
        const turned = Math.abs(Math.atan2(Math.sin((await heading()) - was), Math.cos((await heading()) - was)));
        assert.ok(turned > 5 * Math.PI / 180, `the selected wall turned ${(turned * 180 / Math.PI).toFixed(1)} degrees`);
        await page.click("#undo");
        assert.deepEqual(await areasNow(page), before, "one Undo turns it back");

        await page.mouse.move(1100, 700);
        await page.keyboard.press("Escape");
        assert.equal(await selectedWall(page), null, "Escape clears the selection");
        assert.equal(await handle(page), null, "and hides the handle");

        // after a full run, preset C has a wall that ends where the boundary bends: it cannot be picked, so no handle
        await page.click("button[data-open=rooms]");
        await page.click("button[data-mode=shape_c]");
        await page.waitForFunction(() => { const s = document.getElementById("status").textContent; return s.startsWith("shape_c") && s.includes("ready"); });
        await page.click("button[data-open=optimize]");
        await page.click("#run");
        await page.waitForFunction(() => document.getElementById("run").getAttribute("aria-label") === "Pause");
        await page.waitForFunction(() => document.getElementById("run").getAttribute("aria-label") === "Run", null, { timeout: 120_000 });
        await page.click("button[data-open=adjust]");
        let locked = null;
        for (let y = 450; y <= 620 && !locked; y += 10) for (let x = 400; x <= 700 && !locked; x += 10) {
            await page.mouse.move(x, y);
            if (await page.$eval("#plan", (c) => c.hasAttribute("data-wall-locked"))) locked = [x, y];
        }
        assert.ok(locked, "a locked wall of preset C can be found");
        await page.mouse.down();
        await page.mouse.up();
        assert.equal(await selectedWall(page), null, "a locked wall is not selected");
        assert.equal(await handle(page), null, "and shows no handle");
    });
});
