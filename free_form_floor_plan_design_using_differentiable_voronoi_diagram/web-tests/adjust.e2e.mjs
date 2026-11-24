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
