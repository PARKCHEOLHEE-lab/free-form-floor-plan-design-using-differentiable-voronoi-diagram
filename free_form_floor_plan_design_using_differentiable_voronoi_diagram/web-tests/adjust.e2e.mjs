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
