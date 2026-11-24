// The demo page in a real browser: serve web/ (it must hold a wasm-pack build in web/pkg),
// run preset E to the end, switch the walls to Aligned, and read the aligned-wall share the plan reports.
// Run: cd web-tests && npm install && npm run e2e
import { test } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { serve } from "./serve.mjs";
import { alignPlan, alignedShare, cellWidth } from "../web/plan/align.js";
import { readFileSync } from "node:fs";
import { fromCells } from "../web/plan/graph.js";

const SHOTS = process.env.E2E_SHOTS; // optional folder for screenshots

test("preset E run to the end: Aligned walls show a higher aligned-wall share than Raw", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
        await page.goto(url);
        await page.locator("button[data-open=optimize]").click();
        await page.locator("#run").click();
        await page.waitForFunction(() => document.getElementById("status").textContent === "done", null, { timeout: 90_000 });
        if (SHOTS) await page.screenshot({ path: `${SHOTS}/raw.png` });

        // the wall switch sits in the Rooms title row
        await page.locator("button[data-open=rooms]").click();
        await page.locator("#walls button[data-walls=aligned]").click();
        const text = await page.$eval("#plan", (c) => c.dataset.alignedShare);
        if (SHOTS) await page.screenshot({ path: `${SHOTS}/aligned.png` });

        // "61% → 84%": aligned share of wall length before and after alignment
        const m = /(\d+)%\D+(\d+)%/.exec(text ?? "");
        assert.ok(m, `the plan reports two shares: "${text}"`);
        assert.ok(Number(m[2]) > Number(m[1]), `aligned share rose: "${text}"`);
    } finally {
        await browser.close();
        proc.kill();
    }
});

test("point and wall sliders redraw independently without restarting the plan", { timeout: 30_000 }, async () => {
    const { url, proc } = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
        await page.addInitScript(() => {
            window.buildCount = 0;
            const postMessage = Worker.prototype.postMessage;
            Worker.prototype.postMessage = function (message, ...args) {
                if (message.cmd === "build") window.buildCount++;
                return postMessage.call(this, message, ...args);
            };
            const Original = Worker;
            window.Worker = class extends Original {
                constructor(...args) {
                    super(...args);
                    this.addEventListener("message", ({ data }) => {
                        if (data.type === "frame") window.lastFrame = data.frame;
                        if (data.type === "built") window.lastBoundary = data.boundary;
                    });
                }
            };
        });
        await page.goto(url);
        await page.waitForFunction(() => document.getElementById("status").textContent.includes("ready"));
        await page.locator("button[data-open=rooms]").click();
        await page.locator("#walls button[data-walls=aligned]").click();
        const merge = page.getByRole("slider", { name: "Point merge" });
        const snap = page.getByRole("slider", { name: "Wall snap" });
        assert.equal(await merge.count(), 1, "Display has a point merge control");
        assert.equal(await snap.count(), 1, "Display has a wall snap control");
        assert.equal(await merge.inputValue(), "13");
        assert.equal(await snap.inputValue(), "13");

        const share = { textContent: () => page.$eval("#plan", (c) => c.dataset.alignedShare) };
        const originalShare = await share.textContent();
        const builds = await page.evaluate(() => window.buildCount);
        const pixels = () => page.locator("#plan").evaluate(canvas => canvas.toDataURL());
        const setTolerance = (input, value) => input.evaluate((el, current) => {
            el.value = current;
            el.dispatchEvent(new Event("input", { bubbles: true }));
        }, value);
        const originalPixels = await pixels();
        await setTolerance(merge, "30");
        assert.equal(await snap.inputValue(), "13");
        assert.notEqual(await pixels(), originalPixels, "raising the point tolerance redraws the wall geometry");
        // On E's first frame, raising only the point tolerance keeps the rounded share.
        assert.equal(await share.textContent(), originalShare);
        await merge.press("End");
        assert.equal(await page.locator("#merge_tol_v").textContent(), "50%");
        await setTolerance(merge, "13");   // back to the default (Home now goes to 0)

        await setTolerance(snap, "30");
        assert.equal(await merge.inputValue(), "13");
        // The reported share is the alignment of the same frame with merge 13 % and snap 30 % of the cell width,
        // the same steps the page takes (web/index.html alignFrame).
        const { frame, boundary } = await page.evaluate(() => ({ frame: window.lastFrame, boundary: window.lastBoundary }));
        const readout = (mergeTol, snapTol) => {
            const g = fromCells(frame.cells, boundary, { roomCount: new Set(frame.cells.map(c => c.room)).size, simplifyTol: 0.004 });
            const before = alignedShare(g, boundary);
            alignPlan(g, boundary, { mergeTol, snapTol });
            return `${Math.round(before * 100)}% → ${Math.round(alignedShare(g, boundary) * 100)}%`;
        };
        const cell = cellWidth(boundary, frame.sites.length);
        const expected = readout(0.13 * cell, 0.30 * cell);
        assert.notEqual(expected, readout(0.13 * cell, 0.13 * cell), "on this frame a 30 % snap changes the share");
        assert.equal(await share.textContent(), expected);
        await snap.press("End");
        assert.equal(await page.locator("#align_tol_v").textContent(), "50%");
        await setTolerance(snap, "13");   // back to the default (Home now goes to 0)
        assert.equal(await share.textContent(), originalShare);
        assert.equal(await page.evaluate(() => window.buildCount), builds, "the display changes without a new optimization");
    } finally {
        await browser.close();
        proc.kill();
    }
});

test("optimization: changing a loss weight or run setting rebuilds the step 0 preview and stops a run", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
        await page.addInitScript(() => {
            window.builds = [];
            const postMessage = Worker.prototype.postMessage;
            Worker.prototype.postMessage = function (message, ...args) {
                if (message.cmd === "build") window.builds.push(message);
                return postMessage.call(this, message, ...args);
            };
        });
        await page.goto(url);
        await page.waitForFunction(() => document.getElementById("status").textContent.includes("ready"));
        await page.click("button[data-open=optimize]");
        const set = (id, value) => page.locator("#" + id).evaluate((el, v) => {
            el.value = v;
            el.dispatchEvent(new Event("input", { bubbles: true }));
            el.dispatchEvent(new Event("change", { bubbles: true }));
        }, value);
        for (const [id, value, key] of [["w_cosine", "0.4", "cosineWeight"], ["iters", "300", "iters"], ["lr", "0.01", "lr"], ["w_area", "0.3", null]]) {
            const before = await page.evaluate(() => builds.length);
            await set(id, value);
            await page.waitForFunction((n) => builds.length > n && builds.at(-1).then === "preview", before);
            if (key) assert.equal(await page.evaluate((k) => builds.at(-1)[k], key), Number(value), `${id} reaches the new preview`);
        }
        await page.click("#run");
        await page.waitForFunction(() => document.getElementById("run").getAttribute("aria-label") === "Pause");
        const before = await page.evaluate(() => builds.length);
        await set("w_topo", "0.3");
        await page.waitForFunction((n) => builds.length > n && builds.at(-1).then === "preview", before);
        assert.equal(await page.locator("#run").getAttribute("aria-label"), "Run", "the run stops");
    } finally {
        await browser.close();
        proc.kill();
    }
});

/** Open preset E with the Aligned walls, run `fn(page, slider, exported)`, where exported() downloads the plan JSON. */
async function onAligned(fn) {
    const { url, proc } = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
        await page.goto(url);
        await page.waitForFunction(() => document.getElementById("status").textContent.includes("ready"));
        await page.locator("button[data-open=rooms]").click();
        await page.locator("#walls button[data-walls=aligned]").click();
        const slide = (id, value) => page.locator(`#${id}`).evaluate((el, v) => { el.value = v; el.dispatchEvent(new Event("input", { bubbles: true })); }, value);
        const exported = async () => {
            const [file] = await Promise.all([page.waitForEvent("download"), page.click(".toolbar [data-export]")]);
            return JSON.parse(readFileSync(await file.path(), "utf8"));
        };
        await fn(page, slide, exported);
    } finally {
        await browser.close();
        proc.kill();
    }
}
/** The corners of the exported frame aligned the way alignFrame does, with the given distances. */
function alignedCorners(data, mergeTol, snapTol) {
    const g = fromCells(data.frame.cells, data.boundary.ring, { roomCount: data.plan.rooms.length, simplifyTol: 0.004 });
    alignPlan(g, data.boundary.ring, { mergeTol, snapTol });
    return [...g.corners].map(([id, c]) => [id, c.x, c.y]);
}
const cornersOf = (data) => data.plan.corners.map((c) => [c.id, c.x, c.y]);

test("point merge: a slider from 0 to 50 percent of the cell width, 13 percent at first, that merges corners closer than that share of the cell width", { timeout: 60_000 }, async () => {
    await onAligned(async (page, slide, exported) => {
        const merge = page.getByRole("slider", { name: "Point merge" });
        assert.deepEqual(await merge.evaluate((el) => [el.min, el.max, el.step, el.value]), ["0", "50", "1", "13"]);
        assert.equal(await page.locator("#merge_tol_v").textContent(), "13%");
        await slide("merge_tol", "30");
        assert.equal(await page.locator("#merge_tol_v").textContent(), "30%");
        const data = await exported();
        const w = cellWidth(data.boundary.ring, data.frame.sites.length);
        const snap = data.settings.wallSnap / 100 * w;
        assert.deepEqual(cornersOf(data), alignedCorners(data, 0.30 * w, snap), "the plan merges within 30 percent of the cell width");
        assert.notDeepEqual(alignedCorners(data, 0.30 * w, snap), alignedCorners(data, 0.30, snap), "on this frame 30 percent of the cell width differs from a plain 0.30");
        assert.notDeepEqual(alignedCorners(data, 0.30 * w, snap), alignedCorners(data, 0.13 * w, snap), "and from the first 13 percent");
    });
});

test("wall snap: a slider from 0 to 50 percent of the cell width, 13 percent at first, that straightens wall pieces bent by less than that share of the cell width", { timeout: 60_000 }, async () => {
    await onAligned(async (page, slide, exported) => {
        const snap = page.getByRole("slider", { name: "Wall snap" });
        assert.deepEqual(await snap.evaluate((el) => [el.min, el.max, el.step, el.value]), ["0", "50", "1", "13"]);
        assert.equal(await page.locator("#align_tol_v").textContent(), "13%");
        await slide("align_tol", "30");
        assert.equal(await page.locator("#align_tol_v").textContent(), "30%");
        const data = await exported();
        const w = cellWidth(data.boundary.ring, data.frame.sites.length);
        const merge = data.settings.pointMerge / 100 * w;
        assert.deepEqual(cornersOf(data), alignedCorners(data, merge, 0.30 * w), "the plan snaps within 30 percent of the cell width");
        assert.notDeepEqual(alignedCorners(data, merge, 0.30 * w), alignedCorners(data, merge, 0.30), "on this frame 30 percent of the cell width differs from a plain 0.30");
        assert.notDeepEqual(alignedCorners(data, merge, 0.30 * w), alignedCorners(data, merge, 0.13 * w), "and from the first 13 percent");
    });
});
