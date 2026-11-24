// The demo page in a real browser: serve web/ (it must hold a wasm-pack build in web/pkg),
// run preset E to the end, switch the walls to Aligned, and read the aligned-wall readout.
// Run: cd web-tests && npm install && npm run e2e
import { test } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { serve } from "./serve.mjs";
import { alignPlan, alignedShare } from "../web/plan/align.js";
import { fromCells } from "../web/plan/graph.js";

const SHOTS = process.env.E2E_SHOTS; // optional folder for screenshots

test("preset E run to the end: Aligned walls show a higher aligned-wall share than Raw", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
        await page.goto(url);
        await page.locator("#run").click();
        await page.waitForFunction(() => document.getElementById("status").textContent === "done", null, { timeout: 90_000 });
        if (SHOTS) await page.screenshot({ path: `${SHOTS}/raw.png` });

        // the wall switch lives in the Display panel, opened from the bottom-right toolbar
        await page.locator("button[data-open=display]").click();
        await page.locator("#walls button[data-walls=aligned]").click();
        const text = await page.locator("#align_v").textContent();
        if (SHOTS) await page.screenshot({ path: `${SHOTS}/aligned.png` });

        // "61% → 84%": aligned share of wall length before and after alignment
        const m = /(\d+)%\D+(\d+)%/.exec(text ?? "");
        assert.ok(m, `readout shows two shares: "${text}"`);
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
        await page.locator("button[data-open=display]").click();
        await page.locator("#walls button[data-walls=aligned]").click();
        const merge = page.getByRole("slider", { name: "Point merge" });
        const snap = page.getByRole("slider", { name: "Wall snap" });
        assert.equal(await merge.count(), 1, "Display has a point merge control");
        assert.equal(await snap.count(), 1, "Display has a wall snap control");
        assert.equal(await merge.inputValue(), "0.02");
        assert.equal(await snap.inputValue(), "0.02");

        const share = page.locator("#align_v");
        const originalShare = await share.textContent();
        const builds = await page.evaluate(() => window.buildCount);
        const pixels = () => page.locator("#plan").evaluate(canvas => canvas.toDataURL());
        const setTolerance = (input, value) => input.evaluate((el, current) => {
            el.value = current;
            el.dispatchEvent(new Event("input", { bubbles: true }));
        }, value);
        const originalPixels = await pixels();
        await setTolerance(merge, "0.04");
        assert.equal(await snap.inputValue(), "0.02");
        assert.notEqual(await pixels(), originalPixels, "raising the point tolerance redraws the wall geometry");
        // On E's first frame, raising only the point tolerance keeps the rounded share.
        assert.equal(await share.textContent(), originalShare);
        await merge.press("End");
        assert.equal(await page.locator("#merge_tol_v").textContent(), "1.000");
        await merge.press("Home");

        await setTolerance(snap, "0.04");
        assert.equal(await merge.inputValue(), "0.02");
        // The readout is the alignment of the same frame with merge 0.02 and snap 0.04,
        // the same steps the page takes (web/index.html alignFrame).
        const { frame, boundary } = await page.evaluate(() => ({ frame: window.lastFrame, boundary: window.lastBoundary }));
        const readout = (mergeTol, snapTol) => {
            const g = fromCells(frame.cells, boundary, { roomCount: new Set(frame.cells.map(c => c.room)).size, simplifyTol: 0.004 });
            const before = alignedShare(g, boundary);
            alignPlan(g, boundary, { mergeTol, snapTol });
            return `${Math.round(before * 100)}% → ${Math.round(alignedShare(g, boundary) * 100)}%`;
        };
        const expected = readout(0.02, 0.04);
        assert.notEqual(expected, readout(0.02, 0.02), "on this frame a 0.04 snap changes the share");
        assert.equal(await share.textContent(), expected);
        await snap.press("End");
        assert.equal(await page.locator("#align_tol_v").textContent(), "1.000");
        await snap.press("Home");
        assert.equal(await share.textContent(), originalShare);
        assert.equal(await page.evaluate(() => window.buildCount), builds, "the display changes without a new optimization");
    } finally {
        await browser.close();
        proc.kill();
    }
});
