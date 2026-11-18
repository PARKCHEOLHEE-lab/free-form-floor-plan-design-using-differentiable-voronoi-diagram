// The demo page in a real browser: serve web/ (it must hold a wasm-pack build in web/pkg),
// run preset E to the end, switch the walls to Aligned, and read the aligned-wall readout.
// Run: cd web-tests && npm install && npm run e2e
import { test } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { serve } from "./serve.mjs";

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

test("alignment tolerance redraws the current plan without restarting it", { timeout: 30_000 }, async () => {
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
        });
        await page.goto(url);
        await page.waitForFunction(() => document.getElementById("status").textContent.includes("ready"));
        await page.locator("button[data-open=display]").click();
        await page.locator("#walls button[data-walls=aligned]").click();
        const control = page.getByRole("slider", { name: "Alignment tolerance" });
        assert.equal(await control.count(), 1, "Display lets users change the alignment tolerance");
        assert.equal(await control.inputValue(), "0.02", "keep the original tolerance on load");

        const share = page.locator("#align_v");
        const originalShare = await share.textContent();
        const builds = await page.evaluate(() => window.buildCount);
        await control.press("End");
        assert.equal(await control.inputValue(), "0.04");
        assert.equal(await page.locator("#align_tol_v").textContent(), "0.040");
        assert.notEqual(await share.textContent(), originalShare, "the raised tolerance changes the aligned geometry");
        await control.press("Home");
        assert.equal(await share.textContent(), originalShare, "the lower tolerance restores the original alignment");
        assert.equal(await page.evaluate(() => window.buildCount), builds, "the display changes without a new optimization");
    } finally {
        await browser.close();
        proc.kill();
    }
});
