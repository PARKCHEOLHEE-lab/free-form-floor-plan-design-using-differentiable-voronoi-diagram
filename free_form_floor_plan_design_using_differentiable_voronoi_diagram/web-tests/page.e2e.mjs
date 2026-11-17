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
