// The demo page in a real browser: serve web/ (it must hold a wasm-pack build in web/pkg),
// run preset E to the end, switch the walls to Aligned, and read the aligned-wall readout.
// Run: cd web-tests && npm install && npm run e2e
import { test } from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const SHOTS = process.env.E2E_SHOTS; // optional folder for screenshots

const freePort = () =>
    new Promise((resolve) => {
        const s = createServer();
        s.listen(0, () => {
            const { port } = s.address();
            s.close(() => resolve(port));
        });
    });

async function serve() {
    const port = await freePort();
    const proc = spawn("python3", [fileURLToPath(new URL("../web/serve.py", import.meta.url)), String(port)], { stdio: "ignore" });
    const url = `http://localhost:${port}/`;
    for (let i = 0; i < 50; i++) {
        try {
            if ((await fetch(url)).ok) return { url, proc };
        } catch {}
        await new Promise((r) => setTimeout(r, 100));
    }
    proc.kill();
    throw new Error("serve.py did not start");
}

test("preset E run to the end: Aligned walls show a higher aligned-wall share than Raw", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width: 1400, height: 1000 } });
        await page.goto(url);
        await page.locator("#run").click();
        await page.waitForFunction(() => document.getElementById("status").textContent === "done", null, { timeout: 90_000 });
        if (SHOTS) await page.screenshot({ path: `${SHOTS}/raw.png` });

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
