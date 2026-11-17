// Page layout at real window sizes. 1145x820 is about the CSS viewport of a 14" MacBook
// Chrome window with a side panel open; 1440x900 is a common laptop screen.
// Run: cd web-tests && node --test layout.e2e.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { serve } from "./serve.mjs";

const DESKTOP = [[1145, 820], [1440, 900]];

async function measure(url, [width, height]) {
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width, height } });
        await page.goto(url);
        await page.waitForFunction(() => document.getElementById("status").textContent.includes("ready"), null, { timeout: 30_000 });
        return await page.evaluate(() => {
            const box = (el) => el.getBoundingClientRect();
            return {
                scrollHeight: document.scrollingElement.scrollHeight,
                run: box(document.getElementById("run")),
                canvas: box(document.getElementById("plan")),
                lossRows: [...document.querySelectorAll(".lossrow")].map(box),
            };
        });
    } finally {
        await browser.close();
    }
}

test("desktop: the page fits one screen, with Run, the canvas and every loss row in view", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        for (const [w, h] of DESKTOP) {
            const m = await measure(url, [w, h]);
            assert.ok(m.scrollHeight <= h, `${w}x${h}: page is ${m.scrollHeight}px tall, taller than the window`);
            const inView = (r) => r.top >= 0 && r.bottom <= h;
            assert.ok(inView(m.run), `${w}x${h}: Run is out of view`);
            assert.ok(inView(m.canvas), `${w}x${h}: canvas is out of view`);
            assert.equal(m.lossRows.length, 5);
            m.lossRows.forEach((r, i) => assert.ok(inView(r), `${w}x${h}: loss row ${i} is out of view (bottom ${r.bottom})`));
        }
    } finally {
        proc.kill();
    }
});
