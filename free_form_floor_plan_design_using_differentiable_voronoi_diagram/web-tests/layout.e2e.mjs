// The canvas UI at real window sizes: the plan canvas covers the window and the controls
// float over it as icon toolbars and panels. 1145x820 is about the CSS viewport of a 14"
// MacBook Chrome window with a side panel open; 1440x900 is a common laptop screen.
// Run: cd web-tests && node --test layout.e2e.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { serve } from "./serve.mjs";

const DESKTOP = [[1145, 820], [1440, 900]];

/** Open the page at one window size, wait for the first preview, and run `fn(page)`. */
async function atSize(url, [width, height], fn) {
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width, height } });
        await page.goto(url);
        await page.waitForFunction(() => document.getElementById("status").textContent.includes("ready"), null, { timeout: 30_000 });
        return await fn(page);
    } finally {
        await browser.close();
    }
}

test("desktop: the canvas covers the window, the page does not scroll, and there is no title text", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        for (const [w, h] of DESKTOP) {
            const m = await atSize(url, [w, h], (page) =>
                page.evaluate(() => {
                    const c = document.getElementById("plan").getBoundingClientRect();
                    return {
                        canvas: { left: c.left, top: c.top, width: c.width, height: c.height },
                        scrollWidth: document.scrollingElement.scrollWidth,
                        scrollHeight: document.scrollingElement.scrollHeight,
                        titleText: [...document.querySelectorAll("h1, .sub, .lede")].map((e) => e.textContent.trim()).join(" "),
                    };
                }),
            );
            const near = (a, b) => Math.abs(a - b) <= 1;
            assert.ok(near(m.canvas.left, 0) && near(m.canvas.top, 0), `${w}x${h}: canvas starts at ${m.canvas.left},${m.canvas.top}`);
            assert.ok(near(m.canvas.width, w) && near(m.canvas.height, h), `${w}x${h}: canvas is ${m.canvas.width}x${m.canvas.height}`);
            assert.ok(m.scrollWidth <= w && m.scrollHeight <= h, `${w}x${h}: page is ${m.scrollWidth}x${m.scrollHeight}`);
            assert.equal(m.titleText, "", `${w}x${h}: title text is still shown`);
        }
    } finally {
        proc.kill();
    }
});
