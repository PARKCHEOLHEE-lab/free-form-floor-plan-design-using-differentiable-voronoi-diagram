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

/** The panels on screen, by name. */
const openPanels = (page) =>
    page.$$eval(".panel", (ps) => ps.filter((p) => p.getClientRects().length > 0).map((p) => p.dataset.panel));

test("toolbar: an icon opens only its panel, another icon swaps it, the same icon closes it", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1440, 900], async (page) => {
            assert.deepEqual(await openPanels(page), [], "no panel is open at the start");
            await page.click("button[data-open=rooms]");
            assert.deepEqual(await openPanels(page), ["rooms"]);
            await page.click("button[data-open=weights]");
            assert.deepEqual(await openPanels(page), ["weights"]);
            await page.click("button[data-open=weights]");
            assert.deepEqual(await openPanels(page), []);
            await page.click("button[data-open=display]");
            assert.deepEqual(await openPanels(page), ["display"]);
        });
    } finally {
        proc.kill();
    }
});

test("loss card: its 5 rows are in view at the start, and the Losses icon hides and shows it", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const rows = () =>
                page.$$eval(".lossrow", (rs) =>
                    rs.map((r) => {
                        const b = r.getBoundingClientRect();
                        return b.width > 0 && b.top >= 0 && b.bottom <= innerHeight;
                    }),
                );
            assert.deepEqual(await rows(), [true, true, true, true, true], "all 5 rows in view at the start");
            await page.click("button[data-toggle=losses]");
            assert.deepEqual(await rows(), [false, false, false, false, false], "hidden after one click");
            await page.click("button[data-toggle=losses]");
            assert.deepEqual(await rows(), [true, true, true, true, true], "shown again after a second click");
        });
    } finally {
        proc.kill();
    }
});
