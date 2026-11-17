// The three.js view in a real browser: the plan's look in the top view, zoom and pan limits,
// and the isometric 3D view. Run: cd web-tests && node --test view.e2e.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { serve } from "./serve.mjs";

/** Open the page, wait for the first preview, and run `fn(page)`. */
async function withPage(fn, [width, height] = [1145, 820]) {
    const { url, proc } = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width, height } });
        await page.goto(url);
        await page.waitForFunction(() => document.getElementById("status").textContent.includes("ready"), null, { timeout: 30_000 });
        return await fn(page);
    } finally {
        await browser.close();
        proc.kill();
    }
}

/** Pixel counts on the plan canvas: near each given colour, and near the dark wall colour. */
const colourCounts = (page, hexes) =>
    page.evaluate(async (hexes) => {
        const src = document.getElementById("plan");
        const img = new Image();
        img.src = src.toDataURL();
        await img.decode();
        const c = document.createElement("canvas");
        c.width = img.width;
        c.height = img.height;
        const ctx = c.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const px = ctx.getImageData(0, 0, c.width, c.height).data;
        const rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
        const targets = [...hexes.map(rgb), [17, 17, 17]];
        const counts = targets.map(() => 0);
        for (let i = 0; i < px.length; i += 4)
            targets.forEach((t, k) => {
                if (Math.abs(px[i] - t[0]) + Math.abs(px[i + 1] - t[1]) + Math.abs(px[i + 2] - t[2]) <= 12) counts[k]++;
            });
        return { rooms: counts.slice(0, -1), dark: counts.at(-1) };
    }, hexes);

// preset E's six rooms, in room order (index.html COLORS)
const ROOM_COLOURS = ["#bfe4bf", "#ded7ea", "#fee0c2", "#ffffcc", "#9cb6d8", "#f881bf"];

test("three.js top view keeps the 2D look: six pastel rooms and dark wall lines on screen", { timeout: 120_000 }, async () => {
    await withPage(async (page) => {
        assert.ok(await page.$eval("#plan", (c) => !!c.getContext("webgl2")), "the plan canvas is drawn with WebGL (three.js)");
        // The 2D canvas this replaces measured 31,183-52,466 px per room colour and 10,689 px of
        // dark wall line at 1145x820 (2026-11-17); about 60% of those is the floor here.
        const { rooms, dark } = await colourCounts(page, ROOM_COLOURS);
        rooms.forEach((n, r) => assert.ok(n > 20_000, `room ${r} colour covers only ${n} px`));
        assert.ok(dark > 6_000, `dark wall lines cover only ${dark} px`);
    });
});
