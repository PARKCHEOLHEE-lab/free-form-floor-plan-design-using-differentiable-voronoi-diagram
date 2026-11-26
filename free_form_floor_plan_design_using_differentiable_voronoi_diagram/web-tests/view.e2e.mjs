// The three.js view in a real browser: the isometric 3D default with flat walls, the plan's
// look from the top, the mouse, the zoom limits and the view gizmo.
// Run: cd web-tests && node --test view.e2e.mjs
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

test("seen from the top (the gizmo's +Z), the three.js plan keeps the 2D look: six pastel rooms and dark wall lines on screen", { timeout: 120_000 }, async () => {
    await withPage(async (page) => {
        assert.ok(await page.$eval("#plan", (c) => !!c.getContext("webgl2")), "the plan canvas is drawn with WebGL (three.js)");
        await page.click(".gizmo [data-axis='+z']");
        // The 2D canvas this replaces measured 31,183-52,466 px per room colour and 10,689 px of
        // dark wall line at 1145x820 (2026-11-17); about 60% of those is the floor here.
        const { rooms, dark } = await colourCounts(page, ROOM_COLOURS);
        rooms.forEach((n, r) => assert.ok(n > 20_000, `room ${r} colour covers only ${n} px`));
        assert.ok(dark > 6_000, `dark wall lines cover only ${dark} px`);
    });
});

/** The view's camera state, from the canvas's data attributes. */
const camera = (page) =>
    page.$eval("#plan", (c) => ({ zoom: +c.dataset.zoom, azimuth: +c.dataset.azimuth, elevation: +c.dataset.elevation }));

/** Centre of the drawn (non-white) pixels on the plan canvas, in canvas pixels. */
const drawnCentre = (page) =>
    page.evaluate(async () => {
        const img = new Image();
        img.src = document.getElementById("plan").toDataURL();
        await img.decode();
        const c = document.createElement("canvas");
        c.width = img.width;
        c.height = img.height;
        const ctx = c.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const px = ctx.getImageData(0, 0, c.width, c.height).data;
        let sx = 0, sy = 0, n = 0;
        for (let i = 0; i < px.length; i += 4)
            if (px[i] < 250 || px[i + 1] < 250 || px[i + 2] < 250) {
                const p = i / 4;
                sx += p % c.width;
                sy += Math.floor(p / c.width);
                n++;
            }
        return [sx / n, sy / n];
    });

/** Drag on open canvas (away from the toolbar, the loss card and the gizmo) with one button. */
async function drag(page, button, [dx, dy], { shift = false } = {}) {
    if (shift) await page.keyboard.down("Shift");
    await page.mouse.move(600, 400);
    await page.mouse.down({ button });
    await page.mouse.move(600 + dx, 400 + dy, { steps: 10 });
    await page.mouse.up({ button });
    if (shift) await page.keyboard.up("Shift");
}

test("mouse, as in Rhino: a left drag leaves the view alone, a right drag rotates, Shift and a right drag pans without turning, and the wheel zooms but stops at 0.5x and 8x", { timeout: 120_000 }, async () => {
    await withPage(async (page) => {
        const start = await camera(page);
        const startCentre = await drawnCentre(page);
        await drag(page, "left", [100, 0]);
        assert.deepEqual(await camera(page), start, "a left drag moved the view");
        assert.deepEqual(await drawnCentre(page), startCentre, "a left drag moved the plan");

        // Shift + right drag 100 px left and 50 px up: the plan follows the pointer
        await drag(page, "right", [-100, -50], { shift: true });
        const after = await drawnCentre(page);
        const [dx, dy] = [after[0] - startCentre[0], after[1] - startCentre[1]];
        assert.ok(Math.abs(dx + 100) < 3 && Math.abs(dy + 50) < 3, `the plan moved by ${dx.toFixed(1)}, ${dy.toFixed(1)} px`);
        const panned = await camera(page);
        assert.ok(panned.azimuth === start.azimuth && panned.elevation === start.elevation, `the pan turned the view to ${JSON.stringify(panned)}`);

        await drag(page, "right", [150, 0]);
        const turned = await camera(page);
        assert.ok(Math.abs(turned.azimuth - start.azimuth) > 30, `a 150 px right drag turned the view only to azimuth ${turned.azimuth}`);

        // each wheel step of 200 zooms by 1/0.95^2 (OrbitControls), so 40 steps would pass 60x
        for (let i = 0; i < 40; i++) await page.mouse.wheel(0, -200);
        assert.equal((await camera(page)).zoom, 8, "zoom in stops at 8x");
        for (let i = 0; i < 80; i++) await page.mouse.wheel(0, 200);
        assert.equal((await camera(page)).zoom, 0.5, "zoom out stops at 0.5x");
    });
});

test("mouse, as in CAD: a middle drag with the wheel button pans, so the plan follows the pointer without turning or zooming", { timeout: 120_000 }, async () => {
    await withPage(async (page) => {
        const start = await camera(page);
        const startCentre = await drawnCentre(page);
        await drag(page, "middle", [-100, -50]);
        const after = await drawnCentre(page);
        const [dx, dy] = [after[0] - startCentre[0], after[1] - startCentre[1]];
        assert.ok(Math.abs(dx + 100) < 3 && Math.abs(dy + 50) < 3, `the plan moved by ${dx.toFixed(1)}, ${dy.toFixed(1)} px`);
        assert.deepEqual(await camera(page), start, "a middle drag turned or zoomed the view");
    });
});

/** Pixels on the plan canvas in the light neutral greys of the 3D wall sides (205-250). */
const wallSidePixels = (page) =>
    page.evaluate(async () => {
        const img = new Image();
        img.src = document.getElementById("plan").toDataURL();
        await img.decode();
        const c = document.createElement("canvas");
        c.width = img.width;
        c.height = img.height;
        const ctx = c.getContext("2d");
        ctx.drawImage(img, 0, 0);
        const px = ctx.getImageData(0, 0, c.width, c.height).data;
        let n = 0;
        for (let i = 0; i < px.length; i += 4) {
            const [r, g, b] = [px[i], px[i + 1], px[i + 2]];
            if (Math.max(r, g, b) - Math.min(r, g, b) <= 2 && r >= 205 && r <= 250) n++;
        }
        return n;
    });

const near = (a, b, tol = 0.01) => Math.abs(a - b) <= tol;

test("3D by default: the plan opens in isometric with its walls flat on the floor, there is no 3D icon, and a double-click returns to isometric", { timeout: 120_000 }, async () => {
    await withPage(async (page) => {
        let c = await camera(page);
        assert.ok(near(c.azimuth, -45) && near(c.elevation, 35.26) && c.zoom === 1, `the view opens at ${JSON.stringify(c)}`);
        // measured at 1145x820 on 2026-11-17: standing walls gave 71,560-75,086 such px in
        // isometric, flat walls 4,352-4,894 (anti-aliased wall lines)
        const greys = await wallSidePixels(page);
        assert.ok(greys < 20_000, `${greys} px of wall-side grey: the walls stand up`);
        assert.equal(await page.$("button[data-toggle='3d']"), null, "the 3D icon is still in the toolbar");

        for (let i = 0; i < 5; i++) await page.mouse.wheel(0, -200);
        await page.click(".gizmo [data-axis='+x']");
        c = await camera(page);
        assert.ok(c.zoom > 1.5 && near(c.azimuth, 90), `the wheel and +X left ${JSON.stringify(c)}`);
        await page.mouse.dblclick(600, 400);
        c = await camera(page);
        assert.ok(near(c.azimuth, -45) && near(c.elevation, 35.26) && c.zoom === 1, `double-click left ${JSON.stringify(c)}`);
    });
});

/** Each gizmo axis end's centre, in px from the gizmo's centre (x right, y down). */
const gizmoEnds = (page) =>
    page.$$eval(".gizmo [data-axis]", (ends) => {
        const g = document.querySelector(".gizmo").getBoundingClientRect();
        const centre = (r) => [r.left + r.width / 2, r.top + r.height / 2];
        const [cx, cy] = centre(g);
        return Object.fromEntries(ends.map((e) => [e.dataset.axis, centre(e.getBoundingClientRect()).map((v, i) => v - [cx, cy][i])]));
    });

test("gizmo: it sits in the top-right corner, its axes follow the camera, clicking +X, -Y or +Z turns the view, and an axis end names its view on hover", { timeout: 180_000 }, async () => {
    for (const size of [[1145, 820], [600, 900]])
        await withPage(async (page) => {
            const g = await page.$eval(".gizmo", (e) => ({ ...e.getBoundingClientRect().toJSON(), w: innerWidth }));
            const corner = Math.abs(g.top - 32) < 1 && Math.abs(g.w - g.right - 32) < 1;
            const small = g.width >= 48 && g.width <= 120 && g.height >= 48 && g.height <= 120;
            assert.ok(corner && small, `${size}: gizmo at ${g.left},${g.top}-${g.right},${g.bottom}`);
        }, size);

    await withPage(async (page) => {
        let e = await gizmoEnds(page);
        assert.deepEqual(Object.keys(e).sort(), ["+x", "+y", "+z", "-x", "-y"]);
        assert.ok(e["+z"][1] < -10 && Math.abs(e["+z"][0]) < 2, `isometric: +Z at ${e["+z"]}`);
        assert.ok(e["+x"][0] > 5 && e["+x"][1] < 0 && e["+y"][0] < -5 && e["+y"][1] < 0, `isometric: +X at ${e["+x"]}, +Y at ${e["+y"]}`);
        await page.click(".gizmo [data-axis='+z']");
        assert.deepEqual(await camera(page), { zoom: 1, azimuth: 0, elevation: 90 });
        e = await gizmoEnds(page);
        assert.ok(e["+x"][0] > 10 && Math.abs(e["+x"][1]) < 2, `top view: +X at ${e["+x"]}`);
        assert.ok(e["+y"][1] < -10 && Math.abs(e["+y"][0]) < 2, `top view: +Y at ${e["+y"]}`);
        assert.ok(Math.hypot(...e["+z"]) < 2, `top view: +Z at ${e["+z"]}`);

        await page.click(".gizmo [data-axis='+x']");
        let c = await camera(page);
        assert.ok(near(c.azimuth, 90) && near(c.elevation, 35.26), `+X gives ${JSON.stringify(c)}`);
        await page.click(".gizmo [data-axis='-y']");
        c = await camera(page);
        assert.ok(near(c.azimuth, 0) && near(c.elevation, 35.26), `-Y gives ${JSON.stringify(c)}`);
        await page.click(".gizmo [data-axis='+z']");
        assert.deepEqual(await camera(page), { zoom: 1, azimuth: 0, elevation: 90 });

        await page.hover(".gizmo [data-axis='+x']");
        await page.waitForTimeout(300);
        const tip = await page.$eval(".gizmo [data-axis='+x']", (b) => {
            const s = getComputedStyle(b, "::after");
            return { text: s.content, opacity: +s.opacity };
        });
        assert.deepEqual(tip, { text: '"View from east"', opacity: 1 });
    });
});

test("gizmo drag: the gizmo has no background or border, a drag inside it rotates the view (also one that starts on an axis end), and a plain click on an end still turns the view", { timeout: 120_000 }, async () => {
    await withPage(async (page) => {
        const look = await page.$eval(".gizmo", (g) => {
            const s = getComputedStyle(g);
            return { background: s.backgroundColor, shadow: s.boxShadow, border: s.borderTopStyle };
        });
        assert.deepEqual(look, { background: "rgba(0, 0, 0, 0)", shadow: "none", border: "none" });

        const g = await page.$eval(".gizmo", (e) => e.getBoundingClientRect().toJSON());
        const [cx, cy] = [g.left + g.width / 2, g.top + g.height / 2];
        const gizmoDrag = async ([x, y], [dx, dy]) => {
            await page.mouse.move(x, y);
            await page.mouse.down();
            await page.mouse.move(x + dx, y + dy, { steps: 10 });
            await page.mouse.up();
        };
        // in isometric no axis end sits at the centre
        const start = await camera(page);
        await gizmoDrag([cx, cy], [60, 0]);
        let c = await camera(page);
        assert.ok(Math.abs(c.azimuth - start.azimuth) > 15, `a 60 px drag in the gizmo turned the view to azimuth ${c.azimuth}`);

        const e = await gizmoEnds(page);
        const before = c;
        await gizmoDrag([cx + e["+x"][0], cy + e["+x"][1]], [0, 40]);
        c = await camera(page);
        assert.ok(Math.abs(c.elevation - before.elevation) > 5, `a drag from the +X end left the elevation at ${c.elevation}`);
        assert.ok(!near(c.azimuth, 90), "a drag that started on +X turned the view to the east instead of rotating it");

        await page.click(".gizmo [data-axis='-y']");
        c = await camera(page);
        assert.ok(near(c.azimuth, 0) && near(c.elevation, 35.26), `a plain click on -Y gives ${JSON.stringify(c)}`);
    });
});
