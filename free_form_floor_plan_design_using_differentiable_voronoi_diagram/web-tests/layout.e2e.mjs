// The canvas UI at real window sizes: the plan canvas covers the window and the controls
// float over it as icon toolbars and panels. 1145x820 is about the CSS viewport of a 14"
// MacBook Chrome window with a side panel open; 1440x900 is a common laptop screen.
// Run: cd web-tests && node --test layout.e2e.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { serve } from "./serve.mjs";

const DESKTOP = [[1145, 820], [1440, 900]];

test('floating controls keep 32px edge spacing and fit desktop and narrow windows',{timeout:120000},async()=>{
    const {url,proc}=await serve();
    try{
        for(const size of [[1440,900],[600,900],[390,720]])await atSize(url,size,async page=>{
            const rect=selector=>page.$eval(selector,e=>e.getBoundingClientRect().toJSON());
            const toolbar=await rect('.toolbar'),gizmo=await rect('.gizmo');
            const near=(a,b)=>Math.abs(a-b)<1;
            assert.ok(near(toolbar.left,32)&&near(toolbar.top,32),'toolbar has 32px outer spacing');
            assert.ok(near(gizmo.top,32)&&near(size[0]-gizmo.right,32),'gizmo has matching top/right spacing');
            for(const name of ['boundary','rooms','optimize','display']){
                await page.click(`button[data-open=${name}]`);
                const panel=await rect(`.panel[data-panel=${name}]`);
                assert.ok(near(panel.top,32)&&panel.left>=toolbar.right&&panel.left-toolbar.right<=16,`${name} remains beside its toolbar`);
                assert.ok(panel.right<=size[0]-31&&panel.bottom<=size[1]-31,`${name} stays inside the outer spacing`);
                await page.click(`button[data-open=${name}]`);
            }
            await page.click('button[data-open=rooms]');
            const dir=new URL('../../.agents/cache/screens/',import.meta.url);await mkdir(dir,{recursive:true});
            await page.screenshot({path:fileURLToPath(new URL(`ui-spacing-${size[0]}.png`,dir))});
        });
    }finally{proc.kill();}
});

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
            await page.click("button[data-open=optimize]");
            assert.deepEqual(await openPanels(page), ["optimize"]);
            await page.click("button[data-open=optimize]");
            assert.deepEqual(await openPanels(page), []);
            await page.click("button[data-open=display]");
            assert.deepEqual(await openPanels(page), ["display"]);
        });
    } finally {
        proc.kill();
    }
});

test("one toolbar: every icon sits in the top-left toolbar, and the Display panel opens beside it", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1440, 900], async (page) => {
            const bars = await page.$$eval(".toolbar", (ts) => ts.map((t) => t.getBoundingClientRect()));
            assert.equal(bars.length, 1, "one toolbar");
            assert.ok(bars[0].left < 40 && bars[0].top < 40, `toolbar at ${bars[0].left},${bars[0].top}`);
            for (const sel of ["button[data-open=display]", "button[data-open=optimize]", "#run", "#reset"])
                assert.equal(await page.$eval(sel, (b) => !!b.closest(".toolbar")), true, `${sel} is in the toolbar`);
            await page.click("button[data-open=display]");
            const panel = await page.$eval(".panel[data-panel=display]", (p) => p.getBoundingClientRect().toJSON());
            const tool = await page.$eval("button[data-open=display]", (b) => b.getBoundingClientRect().toJSON());
            assert.ok(panel.left >= bars[0].right && panel.left - bars[0].right <= 16, `Display panel starts at ${panel.left}, toolbar ends at ${bars[0].right}`);
            assert.ok(panel.top <= tool.top, `Display panel top ${panel.top} is below its icon ${tool.top}`);
        });
    } finally {
        proc.kill();
    }
});

test("optimization panel: the plan drawing stays the same, pixel for pixel", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const pixels = () => page.$eval("#plan", (c) => c.toDataURL());
            const shown = await pixels();
            await page.click("button[data-open=optimize]");
            assert.ok((await pixels()) === shown, "the plan changed when the Optimization panel opened");
            await page.click("button[data-open=optimize]");
            assert.ok((await pixels()) === shown, "the plan changed when the Optimization panel closed");
        });
    } finally {
        proc.kill();
    }
});

test("panels: an open panel is see-through, so the plan shows behind it", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            for (const name of ["boundary", "rooms", "optimize", "display"]) {
                await page.click(`button[data-open=${name}]`);
                const alpha = await page.$eval(`.panel[data-panel=${name}]`, (p) => {
                    const m = getComputedStyle(p).backgroundColor.match(/rgba?\(([^)]+)\)/);
                    const parts = m ? m[1].split(",").map(Number) : [];
                    return parts.length === 4 ? parts[3] : 1;
                });
                assert.ok(alpha >= 0.5 && alpha <= 0.85, `${name} panel background alpha is ${alpha}`);
                await page.click(`button[data-open=${name}]`);
            }
        });
    } finally {
        proc.kill();
    }
});

test("tooltips: hovering a toolbar icon shows its name to its right within 0.3 s, Run's name follows Pause, and no browser title tooltip remains", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const tip = (el) =>
                el.evaluate((b) => {
                    const s = getComputedStyle(b, "::after");
                    return { text: s.content, opacity: +s.opacity, left: parseFloat(s.left), width: b.getBoundingClientRect().width };
                });
            const tools = await page.$$(".tool");
            assert.equal(tools.length, 6);
            for (const tool of tools) {
                const name = await tool.getAttribute("aria-label");
                assert.equal((await tip(tool)).opacity, 0, `${name}: the tooltip shows before hover`);
                await tool.hover();
                await page.waitForTimeout(300);
                const t = await tip(tool);
                assert.equal(t.text, JSON.stringify(name), `${name}: tooltip reads ${t.text}`);
                assert.equal(t.opacity, 1, `${name}: tooltip opacity ${t.opacity} after 0.3 s`);
                assert.ok(t.left >= t.width, `${name}: tooltip starts at ${t.left}px, inside the ${t.width}px button`);
                assert.equal(await tool.getAttribute("title"), null, `${name}: the browser's own title tooltip is still set`);
            }
            await page.click("#run");
            await page.waitForTimeout(300);
            assert.equal((await tip(await page.$("#run"))).text, '"Pause"', "Run's tooltip follows its label while running");
        });
    } finally {
        proc.kill();
    }
});

test("narrow: at 600px the toolbar and each open panel stay inside the window", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [600, 900], async (page) => {
            const outside = (sel) =>
                page.$$eval(sel, (es) =>
                    es
                        .filter((e) => e.getClientRects().length > 0)
                        .map((e) => [e.className, e.getBoundingClientRect()])
                        .filter(([, b]) => b.left < 0 || b.top < 0 || b.right > innerWidth || b.bottom > innerHeight)
                        .map(([name, b]) => `${name} ${Math.round(b.left)},${Math.round(b.top)}-${Math.round(b.right)},${Math.round(b.bottom)}`),
                );
            assert.deepEqual(await outside(".toolbar"), []);
            for (const name of ["boundary", "rooms", "optimize", "display"]) {
                await page.click(`button[data-open=${name}]`);
                assert.deepEqual(await outside(".panel"), [], `${name} panel`);
                await page.click(`button[data-open=${name}]`);
            }
        });
    } finally {
        proc.kill();
    }
});

test("optimization: one icon opens the loss weights, the run settings and the six-row loss graph", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const labels = await page.$$eval(".toolbar .tool", (ts) => ts.map((t) => t.getAttribute("aria-label")));
            assert.deepEqual(labels, ["Run", "Reset", "Boundary", "Rooms", "Optimization", "Display"]);
            assert.equal(await page.locator(".losscard").count(), 0, "no separate loss card");
            assert.equal(await page.locator(".panel[data-panel=weights]").count(), 0, "no separate weights panel");
            for (const id of ["w_wall", "w_cosine", "w_area", "w_lloyd", "w_topo", "iters", "lr", "seed", "losspanel"])
                assert.equal(await page.$eval(`#${id}`, (e) => e.closest(".panel")?.dataset.panel ?? null), "optimize", `#${id} is in the Optimization panel`);
            const order = await page.$eval(".panel[data-panel=optimize]", (p) => [...p.querySelectorAll("#w_topo, #iters, #losspanel")].map((e) => e.id));
            assert.deepEqual(order, ["w_topo", "iters", "losspanel"], "weights, then run settings, then the graph");
            const rows = () =>
                page.$$eval(".lossrow", (rs) =>
                    rs.map((r) => {
                        const b = r.getBoundingClientRect();
                        return b.width > 0 && b.top >= 0 && b.bottom <= innerHeight;
                    }),
                );
            assert.deepEqual(await rows(), [false, false, false, false, false, false], "the graph is hidden while the panel is closed");
            await page.click("button[data-open=optimize]");
            assert.deepEqual(await rows(), [true, true, true, true, true, true], "all 6 rows are in view in the open panel");
        });
    } finally {
        proc.kill();
    }
});
