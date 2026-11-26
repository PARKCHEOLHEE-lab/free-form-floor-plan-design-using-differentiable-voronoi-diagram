// The canvas UI at real window sizes: the plan canvas covers the window and the controls
// float over it as icon toolbars and panels. 1145x820 is about the CSS viewport of a 14"
// MacBook Chrome window with a side panel open; 1440x900 is a common laptop screen.
// Run: cd web-tests && node --test layout.e2e.mjs
import { test } from "node:test";
import assert from "node:assert/strict";
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import { readFileSync } from "node:fs";
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
            for(const name of ['rooms','optimize']){
                await page.click(`button[data-open=${name}]`);
                const panel=await rect(`.panel[data-panel=${name}]`);
                const plan=await rect('#plan');
                if (size[0] <= 700) {
                    assert.ok(near(panel.left,32)&&near(size[1]-panel.bottom,32)&&panel.top>=plan.bottom,`${name} docks below the plan`);
                } else {
                    assert.ok(near(panel.top,32)&&panel.left>=toolbar.right&&panel.left-toolbar.right<=16,`${name} remains beside its toolbar`);
                    assert.equal(plan.left, 0, `${name} floats over the plan`);
                    assert.equal(plan.width, size[0], `${name} keeps the full plan width`);
                }
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
        });
    } finally {
        proc.kill();
    }
});

test("one toolbar: every icon sits in the top-left toolbar, and the Rooms panel opens beside it", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1440, 900], async (page) => {
            const bars = await page.$$eval(".toolbar", (ts) => ts.map((t) => t.getBoundingClientRect()));
            assert.equal(bars.length, 1, "one toolbar");
            assert.ok(bars[0].left < 40 && bars[0].top < 40, `toolbar at ${bars[0].left},${bars[0].top}`);
            for (const sel of ["button[data-open=rooms]", "button[data-open=optimize]"])
                assert.equal(await page.$eval(sel, (b) => !!b.closest(".toolbar")), true, `${sel} is in the toolbar`);
            await page.click("button[data-open=rooms]");
            const panel = await page.$eval(".panel[data-panel=rooms]", (p) => p.getBoundingClientRect().toJSON());
            const tool = await page.$eval("button[data-open=rooms]", (b) => b.getBoundingClientRect().toJSON());
            assert.ok(panel.left >= bars[0].right && panel.left - bars[0].right <= 16, `Rooms panel starts at ${panel.left}, toolbar ends at ${bars[0].right}`);
            assert.ok(panel.top <= tool.top, `Rooms panel top ${panel.top} is below its icon ${tool.top}`);
        });
    } finally {
        proc.kill();
    }
});

test("floating panels keep the full panned plan in place", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        for (const [width, height] of [[1273, 822], [768, 1024]]) await atSize(url, [width, height], async (page) => {
            const rooms = page.getByRole("button", { name: "Rooms", exact: true });
            await rooms.click();
            await page.getByRole("button", { name: "Boundary C", exact: true }).click();
            await page.waitForFunction(() => /^shape_c.*ready/.test(document.querySelector("#status").textContent));
            await page.getByRole("checkbox", { name: "Aligned walls", exact: true }).check();
            await page.getByRole("slider", { name: "Point merge", exact: true }).press("End");
            await rooms.click();
            await page.mouse.move(width * 0.73, height * 0.51);
            await page.mouse.down({ button: "middle" });
            await page.mouse.move(width * 0.56, height * 0.64, { steps: 12 });
            await page.mouse.up({ button: "middle" });

            const state = () => page.$eval("#plan", c => ({
                pose: [c.dataset.azimuth, c.dataset.elevation, c.dataset.zoom],
                bounds: c.getBoundingClientRect().toJSON(),
                pixels: c.toDataURL(),
            }));
            const before = await state();
            for (const name of ["Rooms", "Optimization"]) {
                const toggle = page.getByRole("button", { name, exact: true });
                await toggle.click();
                const opened = await state();
                assert.deepEqual(opened.bounds, before.bounds, name + " keeps the full canvas");
                assert.deepEqual(opened.pose, before.pose);
                assert.ok(opened.pixels === before.pixels, name + " leaves the drawing in the same place");
                assert.equal(await page.locator("#room-tags").evaluate(el => getComputedStyle(el).clipPath), "inset(0px)");
                await toggle.click();
                assert.ok((await state()).pixels === before.pixels, "closing the panel also keeps the drawing in place");
            }
        });
    } finally {
        proc.kill();
    }
});

test("panels: an open panel is see-through, so the plan shows behind it", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            for (const name of ["rooms", "optimize"]) {
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
            const tools = await page.$$(".toolbar .tool");
            assert.equal(tools.length, 4);
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
            await page.click("button[data-open=optimize]");
            await page.hover("#run");
            await page.click("#run");
            await page.waitForTimeout(300);
            assert.equal(await page.locator(".tip").textContent(), "Pause", "Run's tooltip follows its label while running");
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
            for (const name of ["rooms", "optimize"]) {
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
            assert.deepEqual(labels, ["Rooms", "Optimization", "Adjustment", "Export JSON"]);
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

test("optimization: after a run with the panel closed, opening it puts the playhead and value tags in the graph", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=optimize]");
            await page.click("#run");
            await page.click("button[data-open=optimize]"); // close the panel while it runs
            await page.waitForFunction(() => +(/iter (\d+)/.exec(document.getElementById("tick").textContent)?.[1] ?? 0) >= 20);
            await page.click("button[data-open=optimize]");
            await page.click("#run"); // pause
            await page.waitForTimeout(300);
            const m = await page.evaluate(() => {
                const box = (el) => el.getBoundingClientRect();
                const first = box(document.getElementById("sp_total").closest(".spark"));
                const last = box(document.getElementById("sp_topo").closest(".spark"));
                const line = box(document.getElementById("playhead"));
                const tag = box(document.getElementById("vt_total"));
                return { first: first.toJSON(), last: last.toJSON(), line: line.toJSON(), tag: tag.toJSON(),
                    text: document.getElementById("vt_total").textContent };
            });
            const near = (a, b) => Math.abs(a - b) <= 1;
            assert.ok(near(m.line.top, m.first.top) && near(m.line.bottom, m.last.bottom), `the line spans the graph rows: ${JSON.stringify(m.line)}`);
            assert.ok(m.line.left > m.first.left && m.line.left < m.first.right, `the line sits inside the graph: ${m.line.left} in ${m.first.left}..${m.first.right}`);
            assert.ok(m.text.length > 0, "the total tag shows a value");
            assert.ok(near((m.tag.top + m.tag.bottom) / 2, (m.first.top + m.first.bottom) / 2), "the total tag sits on its row");
            assert.ok(m.tag.left >= m.first.left - 1 && m.tag.right <= m.first.right + 1, `the total tag stays inside the graph: ${m.tag.left}..${m.tag.right}`);
        });
    } finally {
        proc.kill();
    }
});

test("rooms: one icon opens the boundary choice above the room bubbles", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const labels = await page.$$eval(".toolbar .tool", (ts) => ts.map((t) => t.getAttribute("aria-label")));
            assert.deepEqual(labels, ["Rooms", "Optimization", "Adjustment", "Export JSON"]);
            assert.equal(await page.locator(".panel[data-panel=boundary]").count(), 0, "no separate boundary panel");
            for (const sel of ["#presets", "#custom-boundaries", "#boundary-draw", "#bubble-message", "#roomareas", "#connections"])
                assert.equal(await page.$eval(sel, (e) => e.closest(".panel")?.dataset.panel ?? null), "rooms", `${sel} is in the Rooms panel`);
            const order = await page.$eval(".panel[data-panel=rooms]", (p) => [...p.querySelectorAll("#boundary-draw, #bubble-message")].map((e) => e.id));
            assert.deepEqual(order, ["boundary-draw", "bubble-message"], "the boundary comes before the room bubbles");
            assert.equal(await page.$eval(".panel[data-panel=rooms] .label", (e) => e.textContent), "Rooms");
            await page.click("button[data-open=rooms]");
            await page.click("button[data-mode=shape_a]");
            await page.waitForFunction(() => document.getElementById("status").textContent.startsWith("shape_a") && document.getElementById("status").textContent.includes("ready"));
            assert.deepEqual(await openPanels(page), ["rooms"], "picking a boundary keeps the Rooms panel open");
        });
    } finally {
        proc.kill();
    }
});

test("loss graph: value tags stay hidden until the mouse is on their row", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=optimize]");
            await page.click("#run");
            await page.waitForFunction(() => +(/iter (\d+)/.exec(document.getElementById("tick").textContent)?.[1] ?? 0) >= 10);
            await page.click("#run"); // pause
            const shown = () =>
                page.$$eval(".playhead .vtag", (ts) => ts.filter((t) => getComputedStyle(t).visibility === "visible" && t.textContent.length > 0).map((t) => t.id));
            assert.deepEqual(await shown(), [], "no value shows until the mouse is on a row");
            await page.locator(".lossrow").nth(0).hover();
            assert.deepEqual(await shown(), ["vt_total"], "the total row shows only its own value");
            await page.locator(".lossrow").nth(3).hover();
            assert.deepEqual(await shown(), ["vt_area"], "moving to the area row swaps the value");
            await page.mouse.move(1100, 700);
            assert.deepEqual(await shown(), [], "leaving the graph hides the value again");
        });
    } finally {
        proc.kill();
    }
});

test("names: the bubble tools and their Display toggle read Bubble Diagram", { timeout: 120_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const labels = await page.$$eval(".panel[data-panel=rooms] .label", (ls) => ls.map((l) => l.textContent));
            assert.ok(labels.includes("Bubble Diagram"), `Rooms panel sections: ${labels.join(", ")}`);
            assert.equal(await page.getByLabel("Bubble Diagram", { exact: true }).count(), 1, "the Display toggle is named Bubble Diagram");
            assert.equal(await page.getByText("Room bubbles").count(), 0, "the old name is gone");
        });
    } finally {
        proc.kill();
    }
});

test("rooms: no Delete selected button; the help text says to press Delete", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            assert.equal(await page.locator("#bubble-delete").count(), 0, "the Delete key replaces the button");
            assert.match(await page.locator("#bubble-message").textContent(), /Press Delete to remove the selected room or door\./);
        });
    } finally { proc.kill(); }
});

test("editing bar: Undo and Redo sit beside Editing Bubble Diagram, only while Rooms is open", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            for (const [id, name] of [["undo", "Undo"], ["redo", "Redo"]]) {
                const button = page.locator("#" + id);
                assert.equal(await button.evaluate((b) => !!b.closest("#bubble-editing")), true, `${name} is in the editing bar`);
                assert.equal(await button.evaluate((b) => !!b.closest(".panel")), false, `${name} is not in the Rooms panel`);
                assert.equal((await button.textContent()).trim(), "", `${name} shows an icon, not text`);
                assert.equal(await button.locator("svg use").count(), 1, `${name} has one icon`);
                assert.equal(await button.getAttribute("aria-label"), name);
                assert.equal(await button.isVisible(), true);
                await button.hover();
                await page.waitForTimeout(300);
                const tip = await button.evaluate((b) => { const s = getComputedStyle(b, "::after"); return { text: s.content, opacity: +s.opacity, top: parseFloat(s.top), height: b.getBoundingClientRect().height }; });
                assert.deepEqual([tip.text, tip.opacity], [JSON.stringify(name), 1], `${name} shows its name on hover`);
                assert.ok(tip.top >= tip.height, `${name}'s name shows below the bar`);
                assert.equal(await button.getAttribute("title"), null, `${name} has no browser title tooltip`);
            }
            const bar = await page.locator("#bubble-editing").boundingBox();
            assert.ok(Math.abs(bar.x + bar.width / 2 - 1145 / 2) < 2 && bar.y < 40, "the bar sits at the top center");
            await page.click("button[data-open=rooms]");
            assert.equal(await page.locator("#undo").isVisible(), false, "the bar and its buttons hide with the Rooms panel");
        });
    } finally { proc.kill(); }
});

test("rooms: Draw boundary is an icon-only button after the last preset, sized like a preset", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            const draw = page.locator("#boundary-draw");
            assert.equal((await draw.textContent()).trim(), "", "Draw boundary shows only its icon");
            assert.equal(await draw.getAttribute("aria-label"), "Draw boundary");
            assert.equal(await draw.locator("svg use").getAttribute("href"), "#i-vector-square");
            assert.deepEqual(await page.$eval("#i-vector-square", (s) => [...s.children].map((c) => c.tagName)), ["path", "path", "path", "path", "rect", "rect", "rect", "rect"], "four sides and four corner points");
            assert.equal(await draw.locator("svg").evaluate((s) => s.getBoundingClientRect().width), 20, "the icon matches the 20px preset icons in its row");
            const last = await page.locator("#presets button").last().boundingBox(), button = await draw.boundingBox();
            assert.ok(Math.abs(button.y - last.y) < 1 && Math.abs(button.height - last.height) < 1, "Draw boundary sits on the preset row");
            assert.ok(button.x > last.x + last.width && button.x - (last.x + last.width) < 10, "Draw boundary comes right after E");
            assert.ok(Math.abs(button.width - last.width) < 1, "Draw boundary is as wide as a preset");
            const panel = await page.locator(".panel[data-panel=rooms]").boundingBox();
            assert.ok(button.x + button.width <= panel.x + panel.width, "the row fits in the panel");
        });
    } finally { proc.kill(); }
});

test("rooms: preset buttons show an outline of their own boundary instead of a letter", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    // Scale a ring into a unit box from its lower-left corner, keeping its proportions.
    const unit = (ring) => {
        const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
        const size = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
        return ring.map(([x, y]) => [(x - Math.min(...xs)) / size, (y - Math.min(...ys)) / size]);
    };
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            for (const k of ["a", "b", "c", "d", "e"]) {
                const button = page.locator(`#presets button[data-mode=shape_${k}]`);
                const name = `Boundary ${k.toUpperCase()}`;
                assert.equal((await button.textContent()).trim(), "", `${name} shows no letter`);
                assert.equal(await button.getAttribute("aria-label"), name);
                const href = await button.locator("svg use").getAttribute("href");
                const icon = await page.$eval(href, (s) => s.querySelector("polygon").getAttribute("points").trim().split(/\s+/).map((p) => p.split(",").map(Number)));
                const boundary = JSON.parse(readFileSync(new URL(`./fixtures/shape_${k}.json`, import.meta.url))).boundary.slice(0, -1);
                const drawn = unit(icon.map(([x, y]) => [x, -y])), expected = unit(boundary);   // SVG y points down
                assert.equal(drawn.length, expected.length, `${name} draws every corner`);
                drawn.forEach((p, i) => assert.ok(Math.hypot(p[0] - expected[i][0], p[1] - expected[i][1]) < 0.01, `${name} corner ${i} is at ${p}, expected ${expected[i]}`));
            }
        });
    } finally { proc.kill(); }
});

test("details: a disabled segmented button looks disabled like a disabled toolbar icon", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            await page.evaluate(() => { document.getElementById("boundary-draw").disabled = true; document.querySelector("[data-mode=shape_a]").disabled = true; });
            const look = (sel) => page.$eval(sel, (b) => [getComputedStyle(b).opacity, getComputedStyle(b).cursor]);
            const tool = await look("#redo");
            assert.deepEqual(tool, ["0.4", "default"], "a disabled toolbar icon is faded");
            assert.deepEqual(await look("#boundary-draw"), tool, "disabled Draw boundary looks the same");
            assert.deepEqual(await look("[data-mode=shape_a]"), tool, "a disabled preset looks the same");
        });
    } finally { proc.kill(); }
});

test("details: sliders show a focus ring when reached by keyboard", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=optimize]");
            await page.locator("#w_topo").focus();
            await page.keyboard.press("Tab");
            assert.equal(await page.evaluate(() => document.activeElement.id), "sites");
            const ring = await page.$eval("#sites", (s) => [getComputedStyle(s).outlineStyle, parseFloat(getComputedStyle(s).outlineWidth)]);
            assert.notEqual(ring[0], "none", "the focused slider shows an outline");
            assert.ok(ring[1] >= 2, `the ring is ${ring[1]}px wide`);
        });
    } finally { proc.kill(); }
});

test("details: target area percentages keep the value style after the first room is drawn", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            await page.click("button[data-mode=shape_a]");
            await page.waitForFunction(() => { const s = document.getElementById("status").textContent; return s.startsWith("shape_a") && s.includes("ready"); });
            await page.click('.gizmo [data-axis="+z"]');
            await page.mouse.move(600, 450); await page.mouse.down(); await page.mouse.move(630, 450, { steps: 8 }); await page.mouse.up();
            const style = (sel) => page.$eval(sel, (v) => { const s = getComputedStyle(v); return [s.fontFamily, s.fontSize, s.textAlign, s.fontVariantNumeric]; });
            assert.deepEqual(await style(".room-row .val"), await style("#sites_v"), "a room's percent uses the same value style as the other values");
        });
    } finally { proc.kill(); }
});

test("details: the preset row keeps its 12px bottom margin", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            assert.equal(await page.$eval(".boundary-row", (r) => getComputedStyle(r).marginBottom), "12px");
        });
    } finally { proc.kill(); }
});

const openShapeA = async (page) => {
    await page.click("button[data-open=rooms]");
    await page.click("button[data-mode=shape_a]");
    await page.waitForFunction(() => { const s = document.getElementById("status").textContent; return s.startsWith("shape_a") && s.includes("ready"); });
    await page.click('.gizmo [data-axis="+z"]');
};
const drawRoomAt = async (page, [x, y]) => { await page.mouse.move(x, y); await page.mouse.down(); await page.mouse.move(x + 30, y, { steps: 8 }); await page.mouse.up(); };

test("polish: at 900px the editing bar does not cover the open Rooms panel", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [900, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            const panel = await page.locator(".panel[data-panel=rooms]").boundingBox(), bar = await page.locator("#bubble-editing").boundingBox();
            assert.ok(bar.x >= panel.x + panel.width, `the bar starts at ${bar.x}px, the panel ends at ${panel.x + panel.width}px`);
            assert.ok(bar.x + bar.width <= 900, "the bar stays in the window");
        });
    } finally { proc.kill(); }
});

test("polish: the selected preset and room look different from hovered ones", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const look = (sel) => page.$eval(sel, (b) => { const s = getComputedStyle(b); return [s.backgroundColor, s.borderColor, s.boxShadow].join(" "); });
            await openShapeA(page);
            await page.hover("[data-mode=shape_b]");
            await page.waitForTimeout(200);
            assert.notEqual(await look("[data-mode=shape_a]"), await look("[data-mode=shape_b]"), "selected A and hovered B differ");
            await drawRoomAt(page, [600, 450]);
            await drawRoomAt(page, [550, 530]);
            await page.locator(".room-row").first().click();
            await page.locator(".room-row").last().hover();
            await page.waitForTimeout(200);
            assert.notEqual(await look(".room-row.selected"), await look(".room-row:not(.selected)"), "selected and hovered rooms differ");
        });
    } finally { proc.kill(); }
});

test("polish: Display section labels fit on one line", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            const heights = await page.$$eval(".display-rows .row > label", (ls) => ls.map((l) => [l.textContent, l.getBoundingClientRect().height, parseFloat(getComputedStyle(l).lineHeight)]));
            for (const [text, height, line] of heights) assert.ok(height < line * 1.5, `"${text}" is ${height}px tall, one line is ${line}px`);
        });
    } finally { proc.kill(); }
});

test("polish: panel titles are larger than their section labels", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            for (const name of ["rooms", "optimize"]) {
                const sizes = await page.$$eval(`.panel[data-panel=${name}] .label`, (ls) => ls.map((l) => parseFloat(getComputedStyle(l).fontSize)));
                for (const size of sizes.slice(1)) assert.ok(sizes[0] > size, `${name}: title ${sizes[0]}px vs label ${size}px`);
            }
        });
    } finally { proc.kill(); }
});

test("polish: Draw boundary's icon matches the preset icons", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            const icon = (sel) => page.$eval(sel, (s) => [s.getBoundingClientRect().width, getComputedStyle(s).strokeWidth]);
            assert.deepEqual(await icon("#boundary-draw svg"), await icon("[data-mode=shape_a] svg"));
        });
    } finally { proc.kill(); }
});

test("polish: every section label has the same space above it", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 900], async (page) => {
            for (const name of ["rooms", "optimize"]) {
                await page.click(`button[data-open=${name}]`);
                const gaps = await page.$$eval(`.panel[data-panel=${name}] .label`, (ls) => ls.filter((l) => l.getClientRects().length).slice(1).map((l) => {   // a hidden heading (no door links yet) has no space to measure
                    let prev = l.previousElementSibling;
                    while (prev && !prev.getBoundingClientRect().height) prev = prev.previousElementSibling;
                    return [l.textContent, Math.round(l.getBoundingClientRect().top - prev.getBoundingClientRect().bottom)];
                }));
                for (const [text, gap] of gaps) assert.equal(gap, gaps[0][1], `${name}: ${text} has ${gap}px above it, ${gaps[0][0]} has ${gaps[0][1]}px`);
                await page.click(`button[data-open=${name}]`);
            }
        });
    } finally { proc.kill(); }
});

test("polish: errors in the bubble help and the program status share one error color", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            const colors = await page.evaluate(() => {
                const help = document.getElementById("bubble-message"), status = document.getElementById("program-status");
                help.classList.add("error"); status.dataset.state = "error"; status.textContent = "x";
                return [getComputedStyle(help).color, getComputedStyle(status).color, getComputedStyle(document.documentElement).getPropertyValue("--ui-error").trim()];
            });
            assert.equal(colors[0], colors[1], "both errors use the same color");
            assert.ok(colors[2], "the color is the --ui-error token");
        });
    } finally { proc.kill(); }
});

test("polish: the door-link drag line looks like the boundary sketch line", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const lines = await page.evaluate(() => {
                const svg = document.querySelector(".bubble-link-preview"), line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                svg.append(line);
                const sketch = document.querySelector("#boundary-sketch [data-edge-guide]");
                const look = (l) => [getComputedStyle(l).stroke, getComputedStyle(l).strokeDasharray, getComputedStyle(l).strokeWidth];
                const out = [look(line), look(sketch)];
                line.remove();
                return out;
            });
            assert.deepEqual(lines[0], lines[1]);
        });
    } finally { proc.kill(); }
});

test("polish: disabled buttons get no hover fill, and rows fade their hover like buttons", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await openShapeA(page);
            await page.hover("#redo");
            await page.waitForTimeout(200);
            assert.equal(await page.$eval("#redo", (b) => b.disabled && getComputedStyle(b).backgroundColor), "rgba(0, 0, 0, 0)", "disabled Redo stays clear on hover");
            await page.evaluate(() => { document.querySelector("[data-mode=shape_b]").disabled = true; });
            await page.hover("[data-mode=shape_b]");
            await page.waitForTimeout(200);
            assert.equal(await page.$eval("[data-mode=shape_b]", (b) => getComputedStyle(b).backgroundColor), "rgba(0, 0, 0, 0)", "a disabled preset stays clear on hover");
            await drawRoomAt(page, [600, 450]);
            const fade = (sel) => page.$eval(sel, (b) => getComputedStyle(b).transitionDuration.split(",")[0]);
            assert.equal(await fade(".room-row"), await fade(".seg button"));
        });
    } finally { proc.kill(); }
});

test("walls: a single Aligned walls icon switches between Aligned and Raw", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            const cell = page.locator(".toggle:has(#show-aligned)");
            const aligned = page.getByRole("checkbox", { name: "Aligned walls", exact: true });
            assert.equal((await cell.textContent()).trim(), "");
            assert.equal(await cell.locator("svg use").getAttribute("href"), "#i-walls-aligned");
            assert.equal(await aligned.isChecked(), false);
            await cell.click();
            assert.equal(await aligned.isChecked(), true);
            await cell.click();
            assert.equal(await aligned.isChecked(), false);
        });
    } finally { proc.kill(); }
});

test("tooltips: icon names in panels show centered under the icon and are never cut off", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const check = async (sel, name) => {
                await page.hover(sel);
                await page.waitForTimeout(300);
                const b = await page.locator(sel).boundingBox(), tip = page.locator(".tip");
                assert.equal(await tip.textContent(), name);
                assert.equal(await tip.evaluate((t) => +getComputedStyle(t).opacity), 1, `${name} shows`);
                assert.equal(await tip.evaluate((t) => !!t.closest(".panel")), false, `${name} is not inside the panel that would cut it off`);
                const t = await tip.boundingBox();
                assert.ok(Math.abs(t.x + t.width / 2 - (b.x + b.width / 2)) < 1, `${name} is centered under its icon`);
                assert.ok(t.y >= b.y + b.height, `${name} is below its icon`);
                assert.ok(t.x >= 0 && t.x + t.width <= 1145, `${name} is inside the window`);
            };
            await page.click("button[data-open=rooms]");
            await check(".toggle:has(#show-aligned)", "Aligned walls");
            await check(".toggle:has(#show-room-tags)", "Room tags");
            await check("#boundary-draw", "Draw boundary");
            await check("[data-mode=shape_a]", "Boundary A");
            await page.mouse.move(700, 700);
            await page.waitForTimeout(200);
            assert.equal(await page.locator(".tip").evaluate((t) => +getComputedStyle(t).opacity), 0, "the name hides when the pointer leaves");
        });
    } finally { proc.kill(); }
});

test("display: no aligned readout row; the plan keeps the aligned share for checks", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            const labels = await page.$$eval(".display-rows .row > label", (ls) => ls.map((l) => l.textContent));
            assert.ok(!labels.includes("aligned"), `Display rows: ${labels.join(", ")}`);
            assert.equal(await page.locator("#align_v").count(), 0);
            await page.waitForFunction(() => document.getElementById("plan").dataset.alignedShare);
            assert.match(await page.$eval("#plan", (c) => c.dataset.alignedShare), /^\d+% → \d+%$/);
        });
    } finally { proc.kill(); }
});

const presetCell = async (page) => {
    await page.click("button[data-open=rooms]");
    const cell = await page.locator("[data-mode=shape_a]").boundingBox();
    await page.click("button[data-open=rooms]");
    return cell;
};

test("toolbar: only the Rooms, Optimization, Adjustment and Export JSON icons", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            assert.deepEqual(await page.$$eval(".toolbar .tool", (ts) => ts.map((t) => t.getAttribute("aria-label"))), ["Rooms", "Optimization", "Adjustment", "Export JSON"]);
            assert.equal(await page.locator("#reset").count(), 0, "no Reset");
            assert.equal(await page.locator("button[data-open=display]").count(), 0, "no Display icon");
            assert.equal(await page.locator(".toolbar #run").count(), 0, "Run is not in the toolbar");
        });
    } finally { proc.kill(); }
});

test("optimization: Run is an icon cell on the right of the Optimization title", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const cell = await presetCell(page);
            await page.click("button[data-open=optimize]");
            const run = page.locator("#run");
            assert.equal(await run.evaluate((b) => !!b.closest(".panel[data-panel=optimize] .panel-head")), true, "Run sits in the Optimization title row");
            const box = await run.boundingBox(), head = await page.locator(".panel[data-panel=optimize] .panel-head").boundingBox();
            const title = await page.locator(".panel[data-panel=optimize] .panel-head .label").boundingBox();
            assert.ok(Math.abs(box.width - cell.width) < 1 && Math.abs(box.height - cell.height) < 1, `Run is ${box.width}x${box.height}, a preset cell is ${cell.width}x${cell.height}`);
            assert.ok(Math.abs(box.x + box.width - (head.x + head.width)) < 1, "Run is the last cell of the row");
            assert.ok(Math.abs(box.y + box.height / 2 - (title.y + title.height / 2)) < 4, "Run is level with the title");
            await run.hover();
            await page.waitForTimeout(300);
            assert.equal(await page.locator(".tip").textContent(), "Run", "hovering shows Run under it");
            await run.click();
            await page.waitForFunction(() => document.getElementById("run").getAttribute("aria-label") === "Pause");
            await run.click();
            await page.waitForFunction(() => document.getElementById("run").getAttribute("aria-label") === "Resume");
        });
    } finally { proc.kill(); }
});

test("rooms: View is the last section of Rooms and holds the display controls", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 900], async (page) => {
            assert.equal(await page.locator(".panel[data-panel=display]").count(), 0, "no separate Display panel");
            await page.click("button[data-open=rooms]");
            const order = await page.$eval(".panel[data-panel=rooms]", (p) => {
                const display = [...p.querySelectorAll(".label")].at(-1);
                return { last: display.textContent, after: ["wall", "merge_tol", "align_tol", "show-aligned", "show-cells", "show-sites", "show-doors", "show-bubbles", "show-room-tags"].map((id) => {
                    const el = document.getElementById(id);
                    return !!el && p.contains(el) && !!(display.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING);
                }) };
            });
            assert.equal(order.last, "View", "View is the last section of Rooms");
            assert.deepEqual(order.after, Array(9).fill(true), "the display controls sit under it");
        });
    } finally { proc.kill(); }
});

test("consistency: every slider in both panels starts and ends at the same x, and the loss graph starts there too", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 1000], async (page) => {
            const edges = [];
            for (const name of ["rooms", "optimize"]) {
                await page.click(`button[data-open=${name}]`);
                edges.push(...await page.$$eval(`.panel[data-panel=${name}] input[type=range]`, (rs) => rs.map((r) => { const b = r.getBoundingClientRect(); return [r.id, Math.round(b.left), Math.round(b.right)]; })));
                if (name === "optimize") {
                    const graph = await page.$eval("#sp_total", (s) => Math.round(s.closest(".spark").getBoundingClientRect().left));
                    assert.equal(graph, edges[0][1], `the loss graph starts at ${graph}, the sliders at ${edges[0][1]}`);
                }
                await page.click(`button[data-open=${name}]`);
            }
            for (const [id, left, right] of edges) {
                assert.equal(left, edges[0][1], `${id} starts at ${left}, ${edges[0][0]} at ${edges[0][1]}`);
                assert.equal(right, edges[0][2], `${id} ends at ${right}, ${edges[0][0]} at ${edges[0][2]}`);
            }
        });
    } finally { proc.kill(); }
});

test("consistency: row labels use sentence case and one muted color", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 1000], async (page) => {
            const labels = [];
            for (const name of ["rooms", "optimize"]) {
                await page.click(`button[data-open=${name}]`);
                labels.push(...await page.$$eval(`.panel[data-panel=${name}] :is(.row > label, .lossrow .lk, .roomareas .raswatch)`, (ls) => ls.map((l) => [l.textContent.trim(), getComputedStyle(l).color])));
                await page.click(`button[data-open=${name}]`);
            }
            const muted = await page.evaluate(() => getComputedStyle(document.querySelector(".row > label")).color);
            for (const [text, color] of labels) {
                assert.match(text, /^[A-Z0-9]/, `"${text}" starts with a capital`);
                assert.equal(color, muted, `"${text}" is ${color}, row labels are ${muted}`);
            }
            const texts = labels.map(([t]) => t);
            for (const name of ["Sites", "Wall", "Area", "Lloyd", "Topo", "Iterations", "LR", "Seed", "Total"]) assert.ok(texts.includes(name), `${name} is a row label: ${texts.join(", ")}`);
        });
    } finally { proc.kill(); }
});

test("rooms: Sites is the first Run setting in Optimization, and an empty Door links heading hides", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 1000], async (page) => {
            assert.equal(await page.$eval("#sites", (e) => e.closest(".panel").dataset.panel), "optimize");
            assert.deepEqual(await page.$$eval(".panel[data-panel=optimize] #sites, .panel[data-panel=optimize] #iters, .panel[data-panel=optimize] #lr, .panel[data-panel=optimize] #seed", (rs) => rs.map((r) => r.id)), ["sites", "iters", "lr", "seed"]);
            await page.click("button[data-open=rooms]");
            assert.equal(await page.locator("#connections button").count(), 0, "the preset has no door links");
            const visible = await page.$$eval(".panel[data-panel=rooms] .label", (ls) => ls.filter((l) => l.getClientRects().length).map((l) => l.textContent));
            assert.ok(!visible.includes("Door links"), `no empty Door links heading: ${visible.join(", ")}`);
        });
    } finally { proc.kill(); }
});

test("view: the View heading sits above six icon cells lined up with the Boundary cells", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 1000], async (page) => {
            await page.click("button[data-open=rooms]");
            const panel = page.locator(".panel[data-panel=rooms]");
            assert.equal(await panel.locator(".panel-head").count(), 0, "the Rooms title has no cells");
            const row = panel.locator(".view-row");
            const title = await panel.locator(".label", { hasText: /^View$/ }).boundingBox(), rowBox = await row.boundingBox();
            assert.ok(title && rowBox && title.y + title.height <= rowBox.y, "View is a heading on its own line, above its cells");
            const boundary = ["shape_a", "shape_b", "shape_c", "shape_d", "shape_e"].map((m) => `[data-mode=${m}]`).concat("#boundary-draw");
            const cells = ["show-aligned", "show-cells", "show-sites", "show-doors", "show-bubbles", "show-room-tags"].map(id => `.toggle:has(#${id})`);
            for (const [i, sel] of cells.entries()) {
                const box = await row.locator(sel).boundingBox(), under = await page.locator(boundary[i]).boundingBox();
                assert.ok(box, `${sel} sits in the View row`);
                assert.ok(Math.abs(box.x - under.x) < 1 && Math.abs(box.width - under.width) < 1 && Math.abs(box.height - under.height) < 1,
                    `${sel} is at x ${box.x}, ${box.width}x${box.height}; Boundary cell ${i + 1} is at x ${under.x}, ${under.width}x${under.height}`);
                assert.ok(Math.abs(box.y - rowBox.y) < 1, `${sel} is in the one row`);
            }
            assert.equal(await page.locator("input[type=checkbox]:visible").evaluateAll((cs) => cs.filter((c) => +getComputedStyle(c).opacity > 0).length), 0, "no checkbox box shows");
            for (const [id, name] of [["show-cells", "Voronoi cells"], ["show-sites", "Voronoi sites"], ["show-doors", "Doors"], ["show-bubbles", "Bubble Diagram"], ["show-room-tags", "Room tags"]]) {
                const toggle = row.locator(`.toggle:has(#${id})`);
                const on = await toggle.evaluate((t) => getComputedStyle(t).borderColor);
                await toggle.click();
                assert.equal(await page.$eval(`#${id}`, (c) => c.checked), false, `clicking ${name} turns it off`);
                await page.waitForTimeout(200);   // the border color eases over .12 s
                assert.notEqual(await toggle.evaluate((t) => getComputedStyle(t).borderColor), on, `${name} looks different when off`);
                await toggle.hover();
                await page.waitForTimeout(300);
                assert.equal(await page.locator(".tip").textContent(), name, `hovering shows ${name}`);
                await toggle.click();
                assert.equal(await page.$eval(`#${id}`, (c) => c.checked), true, `clicking ${name} again turns it on`);
            }
            await page.locator("#show-aligned").check();
            await page.waitForTimeout(200);
            assert.equal(await page.$eval(".toggle:has(#show-doors)", (t) => getComputedStyle(t).borderColor), await page.$eval(".toggle:has(#show-aligned)", (b) => getComputedStyle(b).borderColor), "the wall switch uses the same selected style");
        });
    } finally { proc.kill(); }
});

test("sketch color: Draw boundary, Adjustment and the door link preview draw in black, not blue", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            const black = "rgb(10, 10, 10)";   // --ui-fg, the black of the panels and icons
            for (const id of ["boundary-sketch", "wall-sketch"])
                assert.equal(await page.$eval(`#${id}`, (s) => getComputedStyle(s).color), black, `#${id} is black`);
            const link = await page.evaluate(() => {
                const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                svg.classList.add("bubble-link-preview");
                const line = svg.appendChild(document.createElementNS("http://www.w3.org/2000/svg", "line"));
                document.body.append(svg);
                const stroke = getComputedStyle(line).stroke;
                svg.remove();
                return stroke;
            });
            assert.equal(link, black, "the door link preview is black");
        });
    } finally { proc.kill(); }
});

test("view: the cells toggle and the sites toggle each switch their own layer with Raw and with Aligned walls", { timeout: 60_000 }, async () => {
    const { url, proc } = await serve();
    try {
        await atSize(url, [1145, 820], async (page) => {
            await page.click("button[data-open=rooms]");
            for (const id of ["show-cells", "show-sites"]) assert.equal(await page.$eval(`#${id}`, (c) => c.checked), true, `${id} is on at first`);
            const shot = async () => { await page.waitForTimeout(150); return page.$eval("#plan", (c) => c.toDataURL()); };
            const set = async (id, on) => { if (await page.$eval(`#${id}`, (c) => c.checked) !== on) await page.click(`.toggle:has(#${id})`); };
            for (const mode of ["raw", "aligned"]) {
                await page.locator("#show-aligned").setChecked(mode === "aligned");
                const pics = new Map();
                for (const [cells, sites] of [[true, true], [false, true], [true, false], [false, false]]) {
                    await set("show-cells", cells);
                    await set("show-sites", sites);
                    pics.set(`cells ${cells}, sites ${sites}`, await shot());
                }
                assert.equal(new Set(pics.values()).size, 4, `${mode}: each on/off pair of cells and sites draws a different picture`);
                await set("show-cells", true);
                await set("show-sites", true);
                assert.equal(await shot(), pics.get("cells true, sites true"), `${mode}: both on again draws as at first`);
            }
        });
    } finally { proc.kill(); }
});
