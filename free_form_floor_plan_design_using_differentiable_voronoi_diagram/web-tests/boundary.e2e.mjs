import assert from 'node:assert/strict';
import { test } from 'node:test';
import { chromium } from 'playwright';
import { serve } from './serve.mjs';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

async function withPlan(callback, query = '', coverageName = 'sketch') {
    const { url, proc } = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({ viewport: { width: 1200, height: 840 } });
        if (process.env.VORONOI_COVERAGE_DIR) await page.coverage.startJSCoverage({resetOnNavigation:false});
        await page.addInitScript(() => {
            const NativeWorker = Worker;
            window.planProof = { builds: [], messages: [] };
            window.Worker = class extends NativeWorker {
                constructor(...args) {
                    super(...args);
                    this.addEventListener('message', event => {
                        const message = event.data;
                        if (message.type === 'frame' && message.gen === planProof.rejectPreviewGen) {
                            delete planProof.rejectPreviewGen;
                            event.stopImmediatePropagation();
                            this.dispatchEvent(new MessageEvent('message', {
                                data: { type: 'error', gen: message.gen, msg: 'Preview could not be rendered.' },
                            }));
                            return;
                        }
                        planProof.messages.push(message);
                    });
                }
                postMessage(message, ...args) {
                    if (message.cmd === 'build') planProof.builds.push(message);
                    if (message.kind === 'custom' && planProof.rejectNextPreview) {
                        planProof.rejectNextPreview = false;
                        planProof.rejectPreviewGen = message.gen;
                    }
                    if (message.kind === 'custom' && planProof.rejectNextCustom) {
                        planProof.rejectNextCustom = false;
                        message = {...message, boundary:[[0,0],[2,2],[0,3],[3,0]]};
                    }
                    super.postMessage(message, ...args);
                }
            };
        });
        await page.goto(url + query);
        await page.waitForFunction(() => document.querySelector('#status').textContent.includes('ready'));
        await callback(page);
        if (process.env.VORONOI_COVERAGE_DIR) {
            const coverage = (await page.coverage.stopJSCoverage()).filter(entry => entry.url.startsWith(url));
            await writeFile(join(process.env.VORONOI_COVERAGE_DIR, 'boundary-' + coverageName + '.json'),JSON.stringify(coverage));
        }
    } finally {
        await browser.close();
        proc.kill();
    }
}

test('boundary sketch uses a temporary overlay, snaps with Shift, rejects crossings and cancels', { timeout: 30000 }, async () => {
    await withPlan(async page => {
        await page.click('button[data-open=rooms]');
        const draw = page.getByRole('button', { name: 'Draw boundary', exact: true });
        assert.equal(await draw.count(), 1, 'Boundary offers a drawing tool on the plan');
        await draw.click();

        const sketch = page.locator('#boundary-sketch');
        const camera = () => page.locator('#plan').evaluate(c => [c.dataset.azimuth, c.dataset.elevation, c.dataset.zoom]);
        const before = await camera();
        await page.mouse.click(420, 290);
        await page.mouse.move(625, 310);
        await page.keyboard.down('Shift');
        assert.equal(await sketch.getAttribute('data-snap'),'axis','pressing Shift updates the current snap without moving the pointer');
        await page.mouse.click(625, 310);
        await page.keyboard.up('Shift');
        assert.deepEqual(await camera(), before, 'the draft never refits or rotates the view');

        await page.mouse.move(850, 350);
        await page.mouse.down({button:'right'});
        await page.mouse.move(890, 380, {steps:5});
        await page.mouse.up({button:'right'});
        assert.notDeepEqual(await camera(), before, 'right-drag still controls the view while sketching');
        await page.keyboard.press('Escape');
        assert.equal(await draw.getAttribute('aria-pressed'), 'false');
        assert.equal(await sketch.getAttribute('data-snap'), null);

        await page.click('.gizmo [data-axis="+z"]');
        await draw.click();
        await page.mouse.click(420,290);
        await page.mouse.click(580,450);
        await page.mouse.move(421,290);
        assert.equal(await sketch.getAttribute('data-snap'),'vertex');
        await page.mouse.move(502,370);
        assert.equal(await sketch.getAttribute('data-snap'),'midpoint');
        await page.mouse.move(720,586);
        assert.equal(await sketch.getAttribute('data-snap'),'parallel');
        assert.equal(await sketch.locator('text').textContent(),'Parallel');
        await page.mouse.move(722,311);
        assert.equal(await sketch.getAttribute('data-snap'),'perpendicular');
        assert.equal(await sketch.locator('text').textContent(),'Perpendicular');
        if (process.env.BOUNDARY_SCREEN) await page.screenshot({path:process.env.BOUNDARY_SCREEN.replace('.png','-sketch.png')});
        const cursor = sketch.locator('[data-cursor]');
        const snapped = [+(await cursor.getAttribute('cx')), +(await cursor.getAttribute('cy'))];
        await page.mouse.click(722,311);
        const points = (await sketch.locator('polyline').getAttribute('points')).split(' ').map(pair => pair.split(',').map(Number));
        assert.equal(points.length,3);
        assert.deepEqual(points[2],snapped,'the placed point is the previewed snap');
        const [a,b,c] = points;
        const dot = (b[0]-a[0])*(c[0]-b[0]) + (b[1]-a[1])*(c[1]-b[1]);
        assert.ok(Math.abs(dot)<1e-6,'the two actual segments meet at 90 degrees in the top view');
        await draw.click();
        assert.equal(await sketch.getAttribute('data-snap'),null);
        await draw.click();
        for (const point of [[410,300],[760,550],[410,510],[710,290],[410,300]]) await page.mouse.click(...point);
        assert.match(await page.locator('#boundary-message').textContent(), /cross/i);
        assert.equal(await draw.getAttribute('aria-pressed'), 'true', 'an invalid ring remains a draft');
        await page.keyboard.press('Backspace');
        await page.keyboard.press('Escape');
        assert.equal(await page.evaluate(() => planProof.builds.length), 1, 'cancel and rejected closure never build a new plan');
        assert.equal(await page.getByRole('button', {name:'Run', exact:true}).isEnabled(), true);
    });
});

test('Shift tracks a draft endpoint height and a midpoint with a visible guide', { timeout: 30000 }, async () => {
    await withPlan(async page => {
        await page.click('button[data-open=rooms]');
        await page.click('.gizmo [data-axis="+z"]');
        await page.getByRole('button',{ name:'Draw boundary', exact:true }).click();
        for (const point of [[485,210],[485,555],[630,555],[630,730],[840,730]]) await page.mouse.click(...point);

        const sketch = page.locator('#boundary-sketch');
        const guide = sketch.locator('[data-tracking-guide]');
        const cursor = sketch.locator('[data-cursor]');
        const cursorPoint = async () => [+(await cursor.getAttribute('cx')), +(await cursor.getAttribute('cy'))];
        const near = (a,b) => assert.ok(Math.abs(a-b)<0.01, a + ' should equal ' + b);
        await page.keyboard.down('Shift');
        await page.mouse.move(841,548);
        assert.equal(await sketch.getAttribute('data-snap'),'vertex','the shoulder height takes precedence over plain X/Y');
        assert.equal(await sketch.locator('text').textContent(),'Endpoint');
        // A horizontal SVG line has a zero-height DOM box even when its stroke is drawn.
        assert.notEqual(await guide.evaluate(el => getComputedStyle(el).display),'none');
        const [x,y] = await cursorPoint();
        near(x,840);
        near(y,555);
        near(+(await guide.getAttribute('y1')),y);
        near(+(await guide.getAttribute('y2')),y);

        await page.mouse.move(841,639);
        assert.equal(await sketch.getAttribute('data-snap'),'midpoint');
        assert.equal(await sketch.locator('text').textContent(),'Midpoint');
        assert.notEqual(await guide.evaluate(el => getComputedStyle(el).display),'none');
        const midpoint = await cursorPoint();
        near(midpoint[0],840);
        near(midpoint[1],642.5);
        if (process.env.BOUNDARY_SCREEN) await page.screenshot({path:process.env.BOUNDARY_SCREEN});
        await page.mouse.click(841,639);
        const placed = (await sketch.locator('polyline').getAttribute('points')).split(' ').at(-1).split(',').map(Number);
        assert.deepEqual(placed,midpoint,'clicking places the exact previewed tracking intersection');
        await page.keyboard.up('Shift');
        assert.equal(await guide.getAttribute('hidden'),'');
    }, '', 'tracking');
});

test('a drawn boundary runs with rooms and retains one custom design across replacements', { timeout: 90000 }, async () => {
    await withPlan(async page => {
        const ready = name => page.waitForFunction(name => document.querySelector('#status').textContent.includes(name + ' · ready'), name);
        const drawRoom = async (x, y, radius) => {
            await page.mouse.move(x,y);
            await page.mouse.down();
            await page.mouse.move(x + radius,y,{ steps:8 });
            await page.mouse.up();
        };
        const labels = () => page.locator('.bubble-label:not(.draft)').evaluateAll(all => all.map(node => ({...node.dataset})));
        // Open a panel unless it is already open (the boundary controls live in Rooms).
        const selectPanel = async name => {
            if (!(await page.locator('.panel[data-panel=' + name + ']').evaluate(p => p.classList.contains('open')))) await page.click('button[data-open=' + name + ']');
        };

        await selectPanel('rooms');
        await page.click('button[data-mode=shape_a]');
        await ready('shape_a');
        await page.click('.gizmo [data-axis="+z"]');
        await selectPanel('rooms');
        await page.click('button[data-bubble-tool=draw]');
        await drawRoom(550,530,33);
        await ready('shape_a');
        const presetRooms = await labels();
        assert.equal(presetRooms.length,1);

        await selectPanel('rooms');
        const sketchRectangle = async () => {
            await page.getByRole('button',{ name:'Draw boundary', exact:true }).click();
            for (const p of [[420,285],[790,285],[790,555],[420,555],[420,285]]) await page.mouse.click(...p);
        };
        const custom = page.getByRole('button',{ name:'Custom', exact:true });
        await page.evaluate(() => { planProof.rejectNextCustom = true; });
        await sketchRectangle();
        await page.waitForFunction(() => document.querySelector('#boundary-message').textContent.includes('must enclose'));
        assert.equal(await custom.count(),0);
        assert.deepEqual(await labels(),presetRooms,'a Worker rejection leaves the current design active');

        await sketchRectangle();
        await ready('custom');
        assert.equal(await custom.count(),1,'closing the sketch creates a selectable custom boundary');
        assert.deepEqual(await labels(),[],'the custom outline starts with its own room program');
        const drawn = await page.evaluate(() => planProof.builds.at(-1));
        assert.equal(drawn.kind,'custom');
        assert.equal(drawn.boundary.length,4);
        const captureBoundary = gen => page.evaluate(gen => planProof.messages.find(m => m.gen === gen && m.type === 'built').boundary, gen);
        const outline = await captureBoundary(drawn.gen);

        await selectPanel('rooms');
        await page.click('button[data-bubble-tool=draw]');
        await drawRoom(500,420,32);
        await ready('custom');
        await drawRoom(710,420,36);
        await ready('custom');
        const customRooms = await labels();
        assert.equal(customRooms.length,2);
        await page.click('button[data-bubble-tool=link]');
        await page.mouse.click(500,420);
        await page.mouse.click(710,420);
        await page.waitForFunction(() => document.querySelectorAll('.connection-row').length === 1);
        await ready('custom');

        await selectPanel('optimize');
        await page.locator('#iters').focus();
        await page.keyboard.press('Home');
        await page.getByRole('button',{ name:'Run', exact:true }).click();
        await page.waitForFunction(() => planProof.messages.some(m => m.gen === planProof.builds.at(-1).gen && m.type === 'done'));
        const run = await page.evaluate(() => {
            const build = planProof.builds.at(-1);
            const frames = planProof.messages.filter(m => m.type === 'frame' && m.gen === build.gen);
            return { build, frame: frames.at(-1).frame };
        });
        assert.equal(run.frame.iteration,50);
        if (process.env.BOUNDARY_SCREEN) await page.screenshot({path:process.env.BOUNDARY_SCREEN});
        assert.equal(run.build.cosineWeight,1,'the drawn boundary uses the active loss controls');
        assert.deepEqual(await captureBoundary(run.build.gen),outline,'rebuilds keep the same normalized plan');
        assert.equal(run.build.program.links.length,1);
        run.build.program.rooms.forEach((r,i) => r.center.forEach((coordinate,j) => {
            assert.ok(Math.abs(run.frame.sites[i][j] - coordinate) < 1e-6,'the fixed centers reach the real optimizer');
        }));
        assert.equal(await page.locator('#plan').getAttribute('data-doors'),'1','the requested door is rendered');

        await selectPanel('rooms');
        await page.click('button[data-mode=shape_a]');
        await ready('shape_a');
        assert.deepEqual(await labels(),presetRooms);
        await custom.click();
        await ready('custom');
        assert.deepEqual(await labels(),customRooms);
        assert.equal(await page.locator('.connection-row').count(),1);
        assert.deepEqual(await captureBoundary(await page.evaluate(() => planProof.builds.at(-1).gen)),outline);

        const startReplacement = async () => {
            await page.click('.gizmo [data-axis="+z"]');
            await page.getByRole('button',{ name:'Draw boundary', exact:true }).click();
        };
        const drawReplacement = async () => {
            await startReplacement();
            for (const point of [[470,250],[830,250],[830,500],[630,590],[470,500],[470,250]]) await page.mouse.click(...point);
        };
        const customButtons = page.locator('#custom-boundaries button:visible');

        const beforeCancel = await page.evaluate(() => planProof.builds.length);
        await startReplacement();
        await page.mouse.click(470,250);
        await page.mouse.click(830,250);
        await page.keyboard.press('Escape');
        assert.equal(await page.evaluate(() => planProof.builds.length),beforeCancel);
        assert.deepEqual(await labels(),customRooms,'cancelling keeps the old rooms and outline');

        await page.evaluate(() => { planProof.rejectNextPreview = true; });
        await drawReplacement();
        await page.waitForFunction(() => planProof.messages.some(m => m.gen === planProof.builds.at(-1).gen && m.type === 'error'));
        assert.equal(await customButtons.count(),1,'a replacement is not installed when its first preview fails');
        assert.deepEqual(await labels(),customRooms);
        assert.equal(await page.locator('.connection-row').count(),1);
        assert.match(await page.locator('#boundary-message').textContent(),/preview could not be rendered/i);
        await page.click('#reset');
        await page.waitForFunction(() => planProof.messages.some(m => m.gen === planProof.builds.at(-1).gen && m.type === 'frame'));
        assert.deepEqual(await captureBoundary(await page.evaluate(() => planProof.builds.at(-1).gen)),outline);
        assert.deepEqual(await labels(),customRooms,'rebuilding after a failed preview still uses the old outline and rooms');

        await drawReplacement();
        await page.waitForFunction(() => planProof.messages.some(m => m.gen === planProof.builds.at(-1).gen && m.type === 'frame'));
        assert.equal(await customButtons.count(),1,'a second successful outline replaces the first');
        const latest = await captureBoundary(await page.evaluate(() => planProof.builds.at(-1).gen));
        assert.notDeepEqual(latest,outline);
        assert.deepEqual(await labels(),[],'the replaced custom rooms are removed');
        assert.equal(await page.locator('.connection-row').count(),0);
        assert.equal(await page.locator('#plan').getAttribute('data-doors'),'0');

        await page.click('button[data-mode=shape_a]');
        await ready('shape_a');
        assert.deepEqual(await labels(),presetRooms,'replacement leaves preset rooms intact');
        await custom.click();
        await page.waitForFunction(() => planProof.messages.some(m => m.gen === planProof.builds.at(-1).gen && m.type === 'frame'));
        assert.deepEqual(await captureBoundary(await page.evaluate(() => planProof.builds.at(-1).gen)),latest);
        assert.deepEqual(await labels(),[]);

    }, '', 'run');
});
