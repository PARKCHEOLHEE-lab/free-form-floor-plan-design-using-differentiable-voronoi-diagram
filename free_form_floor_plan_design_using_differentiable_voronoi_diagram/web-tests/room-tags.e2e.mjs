import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { serve } from './serve.mjs';
import { alignPlan, cellWidth } from '../web/plan/align.js';
import { fromCells } from '../web/plan/graph.js';
import { roomShares } from '../web/plan/wall-edit.js';

async function withPlan(coverageName, run) {
    const { url, proc } = await serve();
    let browser;
    try {
        browser = await chromium.launch();
        const page = await browser.newPage({ viewport: { width: 1145, height: 820 } });
        const coverage = process.env.VORONOI_COVERAGE_DIR;
        if (coverage) await page.coverage.startJSCoverage();
        await page.addInitScript(() => {
            window.buildCount = 0;
            const post = Worker.prototype.postMessage;
            Worker.prototype.postMessage = function (message, ...args) {
                if (message.cmd === 'build') window.buildCount++;
                return post.call(this, message, ...args);
            };
        });
        await page.goto(url);
        await ready(page);
        await run(page);
        if (coverage) {
            await mkdir(coverage, { recursive: true });
            const data = (await page.coverage.stopJSCoverage()).filter(e => e.url === url || e.url.includes('/plan/'));
            await writeFile(join(coverage, coverageName + '.json'), JSON.stringify(data));
        }
    } finally {
        await browser?.close();
        proc.kill();
    }
}

const ready = page => page.waitForFunction(() => document.querySelector('#status').textContent.includes('ready'));
const roomName = (page, id) => page.getByRole('textbox', { name: 'Name for Room ' + id, exact: true });
const pixels = page => page.$eval('#plan', canvas => canvas.toDataURL());

async function drag(page, start, end) {
    await page.mouse.move(...start);
    await page.mouse.down();
    await page.mouse.move(...end, { steps: 10 });
    await page.mouse.up();
}

async function editWall(page) {
    await page.getByRole('button', { name: 'Adjustment', exact: true }).click();
    const lines = await page.locator('#wall-sketch [data-movable] line').evaluateAll(elements => elements.map(line =>
        ['x1', 'y1', 'x2', 'y2'].map(k => Number(line.getAttribute(k)))));
    for (const [x1, y1, x2, y2] of lines) {
        const start = [(x1 + x2) / 2, (y1 + y2) / 2];
        if (start[0] < 390 || Math.hypot(x2 - x1, y2 - y1) < 20) continue;
        await page.mouse.move(...start);
        if (!await page.locator('#plan[data-wall-hover]').count()) continue;
        const length = Math.hypot(x2 - x1, y2 - y1);
        await drag(page, start, [start[0] - 36 * (y2 - y1) / length, start[1] + 36 * (x2 - x1) / length]);
        if (await page.locator('#plan[data-wall-edits]').count()) return;
    }
    assert.fail('a movable wall can be dragged in the preset');
}

async function downloadPlan(page) {
    const download = page.waitForEvent('download');
    await page.getByRole('button', { name: 'Export JSON' }).click();
    return JSON.parse(await readFile(await (await download).path(), 'utf8'));
}

test('room names follow their room and are exported without rebuilding the plan', { timeout: 90_000 }, async () => {
    await withPlan('room-names', async page => {
        const rooms = page.getByRole('button', { name: 'Rooms', exact: true });
        await rooms.click();
        const first = roomName(page, 1);
        assert.equal(await first.count(), 1, 'the first preset room has a name field');
        assert.equal(await first.getAttribute('placeholder'), 'Room 1');
        assert.equal((await page.locator('#roomareas .raswatch').first().textContent()).trim(), 'Room 1');
        await page.locator('.gizmo [data-axis="+z"]').click();
        await editWall(page);
        assert.equal((await page.locator('#adjust-areas .raswatch').first().textContent()).trim(), 'Room 1');
        await rooms.click();
        const walls = await pixels(page);
        const builds = await page.evaluate(() => window.buildCount);
        const entered = '<Living & dining>';
        await first.fill(' ' + entered + ' ');
        await first.press('Enter');
        assert.equal(await first.inputValue(), entered);
        assert.equal(await pixels(page), walls, 'renaming retains the rendered wall edit');
        assert.equal(await page.evaluate(() => window.buildCount), builds);
        assert.equal(await page.locator('#discard-walls').isVisible(), false);
        let plan = await downloadPlan(page);
        assert.equal(plan.plan.source, 'edited');
        assert.equal(plan.plan.rooms[0].name, entered);
        assert.equal(plan.program, null, 'naming keeps the preset program');
        await first.fill('');
        await first.press('Tab');
        plan = await downloadPlan(page);
        assert.equal(plan.plan.rooms[0].name, undefined, 'clearing restores the default label');

        await page.getByRole('button', { name: 'Boundary A' }).click();
        await page.getByRole('button', { name: 'Discard' }).click();
        await ready(page);
        await page.locator('.gizmo [data-axis="+z"]').click();
        await drag(page, [550, 530], [585, 530]);
        await ready(page);
        await drag(page, [780, 530], [810, 530]);
        await ready(page);
        await roomName(page, 2).fill('Study');
        await roomName(page, 2).press('Tab');
        await page.locator('.room-row[data-room-id="1"]').click();
        await page.keyboard.press('Delete');
        await ready(page);
        assert.equal(await roomName(page, 2).inputValue(), 'Study', 'deleting a neighbor keeps the name by room ID');
        plan = await downloadPlan(page);
        assert.equal(plan.plan.rooms[0].id, 2);
        assert.equal(plan.plan.rooms[0].name, 'Study');
        await roomName(page, 2).fill(' ');
        await roomName(page, 2).press('Enter');
        assert.equal(await roomName(page, 2).inputValue(), '');
        assert.equal(await roomName(page, 2).getAttribute('placeholder'), 'Room 2');
        assert.equal(await page.locator('#room-tags .room-tag-name').textContent(), 'Room 2');
        assert.deepEqual(await page.locator('#room-tags .room-tag-number').allTextContents(), ['002'],
            'deleting Room 1 keeps the surviving room number');
        assert.equal((await downloadPlan(page)).plan.rooms[0].name, undefined);
    });
});

test('room tags show the current area and follow the displayed walls and camera', { timeout: 90_000 }, async () => {
    await withPlan('room-tags', async page => {
        const tags = page.locator('#room-tags .room-tag');
        const shares = () => tags.locator('.room-tag-share').allTextContents();
        const adjustmentShares = () => page.locator('#adjust-areas [data-now]').allTextContents();
        const screenPoints = () => tags.evaluateAll(elements => elements.map(tag => [tag.style.left, tag.style.top]));
        const rooms = page.getByRole('button', { name: 'Rooms', exact: true });
        await rooms.click();
        await page.getByRole('button', { name: 'Boundary B' }).click();
        await ready(page);
        await page.getByRole('slider', { name: 'Point merge' }).press('Home');
        const snap = page.getByRole('slider', { name: 'Wall snap' });
        await snap.evaluate(el => {
            el.value = '30';
            el.dispatchEvent(new Event('input', { bubbles: true }));
        });
        const { boundary: { ring }, frame: { cells, sites }, plan } = await downloadPlan(page);
        const n = plan.rooms.length;
        assert.equal(await tags.count(), n, 'there is one tag for each room of the Raw plan');
        assert.deepEqual(await tags.locator('.room-tag-name').allTextContents(),
            Array.from({ length: n }, (_, i) => 'Room ' + (i + 1)));
        assert.deepEqual(await tags.locator('.room-tag-number').allTextContents(),
            Array.from({ length: n }, (_, i) => String(i + 1).padStart(3, '0')));
        const raw = fromCells(cells, ring, { roomCount: n, simplifyTol: 0 });
        alignPlan(raw, ring, { mergeTol: 0, snapTol: 0.30 * cellWidth(ring, sites.length) });
        const rawShares = await shares();
        assert.deepEqual(rawShares, roomShares(raw).map(share => (100 * share).toFixed(1) + '%'));
        const tag = await tags.first().evaluate(element => {
            const lines = [...element.children];
            const rects = lines.map(line => line.getBoundingClientRect());
            return {
                text: lines.map(line => line.textContent),
                borders: [element, ...lines].map(line => getComputedStyle(line).borderTopWidth),
                background: getComputedStyle(element).backgroundColor,
                stacked: rects[0].bottom <= rects[1].top && rects[1].bottom <= rects[2].top,
            };
        });
        assert.deepEqual(tag.text, ['Room 1', '001', rawShares[0]]);
        assert.deepEqual(tag.borders, ['0px', '0px', '1px', '0px'], 'only the room number is boxed');
        assert.equal(tag.background, 'rgba(0, 0, 0, 0)', 'the tag leaves the room fill visible');
        assert.equal(tag.stacked, true);

        const fromIso = await screenPoints();
        await page.locator('.gizmo [data-axis="+z"]').click();
        assert.notDeepEqual(await screenPoints(), fromIso, 'the tags move with the camera');
        await page.getByRole('button', { name: 'Aligned walls' }).click();
        const alignedShares = await shares();
        assert.deepEqual(alignedShares, await adjustmentShares());
        assert.notDeepEqual(alignedShares, rawShares, 'preset B shows different shares in the two wall modes at these settings');
        const targets = await page.locator('#roomareas .val').allTextContents();
        assert.notDeepEqual(alignedShares.map(share => Math.round(parseFloat(share)) + '%'), targets,
            'the tag uses current areas instead of targets');

        const name = '<em>Kitchen</em>';
        await roomName(page, 1).fill(name);
        await roomName(page, 1).press('Enter');
        assert.equal(await tags.first().locator('.room-tag-name').textContent(), name);
        assert.equal(await tags.locator('em').count(), 0, 'names are plain text');
        await editWall(page);
        const editedShares = await shares();
        assert.equal(editedShares.length, n);
        assert.deepEqual(editedShares, await adjustmentShares());
        assert.notDeepEqual(editedShares, alignedShares, 'the wall edit updates current areas');
        assert.equal(await tags.first().locator('.room-tag-name').textContent(), name);
    });
});
