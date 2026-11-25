import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { chromium } from 'playwright';
import { serve } from './serve.mjs';

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
const roomName = (page, id) => page.getByRole('textbox', { name: 'Name for room R' + id, exact: true });
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
        assert.equal(await first.getAttribute('placeholder'), 'R1');
        await page.locator('.gizmo [data-axis="+z"]').click();
        await editWall(page);
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
        assert.equal((await downloadPlan(page)).plan.rooms[0].name, undefined);
    });
});
