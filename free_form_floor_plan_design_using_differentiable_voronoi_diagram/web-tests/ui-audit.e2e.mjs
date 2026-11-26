import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { test } from 'node:test';
import { chromium } from 'playwright';
import { serve } from './serve.mjs';

const ready = page => page.waitForFunction(() => document.querySelector('#status').textContent.includes('ready'));
const panel = (page, name) => page.locator(`button[data-open="${name}"]`);

async function withPlan(name, run, viewport = { width: 1280, height: 800 }) {
    const { url, proc } = await serve();
    let browser;
    try {
        browser = await chromium.launch();
        const page = await browser.newPage({ viewport });
        const coverageDir = process.env.VORONOI_AUDIT_COVERAGE;
        if (coverageDir) {
            await page.coverage.startJSCoverage();
            await page.coverage.startCSSCoverage();
        }
        await page.goto(url);
        await ready(page);
        await run(page);
        if (coverageDir) {
            await mkdir(coverageDir, { recursive: true });
            const js = (await page.coverage.stopJSCoverage()).filter(entry => entry.url === url || entry.url.includes('/plan/'));
            const css = await page.coverage.stopCSSCoverage();
            await writeFile(join(coverageDir, name + '.json'), JSON.stringify({ js, css }));
        }
    } finally {
        await browser?.close();
        proc.kill();
    }
}

test('audit: sliders and choices expose their names and selected state', { timeout: 60_000 }, async () => {
    await withPlan('control-names', async page => {
        await panel(page, 'rooms').click();
        assert.equal(await page.getByRole('slider', { name: 'Wall', exact: true }).count(), 1);
        const wallChoice = name => page.getByRole('button', { name: name + ' walls', exact: true });
        assert.equal(await wallChoice('Raw').getAttribute('aria-pressed'), 'true');
        assert.equal(await wallChoice('Aligned').getAttribute('aria-pressed'), 'false');
        await wallChoice('Aligned').click();
        assert.equal(await wallChoice('Raw').getAttribute('aria-pressed'), 'false');
        assert.equal(await wallChoice('Aligned').getAttribute('aria-pressed'), 'true');

        const boundary = name => page.getByRole('button', { name: 'Boundary ' + name, exact: true });
        assert.equal(await boundary('E').getAttribute('aria-pressed'), 'true');
        assert.equal(await boundary('A').getAttribute('aria-pressed'), 'false');
        await boundary('A').click();
        await ready(page);
        assert.equal(await boundary('E').getAttribute('aria-pressed'), 'false');
        assert.equal(await boundary('A').getAttribute('aria-pressed'), 'true');

        await panel(page, 'optimize').click();
        for (const name of ['Area', 'Lloyd', 'Topo', 'Sites', 'Iterations', 'LR', 'Seed']) {
            assert.equal(await page.getByRole('slider', { name, exact: true }).count(), 1, name);
        }
    });
});

test('audit: graph badges meet text contrast and tooltips have matching styles', { timeout: 60_000 }, async () => {
    await withPlan('badges-and-tooltips', async page => {
        await panel(page, 'optimize').click();
        const badges = await page.locator('.vtag').evaluateAll(elements => elements.map(element => {
            const luminance = color => {
                const rgb = color.match(/\d+(?:\.\d+)?/g).slice(0, 3).map(channel => Number(channel) / 255);
                const [r, g, b] = rgb.map(channel => channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4);
                return r * 0.2126 + g * 0.7152 + b * 0.0722;
            };
            const style = getComputedStyle(element);
            const fg = luminance(style.color), bg = luminance(style.backgroundColor);
            return { id: element.id, contrast: (Math.max(fg, bg) + 0.05) / (Math.min(fg, bg) + 0.05) };
        }));
        assert.equal(badges.length, 6);
        for (const { id, contrast } of badges) assert.ok(contrast >= 4.5, id + ' contrast is ' + contrast.toFixed(2));

        const appearance = await page.evaluate(() => {
            const parts = ['backgroundColor', 'color', 'fontSize', 'fontWeight', 'padding', 'borderRadius', 'transitionDuration'];
            return ['.toolbar .tool', '.gizmo .axis', '.tip'].map(selector => {
                const style = getComputedStyle(document.querySelector(selector), selector === '.tip' ? null : '::after');
                return parts.map(part => style[part]);
            });
        });
        assert.deepEqual(appearance[0], appearance[1]);
        assert.deepEqual(appearance[0], appearance[2]);
        await page.locator('#run').hover();
        assert.equal(await page.locator('.tip').textContent(), 'Run');
    });
});
