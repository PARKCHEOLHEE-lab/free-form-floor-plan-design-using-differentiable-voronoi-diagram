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
