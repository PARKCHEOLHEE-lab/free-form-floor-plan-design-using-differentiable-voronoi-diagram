import {test} from 'node:test';
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {serve} from './serve.mjs';

test('angle and step controls reach real WASM and reduce raw step loss without changing the camera',{timeout:180000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1440,height:900}});
        await page.addInitScript(()=>{const Original=Worker;window.wallProof={builds:[],frames:[],boundary:null};window.Worker=class extends Original{
            constructor(...args){super(...args);this.addEventListener('message',e=>{if(e.data.type==='frame')window.wallProof.frames.push(e.data);if(e.data.type==='built')window.wallProof.boundary=e.data.boundary;});}
            postMessage(msg,...rest){if(msg.cmd==='build')window.wallProof.builds.push(msg);return super.postMessage(msg,...rest);}
        };});
        if(process.env.VORONOI_COVERAGE_DIR)await page.coverage.startJSCoverage({resetOnNavigation:false});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        assert.equal(await page.locator('.lossrow').count(),6,'the independent step contribution has its own sixth loss row');
        await page.click('button[data-open=weights]');
        assert.equal(await page.locator('label[for=w_wall]').textContent(),'angle');
        assert.equal(await page.locator('label[for=w_jog]').textContent(),'step');
        const preview=()=>page.evaluate(()=>wallProof.frames.at(-1).frame);
        const camera=()=>page.$eval('#plan',c=>({zoom:c.dataset.zoom,azimuth:c.dataset.azimuth,elevation:c.dataset.elevation}));
        const pose=await camera();const first=await preview();
        assert.ok(first.wall>0 && first.jog>0,'both new weighted terms reach the engine preview');
        const change=async(id,value)=>{
            const generation=await page.evaluate(()=>wallProof.builds.at(-1).gen);
            await page.locator('#'+id).evaluate((e,value)=>{e.value=String(value);e.dispatchEvent(new Event('input',{bubbles:true}));},value);
            if(id==='w_jog'||id==='iters')await page.click('#reset');
            await page.waitForFunction(gen=>wallProof.frames.at(-1)?.gen>gen && document.querySelector('#status').textContent.includes('ready'),generation);
        };
        await change('w_jog',0);const off=await preview();assert.equal(off.jog,0);assert.ok(Math.abs(off.wall-first.wall)<1e-5,'step weight does not change the angle model');
        await change('w_jog',0.7);const doubled=await preview();assert.ok(Math.abs(doubled.jog-2*first.jog)<1e-5,'step importance changes the real engine contribution linearly');
        await change('w_jog',0.35);await page.click('button[data-open=weights]');
        await page.click('button[data-open=rooms]');await change('sites',24);await page.click('button[data-open=rooms]');
        await page.click('button[data-open=optimize]');await change('iters',200);await page.click('button[data-open=optimize]');
        const before=await preview();const dir=new URL('../../.agents/cache/wall-shape-browser/',import.meta.url);await mkdir(dir,{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('before-raw.png',dir))});
        const start=Date.now();await page.click('#run');await page.waitForFunction(()=>document.querySelector('#status').textContent==='done',null,{timeout:150000});
        const after=await preview();assert.ok(after.jog<before.jog,`raw step contribution falls: ${before.jog} -> ${after.jog}`);
        assert.ok(after.wall<before.wall,'angle alignment also improves');
        assert.notDeepEqual(after.cells,before.cells,'the engine changes raw geometry');
        assert.equal(await page.locator('[data-walls=raw].active').count(),1);
        assert.ok((await page.locator('#sp_jog').getAttribute('points')).length>0);
        assert.deepEqual(await camera(),pose,'controls and new loss rows preserve the camera');
        await page.screenshot({path:fileURLToPath(new URL('after-raw.png',dir))});
        await page.click('button[data-open=display]');await page.click('[data-walls=aligned]');await page.click('button[data-open=display]');
        assert.deepEqual(await camera(),pose);await page.screenshot({path:fileURLToPath(new URL('after-aligned.png',dir))});
        await writeFile(new URL('measurements.json',dir),JSON.stringify({before,after,elapsedMs:Date.now()-start},null,2));
        if(process.env.VORONOI_COVERAGE_DIR)await writeFile(new URL('wall-loss-coverage.json',dir),JSON.stringify((await page.coverage.stopJSCoverage()).filter(e=>e.url===url||e.url.startsWith(url+'plan/'))));
    }finally{await browser.close();proc.kill();}
});
