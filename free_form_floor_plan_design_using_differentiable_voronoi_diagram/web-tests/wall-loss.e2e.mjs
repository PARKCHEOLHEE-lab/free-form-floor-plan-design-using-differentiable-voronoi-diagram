import {test} from 'node:test';
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {mkdir,readFile,writeFile} from 'node:fs/promises';
import {basename,join} from 'node:path';
import {fileURLToPath} from 'node:url';
import {serve} from './serve.mjs';
import {summarizeRawSteps} from './raw-wall-steps.mjs';
import {fromCells,wallEdges} from '../web/plan/graph.js';
import {openRing,ringArea,segmentDistance} from '../web/plan/geom.js';
import {evaluatePlan} from '../web/plan/openings.js';

async function observeWorker(page, input = null) {
    // Verify a candidate WASM build without replacing the live server's package.
    const wasm = process.env.WALL_TEST_WASM_DIR;
    if (wasm) {
        await page.context().route('**/pkg/**', route => route.fulfill({
            path:join(wasm,basename(new URL(route.request().url()).pathname)),
        }));
    }
    return page.addInitScript(program => {
        const Original = Worker;
        window.wallProof = { builds: [], frames: [], boundary: null, done: false };
        window.Worker = class extends Original {
            constructor(...args) {
                super(...args);
                this.addEventListener('message', ({ data }) => {
                    if (data.type === 'frame') wallProof.frames.push(data);
                    if (data.type === 'built') wallProof.boundary = data.boundary;
                    if (data.type === 'done') wallProof.done = true;
                });
            }
            postMessage(msg, ...rest) {
                if (msg.cmd === 'build') {
                    if (program) msg = { ...msg, ...program };
                    wallProof.builds.push(msg);
                    wallProof.done = false;
                }
                return super.postMessage(msg, ...rest);
            }
        };
    }, input);
}

test('wall and step controls reach real WASM without changing the camera',{timeout:180000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1440,height:900}});
        await observeWorker(page);
        if(process.env.VORONOI_COVERAGE_DIR)await page.coverage.startJSCoverage({resetOnNavigation:false});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        assert.equal(await page.locator('.lossrow').count(),6,'the independent step contribution has its own sixth loss row');
        await page.click('button[data-open=weights]');
        assert.equal(await page.locator('label[for=w_wall]').textContent(),'wall');
        assert.equal(await page.locator('label[for=w_jog]').textContent(),'step');
        assert.equal(await page.evaluate(()=>wallProof.builds.at(-1).w[0]),1/13,'restore the previous wall weight');
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
        const initialWeight = +(await page.locator('#w_jog').inputValue());
        assert.ok(initialWeight > 0);
        await change('w_jog',0);const off=await preview();assert.equal(off.jog,0);assert.equal(off.wall,first.wall,'step weight does not change the previous wall loss');
        await change('w_jog',0.5);const varied=await preview();
        assert.ok(Math.abs(varied.jog-first.jog*0.5/initialWeight)<1e-5,'step importance changes the real engine contribution linearly');
        await change('w_jog',initialWeight);await page.click('button[data-open=weights]');
        await page.click('button[data-open=rooms]');await change('sites',24);await page.click('button[data-open=rooms]');
        await page.click('button[data-open=optimize]');await change('iters',200);await page.click('button[data-open=optimize]');
        const before=await preview();const dir=new URL('../../.agents/cache/wall-shape-browser/',import.meta.url);await mkdir(dir,{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('before-raw.png',dir))});
        const start=Date.now();await page.click('#run');await page.waitForFunction(()=>document.querySelector('#status').textContent==='done',null,{timeout:150000});
        const after=await preview();assert.ok(after.jog<before.jog,`raw step contribution falls: ${before.jog} -> ${after.jog}`);
        assert.ok(after.wall<before.wall,'the previous wall loss also falls');
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

// A wall's endpoint drift against the nearest boundary axes is comparable
// across loss models. The display alignment uses the same 0.02 tolerance.
function measureWallAlignment(frame,boundary){
    const graph=fromCells(frame.cells,boundary,{roomCount:6,simplifyTol:.004});
    const outline=openRing(boundary);
    const edges=outline.map((a,i)=>[a,outline[(i+1)%outline.length]]);
    let length=0, alignedLength=0, maxDrift=0;
    for(const edge of wallEdges(graph).filter(edge=>edge.rooms.length===2)){
        const a=graph.corners.get(edge.a), b=graph.corners.get(edge.b);
        const midpoint=[(a.x+b.x)/2,(a.y+b.y)/2];
        const [s,t]=edges.reduce((best,edge)=>segmentDistance(midpoint,...edge)<segmentDistance(midpoint,...best)?edge:best);
        const angle=Math.atan2(t[1]-s[1],t[0]-s[0]);
        const x=b.x-a.x,y=b.y-a.y;
        const drift=Math.min(Math.abs(x*Math.cos(angle)+y*Math.sin(angle)),Math.abs(-x*Math.sin(angle)+y*Math.cos(angle)));
        const run=Math.hypot(x,y);
        length+=run;
        if(drift<=run*Math.sin(Math.PI/18)) alignedLength+=run;
        maxDrift=Math.max(maxDrift,drift);
    }
    return {maxDrift,alignedShare:length>0?alignedLength/length:0};
}

test('the six-room Raw walls have smaller visible steps and keep the room program',{timeout:180000},async()=>{
    const input = JSON.parse(await readFile(new URL('./fixtures/room-program-visible-steps.json',import.meta.url)));
    const {url,proc} = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({viewport:{width:1440,height:960}});
        await observeWorker(page,{
            mode:input.shape, sites:input.sites, seed:input.seed, iters:input.iterations,
            lr:input.lr, program:input,
        });
        await page.goto(url);
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        const initial = await page.evaluate(()=>wallProof.frames.at(-1).frame);
        await page.click('#run');
        await page.waitForFunction(()=>wallProof.done,null,{timeout:150000});
        const {frame,boundary} = await page.evaluate(()=>({frame:wallProof.frames.at(-1).frame,boundary:wallProof.boundary}));
        assert.equal(frame.iteration,500);
        assert.deepEqual(frame.sites.slice(0,6),initial.sites.slice(0,6),'bubble centers stay fixed');
        assert.equal(await page.locator('[data-walls=raw].active').count(),1);
        const graph = fromCells(frame.cells,boundary,{roomCount:6,simplifyTol:0});
        assert.deepEqual(evaluatePlan(graph,input,input.doorWidth,{finished:true}).issues,[]);
        const face = ringArea(boundary);
        const weights = input.rooms.map(room=>room.radius**2);
        const totalWeight = weights.reduce((sum,weight)=>sum+weight,0);
        for (const [room,weight] of weights.entries()) {
            const area = frame.cells.filter(cell=>cell.room===room).reduce((sum,cell)=>sum+ringArea(cell.ring),0);
            assert.ok(Math.abs(area/face-weight/totalWeight)<=.05,'each room stays within 5 percentage points of its target area');
        }
        const dir = new URL('../../.agents/cache/wall-quality/',import.meta.url);
        await mkdir(dir,{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('regression-raw.png',dir))});
        const {all,pair,junction,unmatched} = summarizeRawSteps(frame,boundary,6);
        const summary = {all,pair,junction,unmatched,alignment:measureWallAlignment(frame,boundary)};
        await writeFile(new URL('regression-summary.json',dir),JSON.stringify(summary,null,2));
        assert.ok(
            all.max<input.reference.raw_step_max && all.sum<=input.reference.raw_step_sum,
            'the drawn-room case keeps a smaller largest step without growing the summed height: '+JSON.stringify(summary),
        );
        // With step set to zero, the restored L1 model aligns 93.409% of
        // shared wall length within 10 degrees on this same drawn program.
        assert.ok(summary.alignment.alignedShare>=.9340908230878662-1e-9,
            'the step term must not trade steps for more off-axis wall length: '+summary.alignment.alignedShare);
    } finally {await browser.close();proc.kill();}
});

test('the default E preset has fewer visible Raw steps after Run',{timeout:180000},async()=>{
    const {url,proc} = await serve();
    const browser = await chromium.launch();
    try {
        const page = await browser.newPage({viewport:{width:1440,height:960}});
        await observeWorker(page);
        await page.goto(url);
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        const preview = await page.evaluate(()=>({
            frame:wallProof.frames.at(-1).frame,boundary:wallProof.boundary,build:wallProof.builds.at(-1),
        }));
        const areaErrors = ({frame,boundary,build}) => {
            const areas = Array(6).fill(0);
            for (const cell of frame.cells) areas[cell.room] += ringArea(cell.ring);
            return areas.map((area,room)=>Math.abs(area/ringArea(boundary)-build.ratios[room]));
        };
        const previewError = Math.max(...areaErrors(preview));
        assert.ok(previewError<=.05,'the preset starts near its target room areas: '+previewError);
        assert.ok(preview.frame.jog>0,'default preset applies the step weight to the engine');
        await page.click('#run');
        await page.waitForFunction(()=>wallProof.done,null,{timeout:150000});
        const {frame,boundary,build} = await page.evaluate(()=>({
            frame:wallProof.frames.at(-1).frame,boundary:wallProof.boundary,build:wallProof.builds.at(-1),
        }));
        assert.equal(build.mode,'shape_e');
        assert.equal(build.seed,4971);
        assert.equal(build.program,undefined,'opening the app and pressing Run uses the default preset');
        assert.equal(frame.iteration,500);
        assert.equal(await page.locator('[data-walls=raw].active').count(),1);
        const graph = fromCells(frame.cells,boundary,{roomCount:6,simplifyTol:0});
        assert.deepEqual(graph.rooms.map(rings=>rings.length),[1,1,1,1,1,1],'each default room stays connected');
        assert.ok(areaErrors({frame,boundary,build}).every(error=>error<=.05),'each default room stays within 5 percentage points of its target area');
        const {all,pair,junction} = summarizeRawSteps(frame,boundary,6);
        assert.ok(build.jogWeight>0,'default E uses both wall and step objectives');
        // L1 alone reached .009 max/sum on the same initial sites and
        // label assignment. Both must stay within one display tolerance.
        assert.ok(all.max<=.02 && all.sum<=.02,
            'largest and summed visible Raw steps must stay within .02: '+JSON.stringify({all,pair,junction}));
        assert.ok(measureWallAlignment(frame,boundary).maxDrift<=.02,'keep shared walls within the display alignment tolerance');
        const dir=new URL('../../.agents/cache/legacy-wall-step/',import.meta.url);
        await mkdir(dir,{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('default-e-raw.png',dir))});
    } finally {await browser.close();proc.kill();}
});
