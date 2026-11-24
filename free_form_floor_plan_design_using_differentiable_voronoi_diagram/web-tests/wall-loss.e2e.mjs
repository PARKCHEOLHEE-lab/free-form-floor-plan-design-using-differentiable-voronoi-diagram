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

test('the app weighs L1 and Cos walls separately and has no step loss', {timeout:60000}, async()=>{
    const {url,proc}=await serve();
    const browser=await chromium.launch();
    try{
        const page=await browser.newPage();
        await observeWorker(page);
        await page.goto(url);
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('button[data-open=optimize]');
        const l1=page.getByRole('slider',{name:'L1',exact:true});
        const cos=page.getByRole('slider',{name:'Cos',exact:true});
        assert.equal(await l1.count(),1,'the app shows a separate L1 slider');
        assert.equal(await cos.count(),1,'the app shows a separate Cos slider');
        assert.equal(await page.getByRole('slider',{name:'step',exact:true}).count(),0,'the step loss has no slider');
        assert.equal(Number(await l1.inputValue()),1);
        assert.equal(Number(await cos.inputValue()),1);
        assert.deepEqual(await page.locator('.lossrow .lk').allTextContents(),['total','L1','Cos','area','lloyd','topo']);
        const build=await page.evaluate(()=>wallProof.builds.at(-1));
        assert.equal(build.w[0],1/13,'L1 keeps its reference scale');
        assert.equal(build.cosineWeight,1);
        assert.equal(build.jogWeight,undefined,'the page sends no step weight');
        assert.equal(build.cosineWall,undefined,'the page sends no cos-only switch');
        const frame=()=>page.evaluate(()=>wallProof.frames.at(-1).frame);
        // a committed weight change rebuilds the preview by itself (there is no Reset)
        const preview=async(act)=>{
            const before=await page.evaluate(()=>wallProof.builds.at(-1).gen);
            await act();
            await page.waitForFunction(gen=>wallProof.builds.at(-1).gen>gen&&wallProof.frames.at(-1)?.gen===wallProof.builds.at(-1).gen&&document.querySelector('#status').textContent.includes('ready'),before);
            return frame();
        };
        const both=await frame();
        assert.ok(both.wall>0 && both.cosine>0,'both wall terms reach the engine');
        assert.equal(both.jog??0,0,'the engine adds no step contribution');
        const onlyL1=await preview(()=>cos.press('Home'));
        assert.equal(onlyL1.cosine??0,0);
        assert.equal(onlyL1.wall,both.wall);
        const onlyCos=await preview(async()=>{await cos.press('End');await l1.press('Home');});
        assert.equal(onlyCos.wall,0);
        assert.equal(onlyCos.cosine,both.cosine);
        for(const pure of [onlyL1,onlyCos])assert.deepEqual(pure.cells,both.cells);
        await l1.evaluate(input=>{input.value='1';input.dispatchEvent(new Event('input',{bubbles:true}));});
        await page.click('#run');
        await page.waitForFunction(()=>wallProof.frames.some(message=>message.mode==='step'));
        const first=await page.evaluate(()=>wallProof.frames.find(message=>message.mode==='step').frame);
        assert.ok(Math.abs(first.wall-both.wall)<1e-5);
        assert.ok(Math.abs(first.cosine-both.cosine)<1e-5);
        for(const old of ['?wall=cosine','?wall=mixed']){
            const other=await browser.newPage();
            await observeWorker(other);
            await other.goto(new URL(old,url).href);
            await other.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
            const {gen:_,...switched}=await other.evaluate(()=>wallProof.builds.at(-1));
            const {gen:__,...plain}=build;
            assert.deepEqual(switched,plain,`${old} no longer changes the loss`);
            await other.close();
        }
    }finally{await browser.close();proc.kill();}
});

test('the default area weight is 0.5 on every preset', {timeout:60000}, async()=>{
    const {url,proc}=await serve();
    const browser=await chromium.launch();
    try{
        const page=await browser.newPage();
        await observeWorker(page);
        await page.goto(url);
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        for(const preset of ['shape_e','shape_a','shape_b','shape_c','shape_d']){
            if(preset!=='shape_e'){
                await page.click('button[data-open=rooms]');await page.click(`button[data-mode=${preset}]`);await page.click('button[data-open=rooms]');
                await page.waitForFunction(m=>wallProof.builds.at(-1)?.mode===m&&document.querySelector('#status').textContent.includes('ready'),preset);
            }
            assert.equal(await page.locator('#w_area').inputValue(),'0.5',`${preset}: the area slider starts at 0.5`);
            assert.equal(await page.evaluate(()=>wallProof.builds.at(-1).w[1]),0.5/0.49,`${preset}: the engine gets the area weight at its reference scale`);
        }
    }finally{await browser.close();proc.kill();}
});

test('L1 and Cos controls reach real WASM without changing the camera',{timeout:180000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1440,height:900}});
        await observeWorker(page);
        if(process.env.VORONOI_COVERAGE_DIR)await page.coverage.startJSCoverage({resetOnNavigation:false});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        assert.equal(await page.locator('.lossrow').count(),6,'the Cos contribution has its own loss row');
        await page.click('button[data-open=optimize]');
        assert.equal(await page.locator('label[for=w_wall]').textContent(),'L1');
        assert.equal(await page.locator('label[for=w_cosine]').textContent(),'Cos');
        assert.equal(await page.evaluate(()=>wallProof.builds.at(-1).w[0]),1/13,'L1 keeps its reference scale');
        const preview=()=>page.evaluate(()=>wallProof.frames.at(-1).frame);
        const camera=()=>page.$eval('#plan',c=>({zoom:c.dataset.zoom,azimuth:c.dataset.azimuth,elevation:c.dataset.elevation}));
        const pose=await camera();const first=await preview();
        assert.ok(first.wall>0 && first.cosine>0,'both wall terms reach the engine preview');
        const change=async(id,value)=>{
            const generation=await page.evaluate(()=>wallProof.builds.at(-1).gen);
            await page.locator('#'+id).evaluate((e,value)=>{e.value=String(value);e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));},value);
            await page.waitForFunction(gen=>wallProof.frames.at(-1)?.gen>gen && document.querySelector('#status').textContent.includes('ready'),generation);
        };
        const initialWeight = +(await page.locator('#w_cosine').inputValue());
        assert.ok(initialWeight > 0);
        await change('w_cosine',0);const off=await preview();assert.equal(off.cosine??0,0);assert.equal(off.wall,first.wall,'Cos weight does not change the L1 wall loss');
        await change('w_cosine',0.5);const varied=await preview();
        assert.ok(Math.abs(varied.cosine-first.cosine*0.5/initialWeight)<1e-5,'Cos importance changes the real engine contribution linearly');
        await change('w_cosine',initialWeight);await page.click('button[data-open=optimize]');
        await page.click('button[data-open=rooms]');await change('sites',24);await page.click('button[data-open=rooms]');
        await page.click('button[data-open=optimize]');await change('iters',200);await page.click('button[data-open=optimize]');
        const before=await preview();const dir=new URL('../../.agents/cache/wall-shape-browser/',import.meta.url);await mkdir(dir,{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('before-raw.png',dir))});
        await page.click('button[data-open=optimize]');const start=Date.now();await page.click('#run');await page.click('button[data-open=optimize]');await page.waitForFunction(()=>document.querySelector('#status').textContent==='done',null,{timeout:150000});
        const after=await preview();assert.ok(after.cosine<before.cosine,`raw Cos contribution falls: ${before.cosine} -> ${after.cosine}`);
        assert.ok(after.wall<before.wall,'the L1 wall loss also falls');
        assert.notDeepEqual(after.cells,before.cells,'the engine changes raw geometry');
        assert.equal(await page.locator('[data-walls=raw].active').count(),1);
        assert.ok((await page.locator('#sp_cosine').getAttribute('points')).length>0);
        assert.deepEqual(await camera(),pose,'controls and new loss rows preserve the camera');
        await page.screenshot({path:fileURLToPath(new URL('after-raw.png',dir))});
        await page.click('button[data-open=rooms]');await page.click('[data-walls=aligned]');await page.click('button[data-open=rooms]');
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
        await page.click('button[data-open=optimize]');await page.click('#run');
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
            'the wall terms must not trade steps for more off-axis wall length: '+summary.alignment.alignedShare);
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
        assert.ok(preview.frame.cosine>0,'default preset applies the Cos weight to the engine');
        await page.click('button[data-open=optimize]');await page.click('#run');
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
        assert.ok(build.w[0]>0 && build.cosineWeight>0,'default E uses both L1 and Cos wall objectives');
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
