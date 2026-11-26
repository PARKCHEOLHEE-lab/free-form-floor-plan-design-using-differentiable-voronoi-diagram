import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { serve } from './serve.mjs';
import { CDPClient } from 'monocart-coverage-reports';
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';

test('the real worker installs room inputs rejects invalid links and cancels superseded runs', {timeout:90000}, async()=>{
    const {url,proc}=await serve();
    const collecting=!!process.env.VORONOI_COVERAGE_DIR;
    const browser=await chromium.launch({args:collecting?['--remote-debugging-port=0','--remote-debugging-address=127.0.0.1']:[]});
    let harness, coverage;
    try {
        const page=await browser.newPage();
        await page.goto(url);
        harness=await page.evaluateHandle(collecting=>{
            const source=new URL('./worker.js',location.href).href;
            // Delay importing production code until the worker's profiler is ready.
            const scriptUrl=collecting?URL.createObjectURL(new Blob([`onmessage=async()=>{await import(${JSON.stringify(source)});};postMessage({type:'boot'});`],{type:'text/javascript'})):source;
            const worker=new Worker(scriptUrl,{type:'module'}), messages=[], listeners=new Set();
            const receive=m=>{messages.push(m);for(const f of [...listeners])f(m);};
            worker.onmessage=e=>receive(e.data);
            worker.onerror=e=>receive({type:'fatal',msg:e.message});
            const wait=pred=>{
                const found=messages.find(pred);
                if(found)return Promise.resolve(found);
                return new Promise((resolve,reject)=>{
                    const timer=setTimeout(()=>{listeners.delete(f);reject(new Error('Worker response timed out'));},10000);
                    const f=m=>{if(pred(m)){clearTimeout(timer);listeners.delete(f);resolve(m);}};
                    listeners.add(f);
                });
            };
            return {worker,messages,wait,scriptUrl,collecting};
        },collecting);
        if(collecting){
            await harness.evaluate(h=>h.wait(m=>m.type==='boot'));
            const root=await browser.newBrowserCDPSession();
            const {arguments:args}=await root.send('Browser.getBrowserCommandLine');
            const dir=args.find(a=>a.startsWith('--user-data-dir=')).slice('--user-data-dir='.length);
            const port=+(await readFile(join(dir,'DevToolsActivePort'),'utf8')).split('\n')[0];
            const scriptUrl=await harness.evaluate(h=>h.scriptUrl);
            const {targetInfos}=await root.send('Target.getTargets');
            const target=targetInfos.find(t=>t.type==='worker'&&t.url===scriptUrl);
            assert.ok(target,'the real worker is reachable through CDP');
            coverage=await CDPClient({url:`ws://127.0.0.1:${port}/devtools/page/${target.targetId}`});
            assert.ok(coverage,'a coverage client attached to that worker');
            await coverage.startJSCoverage();
            await harness.evaluate(h=>h.worker.postMessage('begin'));
        }
        await harness.evaluate(h=>h.wait(m=>m.type==='ready'||m.type==='fatal'));
        const base={cmd:'build',mode:'shape_a',sites:18,ratios:[0.8,0.1,0.1],w:[0,1,0,0,0,0],seed:777,lr:0.01,iters:200,then:'preview'};
        const program={rooms:[{id:1,center:[-0.5,-0.45],radius:0.05},{id:2,center:[0,-0.45],radius:0.1},{id:3,center:[0.5,-0.45],radius:0.05}],links:[[0,2]],doorWidth:10};
        const send=msg=>harness.evaluate(async(h,msg)=>{
            h.worker.postMessage(msg);
            return h.wait(m=>m.type==='fatal'||(m.gen===msg.gen&&(m.type==='frame'||m.type==='error')));
        },msg);
        const first=await send({...base,gen:1,program});
        assert.equal(first.type,'frame');
        program.rooms.forEach((r,i)=>r.center.forEach((v,j)=>assert.ok(Math.abs(first.frame.sites[i][j]-v)<1e-6,'user centers reach the actual WASM optimizer')));
        assert.ok(first.frame.connection>0,'a door wider than the whole footprint produces a visible connection penalty');
        const legacy=await send({...base,gen:2,program,ratios:[0.1,0.1,0.8]});
        assert.deepEqual(legacy.frame.sites,first.frame.sites);
        assert.equal(legacy.frame.area,first.frame.area,'old slider ratios cannot override the bubble input');
        const changed={...program,rooms:program.rooms.map((r,i)=>({...r,radius:i===0?0.1:0.05}))};
        const radii=await send({...base,gen:3,program:changed});
        assert.notEqual(radii.frame.area,first.frame.area,'changing circle areas changes the engine area objective');
        await send({...base,gen:4,program,then:'start'});
        await send({...base,gen:5,program});
        await page.waitForTimeout(100);
        assert.equal(await harness.evaluate(h=>h.messages.filter(m=>m.type==='frame'&&m.gen===5&&m.mode==='step').length),0,'a preview build cancels the old running timer');
        const bad=[
            {...program,rooms:[]},
            {...program,rooms:program.rooms.map((r,i)=>i===0?{...r,radius:2}:r)},
            {...program,links:[[0,99]]},
            {...program,links:[[0.25,1]]},
            {...program,links:[[1,1]]},
            {...program,links:[[0,1],[1,0]]},
            {...program,doorWidth:0},
            {...program,rooms:program.rooms.map((r,i)=>i===1?{...r,center:[-0.5+1e-12,-0.45]}:r)},
            {...program,rooms:program.rooms.map((r,i)=>i===0?{...r,center:[0.1,0.588591966814866-1e-10],radius:1e-12}:r)},
        ];
        for(let i=0;i<bad.length;i++){
            const error=await send({...base,gen:20+i,program:bad[i]});
            assert.equal(error.type,'error',`bad room input ${i} is rejected rather than silently changed`);
            assert.equal(error.gen,20+i,'errors retain their input generation');
            assert.ok(error.msg);
        }
        assert.equal((await send({...base,gen:40,program,sites:2})).type,'error','the site budget must cover every room');
        assert.equal((await send({...base,gen:42,program,sites:3})).type,'error','fixed collinear centers alone cannot form a Voronoi diagram');
        assert.equal((await send({...base,gen:41,program})).type,'frame','the worker remains usable after rejected inputs');
        const shape={...base,w:[1,0,0,0,0,0]};
        const plain=await send({...shape,gen:43});
        const old=await send({...shape,gen:44,jogWeight:0.35,cosineWall:true});
        assert.equal(old.type,'frame','old step and cos-only fields are ignored, not rejected');
        assert.equal(old.frame.jog??0,0,'the worker adds no step contribution');
        assert.equal(old.frame.wall,plain.frame.wall,'the L1 wall term stays L1');
        for(const [i,weight] of [-1,NaN,Infinity].entries())assert.equal((await send({...shape,gen:45+i,cosineWeight:weight})).type,'error','an invalid Cos weight is rejected');
        const withCos=await send({...shape,gen:48,cosineWeight:1});
        assert.ok(withCos.frame.cosine>0 && withCos.frame.wall===plain.frame.wall,'Cos is its own term next to L1');
        const rectangle = [[2,1],[6,1],[6,3],[2,3]];
        const custom = {...base, mode:'custom', kind:'custom', boundary:rectangle, gen:49};
        const drawn = await send(custom);
        assert.equal(drawn.type, 'frame', 'the Worker builds a user-drawn boundary through WASM');
        const drawnBoundary = await harness.evaluate(h=>h.messages.find(m=>m.type==='built'&&m.gen===49).boundary);
        const expected = [[-2,-1],[2,-1],[2,1],[-2,1],[-2,-1]].map(p=>p.map(v=>v/Math.sqrt(5)));
        assert.deepEqual(drawnBoundary, expected, 'the engine keeps the drawn outline after normalization');
        const invalidOutlines = [
            null, [[0,0],[1,1]], [[0,0],[1,0],[0,Infinity]],
            [[0,0],[1,0],[2,0]], [[0,0],[2,2],[0,3],[3,0]],
            [[0,0],[2,0],[1,0],[2,2],[0,2]],
        ];
        for (const [i,boundary] of invalidOutlines.entries()) {
            const rejected = await send({...custom, gen:50+i, boundary});
            assert.equal(rejected.type, 'error', 'invalid outline is rejected before replacing the optimizer');
            assert.match(rejected.msg, /boundary/i);
            assert.equal(await harness.evaluate((h,gen)=>h.messages.some(m=>m.type==='built'&&m.gen===gen),50+i),false);
        }
        const rebuilt = await send({...custom, gen:56});
        assert.deepEqual(rebuilt.frame, drawn.frame, 'a rejected outline leaves the Worker usable and the custom build repeatable');
        const malformed=await page.evaluate(async()=>{
            const {default:init,WasmOpt}=await import('./pkg/voronoi_floorplan_web.js');
            await init();
            const opt=WasmOpt.from_shape('shape_a',18,Float64Array.of(0.8,0.1,0.1),0,1,0,0,0,0,777,0.01);
            const errors=[];
            try {
                for(const [circles,links] of [[[-0.5,-0.45],[]],[[-0.5,-0.45,0.05],[0]]]) {
                    try { opt.set_program(Float64Array.from(circles),Uint32Array.from(links),0.1); }
                    catch(e) { errors.push(String(e)); }
                }
            } finally { opt.free(); }
            return errors;
        });
        assert.equal(malformed.length,2,'the WASM boundary rejects partial circle and connection tuples');
        assert.ok(malformed.every(e=>!e.includes('not a function')),'errors come from input validation');
        if(coverage){
            const entries=(await coverage.stopJSCoverage()).filter(e=>e.url.endsWith('/worker.js'));
            assert.equal(entries.length,1,'production worker code was measured from import onwards');
            assert.equal(entries[0].source,await readFile(new URL('../web/worker.js',import.meta.url),'utf8'),'coverage source matches the tested file');
            await mkdir(process.env.VORONOI_COVERAGE_DIR,{recursive:true});
            await writeFile(join(process.env.VORONOI_COVERAGE_DIR,'worker.json'),JSON.stringify(entries));
        }
    } finally {
        if(coverage)await coverage.close();
        if(harness)await harness.evaluate(h=>{h.worker.terminate();if(h.collecting)URL.revokeObjectURL(h.scriptUrl);}).catch(()=>{});
        await browser.close();proc.kill();
    }
});
