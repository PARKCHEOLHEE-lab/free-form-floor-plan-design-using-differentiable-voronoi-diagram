import {test} from 'node:test';
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {mkdir,writeFile,readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {serve} from './serve.mjs';
import {fromCells} from '../web/plan/graph.js';
import {evaluatePlan} from '../web/plan/openings.js';
import {ringArea} from '../web/plan/geom.js';

async function drag(page,a,b){
    await page.mouse.move(...a);await page.mouse.down();await page.mouse.move(...b,{steps:12});await page.mouse.up();
}
const ready=page=>page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
const camera=page=>page.$eval('#plan',c=>({zoom:c.dataset.zoom,azimuth:c.dataset.azimuth,elevation:c.dataset.elevation}));
const target=page=>page.$$eval('.bubble-label:not(.draft)',els=>els.map(e=>+e.dataset.ratio));
const filledShares=page=>page.$eval('#plan',async c=>{
    const image=new Image();image.src=c.toDataURL();await image.decode();
    const copy=document.createElement('canvas');copy.width=image.width;copy.height=image.height;
    const ctx=copy.getContext('2d');ctx.drawImage(image,0,0);
    const pixels=ctx.getImageData(0,0,copy.width,copy.height).data,counts=[0,0];
    const colors=[[191,228,191],[222,215,234]];
    for(let i=0;i<pixels.length;i+=4)colors.forEach((color,r)=>{
        if(Math.abs(pixels[i]-color[0])+Math.abs(pixels[i+1]-color[1])+Math.abs(pixels[i+2]-color[2])<20)counts[r]++;
    });
    return {pixels:counts,shares:counts.map(n=>n/(counts[0]+counts[1]))};
});

test('five drawn rooms keep one region each in raw and aligned initial views',{timeout:120000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    // Wide enough that the open Rooms panel, needed for drawing, leaves the whole plan uncovered.
    const width=1600;
    try{
        const page=await browser.newPage({viewport:{width,height:820}});
        // Observe messages from the actual production worker without replacing
        // its optimizer or changing the main page's message handler.
        await page.addInitScript(()=>{
            const Original=Worker;
            window.__planCapture={};
            window.Worker=class extends Original{
                constructor(...args){super(...args);this.addEventListener('message',e=>{
                    if(e.data.type==='built')window.__planCapture.boundary=e.data.boundary;
                    if(e.data.type==='frame')window.__planCapture.frame=e.data.frame;
                });}
            };
        });
        if(process.env.VORONOI_COVERAGE_DIR)await page.coverage.startJSCoverage({resetOnNavigation:false});
        await page.goto(url);await ready(page);
        await page.click('button[data-open=optimize]');await page.locator('#seed').focus();await page.keyboard.press('Home');await ready(page);
        assert.equal(await page.locator('#seed').inputValue(),'0');
        await page.click('button[data-open=optimize]');await page.click('.gizmo [data-axis="+z"]');
        await page.click('button[data-open=rooms]');
        const {boundary}=JSON.parse(await readFile(new URL('./fixtures/shape_e.json',import.meta.url),'utf8'));
        const xs=boundary.map(p=>p[0]),ys=boundary.map(p=>p[1]);
        const cx=(Math.max(...xs)+Math.min(...xs))/2,cy=(Math.max(...ys)+Math.min(...ys))/2;
        const canvas=await page.locator('#plan').boundingBox();
        const fitSize=Math.min(canvas.width,canvas.height);
        const scale=Math.max(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys))/(0.8*fitSize);
        const screen=([x,y])=>[canvas.x+canvas.width/2+(x-cx)/scale,canvas.y+canvas.height/2-(y-cy)/scale];
        for(const [x,y,r] of [[-.48,.35,.11],[-.2,-.32,.21],[.3,.25,.12],[.1,.55,.13],[-.5,-.1,.08]]){
            await drag(page,screen([x,y]),screen([x+r,y]));await ready(page);
        }
        const input=await page.$$eval('.bubble-label:not(.draft)',els=>els.map(e=>({center:[+e.dataset.x,+e.dataset.y],radius:+e.dataset.radius,ratio:+e.dataset.ratio})));
        assert.equal(input.length,5);
        const capture=await page.evaluate(()=>window.__planCapture);
        assert.equal(capture.frame.iteration,0);
        const pose=await camera(page);
        await page.uncheck('#show-bubbles');await page.click('button[data-open=rooms]');
        const dir=new URL('../../.agents/cache/screens/',import.meta.url);await mkdir(dir,{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('five-room-connectivity-raw.png',dir))});
        const graph=fromCells(capture.frame.cells,capture.boundary,{roomCount:5,simplifyTol:0});
        const checked=evaluatePlan(graph,{rooms:input,links:[]},0.1);
        assert.equal(checked.valid,true,JSON.stringify(checked.issues));
        const areas=[0,0,0,0,0];
        for(const cell of capture.frame.cells)areas[cell.room]+=ringArea(cell.ring);
        const total=ringArea(capture.boundary);
        areas.forEach((a,r)=>assert.ok(Math.abs(a/total-input[r].ratio)<=0.05,`room ${r+1} keeps its target area`));
        assert.equal((await page.locator('#program-status').textContent()).trim(),'Ready to run.');
        await page.click('button[data-open=rooms]');await page.locator('#show-aligned').check();await page.click('button[data-open=rooms]');
        assert.equal((await page.locator('#program-status').textContent()).trim(),'Ready to run.','alignment preserves one connected region per room');
        assert.deepEqual(await camera(page),pose);
        await page.screenshot({path:fileURLToPath(new URL('five-room-connectivity-aligned.png',dir))});
        await writeFile(new URL('five-room-connectivity.json',dir),JSON.stringify({input,shares:areas.map(a=>a/total),issues:checked.issues},null,2));
        if(process.env.VORONOI_COVERAGE_DIR){
            await mkdir(process.env.VORONOI_COVERAGE_DIR,{recursive:true});
            await writeFile(new URL('five-room-browser-coverage.json',dir),JSON.stringify(await page.coverage.stopJSCoverage()));
        }
    }finally{await browser.close();proc.kill();}
});

test('resizing a circle changes actual initial room areas before Run without moving the camera',{timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        await page.goto(url);await ready(page);
        await page.click('button[data-open=rooms]');await page.click('button[data-mode=shape_a]');await ready(page);
        await page.click('.gizmo [data-axis="+z"]');
        const canvas=await page.locator('#plan').boundingBox();
        const scale=Math.min(canvas.width,canvas.height)/820;
        const center=([x,y])=>[Math.round(canvas.x+canvas.width/2+(x-1145/2)*scale),
            Math.round(canvas.y+canvas.height/2+(y-820/2)*scale)];
        const first=center([550,530]),second=center([780,530]);
        await drag(page,first,[first[0]+30,first[1]]);await drag(page,second,[second[0]+60,second[1]]);await ready(page);
        const pose=await camera(page),beforeTarget=await target(page);
        assert.ok(Math.abs(beforeTarget[0]-0.2)<1e-6);
        await page.uncheck('#show-bubbles');
        const before=await filledShares(page);
        assert.ok(before.pixels.every(n=>n>10000),'actual room fills, without circle overlays, are visible');
        assert.ok(Math.abs(before.shares[0]-beforeTarget[0])<=0.05,`initial rendered share ${before.shares[0]} must approach ${beforeTarget[0]}`);
        assert.equal(await page.locator('#run').getAttribute('aria-label'),'Run');
        assert.equal(await page.locator('.playhead .tick').textContent(),'iter 0','the preview has not optimized');
        const folder=new URL('../../.agents/cache/screens/',import.meta.url);await mkdir(folder,{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('initial-area-before.png',folder))});
        await page.check('#show-bubbles');
        await page.mouse.click(...first);
        const h=await page.locator('.bubble-handle').boundingBox();assert.ok(h);
        await drag(page,[h.x+h.width/2,h.y+h.height/2],[h.x+h.width/2+30,h.y+h.height/2]);await ready(page);
        const afterTarget=await target(page);
        await page.uncheck('#show-bubbles');
        const after=await filledShares(page);
        assert.ok(Math.abs(afterTarget[0]-0.5)<1e-5);
        assert.ok(Math.abs(after.shares[0]-afterTarget[0])<=0.05,`resized rendered share ${after.shares[0]} must approach ${afterTarget[0]}`);
        assert.ok(after.shares[0]>before.shares[0]+0.2,'resizing changes the actual initial region, not only its readout');
        assert.deepEqual(await camera(page),pose);
        assert.equal(await page.locator('.playhead .tick').textContent(),'iter 0');
        assert.equal(await page.locator('#program-status').getAttribute('data-state'),'ready');
        await page.screenshot({path:fileURLToPath(new URL('initial-area-after.png',folder))});
        await writeFile(new URL('initial-area-measurements.json',folder),JSON.stringify({beforeTarget,before,afterTarget,after},null,2));
    }finally{await browser.close();proc.kill();}
});
