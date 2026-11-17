import {test} from 'node:test';
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
import {serve} from './serve.mjs';
import {inside,segmentDistance} from '../web/plan/geom.js';

const labels=page=>page.$$eval('.bubble-label:not(.draft)',els=>els.map(e=>({id:+e.dataset.id,center:[+e.dataset.x,+e.dataset.y],radius:+e.dataset.radius,ratio:+e.dataset.ratio})));
const middle=async locator=>{const b=await locator.boundingBox();assert.ok(b);return[b.x+b.width/2,b.y+b.height/2];};
async function drag(page,a,b,button='left'){
    await page.mouse.move(...a);await page.mouse.down({button});await page.mouse.move(...b,{steps:12});await page.mouse.up({button});
}

test('room circles draw move and resize on the plan while camera actions preserve their world positions', {timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        if(process.env.VORONOI_COVERAGE_DIR)await page.coverage.startJSCoverage({resetOnNavigation:false});
        await page.goto(url);
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('button[data-open=boundary]');await page.click('button[data-mode=shape_a]');
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('.gizmo [data-axis="+z"]');await page.click('button[data-open=rooms]');
        await page.click('button[data-bubble-tool=draw]');
        await page.setViewportSize({width:1146,height:820});await page.waitForTimeout(200);
        await page.setViewportSize({width:1145,height:820});await page.waitForTimeout(200);
        const boundaryInk=await page.$eval('#plan',async c=>{
            const image=new Image();image.src=c.toDataURL();await image.decode();
            const copy=document.createElement('canvas');copy.width=image.width;copy.height=image.height;
            const ctx=copy.getContext('2d');ctx.drawImage(image,0,0);
            const pixels=ctx.getImageData(0,0,copy.width,copy.height).data;let count=0;
            for(let i=0;i<pixels.length;i+=4)if(pixels[i]<80&&pixels[i+1]<80&&pixels[i+2]<80)count++;
            return count;
        });
        assert.ok(boundaryInk>100,'the boundary remains visible after resizing an empty room diagram');
        const before=await page.$eval('#plan',c=>c.toDataURL());
        await page.mouse.move(600,450);await page.mouse.down();await page.mouse.move(650,450,{steps:10});
        const drawing=await page.$eval('#plan',c=>c.toDataURL());
        await page.mouse.up();await page.waitForTimeout(150);
        assert.equal((await labels(page)).length,1,'dragging inside the plan creates one room circle');
        assert.notEqual(drawing,before,'the circle is actually rendered during the drag before any engine rebuild');
        const first=page.locator('.bubble-label:not(.draft)').first();
        let p=await middle(first);
        assert.ok(Math.hypot(p[0]-600,p[1]-450)<3,'the drawn center stays under the original pointer');
        assert.equal((await labels(page))[0].ratio,1);
        assert.equal(await page.locator('#rooms_v').textContent(),'1');
        assert.equal(await page.locator('input#rooms,input[id^=area_]').count(),0,'circle editing replaces the old count and area sliders');
        await page.click('button[data-bubble-tool=select]');
        const original=(await labels(page))[0];
        await drag(page,p,[p[0]+20,p[1]-20]);
        p=await middle(first);
        assert.ok(Math.hypot(p[0]-620,p[1]-430)<3,'a room follows a drag without snapping its center to the pointer');
        assert.equal((await labels(page))[0].radius,original.radius);
        const handle=await middle(page.locator('.bubble-handle'));
        await drag(page,handle,[handle[0]+30,handle[1]]);
        assert.ok((await labels(page))[0].radius>original.radius,'the round edge handle resizes the circle');
        p=await middle(first);await drag(page,p,[1050,100]);
        const boundary=JSON.parse(await readFile(new URL('./fixtures/shape_a.json',import.meta.url),'utf8')).boundary;
        const checkCircle=r=>inside(r.center,boundary)&&boundary.every((a,i)=>segmentDistance(r.center,a,boundary[(i+1)%boundary.length])+1e-9>=r.radius);
        assert.ok((await labels(page)).every(checkCircle),'an invalid move retains a circle wholly inside the concave boundary');
        p=await middle(first);await drag(page,p,[600,450]);
        const world=await labels(page);
        await page.mouse.dblclick(1000,100);await page.mouse.move(620,420);await page.mouse.wheel(0,-250);
        await drag(page,[650,400],[710,420],'right');
        assert.deepEqual(await labels(page),world,'orbiting and zooming do not edit room data');
        p=await middle(first);const start=[p[0]+15,p[1]];
        await page.click('button[data-bubble-tool=draw]');
        await drag(page,start,[start[0]+20,start[1]]);
        assert.equal((await labels(page)).length,2,'drawing still works after changing the camera');
        const second=await middle(page.locator('.bubble-label:not(.draft)').last());
        assert.ok(Math.hypot(second[0]-start[0],second[1]-start[1])<3,'screen picking and the 3D projection agree after orbit and zoom');
        assert.ok((await labels(page)).every(checkCircle));
        assert.ok(Math.abs((await labels(page)).reduce((s,r)=>s+r.ratio,0)-1)<1e-12);
        const saved=await labels(page);p=await middle(first);
        await page.mouse.move(p[0]-10,p[1]+10);await page.mouse.down();await page.mouse.move(p[0]-20,p[1]+10,{steps:5});
        await page.keyboard.press('Escape');await page.mouse.up();
        assert.deepEqual(await labels(page),saved,'Escape cancels the unfinished circle');
        await drag(page,[1050,650],[1090,650]);
        assert.equal((await labels(page)).length,2,'starting outside the boundary creates no room');
        await page.click('button[data-bubble-tool=select]');
        const screens=new URL('../../.agents/cache/screens/',import.meta.url);await mkdir(screens,{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('bubbles-edit.png',screens))});
        if(process.env.VORONOI_COVERAGE_DIR){
            const data=(await page.coverage.stopJSCoverage()).filter(e=>e.url===url||e.url.includes('/plan/'));
            await mkdir(process.env.VORONOI_COVERAGE_DIR,{recursive:true});
            await writeFile(join(process.env.VORONOI_COVERAGE_DIR,'bubbles-edit.json'),JSON.stringify(data));
        }
    }finally{await browser.close();proc.kill();}
});

test('linking and deleting circles updates visible door links and keeps the remaining room identity', {timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        if(process.env.VORONOI_COVERAGE_DIR)await page.coverage.startJSCoverage({resetOnNavigation:false});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('button[data-open=boundary]');await page.click('button[data-mode=shape_a]');
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('.gizmo [data-axis="+z"]');await page.click('button[data-open=rooms]');await page.click('button[data-bubble-tool=draw]');
        await drag(page,[550,530],[585,530]);await drag(page,[780,530],[810,530]);
        assert.equal((await labels(page)).length,2);
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        const rooms=await labels(page);
        const strip=()=>page.$eval('#plan',async c=>{
            const image=new Image();image.src=c.toDataURL();await image.decode();
            const copy=document.createElement('canvas');copy.width=image.width;copy.height=image.height;
            const ctx=copy.getContext('2d');ctx.drawImage(image,0,0);
            const px=ctx.getImageData(610,528,110,5).data;let count=0;
            for(let i=0;i<px.length;i+=4)if(px[i]<140&&px[i+1]<140&&px[i+2]<140)count++;
            return count;
        });
        const before=await strip();
        await page.click('button[data-bubble-tool=link]');
        await page.mouse.click(550,530);await page.mouse.click(780,530);
        assert.equal(await page.locator('.connection-row').count(),1,'clicking two circles creates their door connection');
        assert.ok(await strip()>before+30,'the connection is drawn in the actual plan canvas');
        const edge=page.locator('.connection-row');
        assert.equal(+await edge.getAttribute('data-from'),rooms[0].id);
        assert.equal(+await edge.getAttribute('data-to'),rooms[1].id);
        await mkdir(new URL('../../.agents/cache/screens/',import.meta.url),{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('../../.agents/cache/screens/bubbles-linked.png',import.meta.url))});
        await page.mouse.click(550,530);await page.mouse.click(780,530);
        assert.equal(await edge.count(),1,'the same pair cannot acquire duplicate door links');
        await edge.click();
        assert.equal(await page.locator('#bubble-delete').isEnabled(),true,'the sidebar selects a link too');
        await page.click('button[data-bubble-tool=select]');await page.mouse.click(1050,650);
        assert.equal(await page.locator('#bubble-delete').isDisabled(),true);
        await page.mouse.click(665,530);
        assert.equal(await page.locator('#bubble-delete').isEnabled(),true,'clicking the rendered line selects it');
        await page.click('#bubble-delete');
        assert.equal(await edge.count(),0,'the selected connection can be deleted without deleting its rooms');
        assert.equal((await labels(page)).length,2);
        await page.click('button[data-bubble-tool=link]');await page.mouse.click(550,530);await page.mouse.click(780,530);
        assert.equal(await edge.count(),1);
        const remainingColor=await page.locator(`.room-row[data-room-id="${rooms[1].id}"] .sw`).evaluate(e=>getComputedStyle(e).backgroundColor);
        await page.locator(`.room-row[data-room-id="${rooms[0].id}"]`).click();await page.click('#bubble-delete');
        assert.equal(await edge.count(),0,'deleting a room removes all of its links');
        const remaining=await labels(page);
        assert.equal(remaining.length,1);assert.equal(remaining[0].id,rooms[1].id);
        assert.equal(remaining[0].ratio,1);
        assert.equal(await page.locator('.bubble-label:not(.draft)').textContent(),`R${rooms[1].id}`,'room names do not change after deleting another room');
        assert.equal(await page.locator(`.room-row[data-room-id="${rooms[1].id}"] .sw`).evaluate(e=>getComputedStyle(e).backgroundColor),remainingColor);
        await page.mouse.click(...await middle(page.locator('.bubble-label:not(.draft)')));await page.keyboard.press('Delete');
        assert.equal((await labels(page)).length,0);assert.equal(await page.locator('#run').isDisabled(),true);
        assert.equal(await page.locator('#rooms_v').textContent(),'0');
        if(process.env.VORONOI_COVERAGE_DIR){
            const data=(await page.coverage.stopJSCoverage()).filter(e=>e.url===url||e.url.includes('/plan/'));
            await mkdir(process.env.VORONOI_COVERAGE_DIR,{recursive:true});
            await writeFile(join(process.env.VORONOI_COVERAGE_DIR,'bubbles-links.json'),JSON.stringify(data));
        }
    }finally{await browser.close();proc.kill();}
});
