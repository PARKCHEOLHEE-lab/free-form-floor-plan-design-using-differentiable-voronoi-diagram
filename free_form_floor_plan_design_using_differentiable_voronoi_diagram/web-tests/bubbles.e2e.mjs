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
// Hover the room at `a` and drag its link dot onto the room at `b`.
async function link(page,a,b){await page.mouse.move(...a);await drag(page,await middle(page.locator('.bubble-link-dot')),b);}

test('room circles draw move and resize on the plan while camera actions preserve their world positions', {timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        if(process.env.VORONOI_COVERAGE_DIR)await page.coverage.startJSCoverage({resetOnNavigation:false});
        await page.goto(url);
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('button[data-open=rooms]');await page.click('button[data-mode=shape_a]');
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        const countLabel = page.locator('.panel[data-panel=rooms] > .row > label').filter({hasText: /^Rooms$/});
        assert.equal(await countLabel.count(), 0, 'the panel omits the separate room count');
        await page.click('.gizmo [data-axis="+z"]');
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
        assert.equal(await page.locator('#roomareas .room-entry').count(),1);
        assert.equal(await page.locator('input#rooms,input[id^=area_]').count(),0,'circle editing replaces the old count and area sliders');
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
        // Empty points two radii from the room, on either side along its link dot (plan +y).
        p=await middle(first);await page.mouse.move(...p);
        const dot=await middle(page.locator('.bubble-link-dot'));
        const start=[3*p[0]-2*dot[0],3*p[1]-2*dot[1]],beyond=[2*dot[0]-p[0],2*dot[1]-p[1]];
        await drag(page,start,[start[0]+20,start[1]]);
        assert.equal((await labels(page)).length,2,'drawing still works after changing the camera');
        const second=await middle(page.locator('.bubble-label:not(.draft)').last());
        assert.ok(Math.hypot(second[0]-start[0],second[1]-start[1])<3,'screen picking and the 3D projection agree after orbit and zoom');
        assert.ok((await labels(page)).every(checkCircle));
        assert.ok(Math.abs((await labels(page)).reduce((s,r)=>s+r.ratio,0)-1)<1e-12);
        const saved=await labels(page);p=await middle(first);
        await page.mouse.move(...beyond);await page.mouse.down();await page.mouse.move(beyond[0]-20,beyond[1],{steps:5});
        assert.equal(await page.locator('.bubble-label.draft').count(),1,'a new circle is being drawn');
        await page.keyboard.press('Escape');await page.mouse.up();
        assert.deepEqual(await labels(page),saved,'Escape cancels the unfinished circle');
        await drag(page,[1050,650],[1090,650]);
        assert.equal((await labels(page)).length,2,'starting outside the boundary creates no room');
        const screens=new URL('../../.agents/cache/screens/',import.meta.url);await mkdir(screens,{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('bubbles-edit.png',screens))});
        if(process.env.VORONOI_COVERAGE_DIR){
            const data=(await page.coverage.stopJSCoverage()).filter(e=>e.url===url||e.url.includes('/plan/'));
            await mkdir(process.env.VORONOI_COVERAGE_DIR,{recursive:true});
            await writeFile(join(process.env.VORONOI_COVERAGE_DIR,'bubbles-edit.json'),JSON.stringify(data));
        }
    }finally{await browser.close();proc.kill();}
});

test('generated plans keep centers and show only requested doors in raw and aligned views', {timeout:180000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        if(process.env.VORONOI_COVERAGE_DIR)await page.coverage.startJSCoverage({resetOnNavigation:false});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('button[data-open=rooms]');await page.click('button[data-mode=shape_a]');
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('.gizmo [data-axis="+z"]');
        await drag(page,[550,530],[585,530]);await drag(page,[780,530],[810,530]);
        await link(page,[550,530],[780,530]);
        await page.click('button[data-open=optimize]');await page.locator('#iters').focus();await page.keyboard.press('Home');
        assert.equal(await page.locator('#iters').inputValue(),'50');
        await page.click('#run');
        await page.waitForFunction(()=>document.querySelector('#run').getAttribute('aria-label')==='Run',null,{timeout:120000});
        assert.equal(await page.locator('#program-status').getAttribute('data-state'),'complete',await page.locator('#program-status').textContent());
        assert.equal(await page.$eval('#plan',c=>+c.dataset.doors),1,'the actual renderer builds the requested physical door');
        await page.click('button[data-open=rooms]');await page.uncheck('#show-bubbles');
        const colors=await page.$eval('#plan',async c=>{
            const image=new Image();image.src=c.toDataURL();await image.decode();
            const copy=document.createElement('canvas');copy.width=image.width;copy.height=image.height;
            const ctx=copy.getContext('2d');ctx.drawImage(image,0,0);
            const r=c.getBoundingClientRect();
            return [[550,530,[191,228,191]],[780,530,[222,215,234]]].map(([screenX,screenY,color])=>{
                const x=(screenX-r.left)*copy.width/r.width,y=(screenY-r.top)*copy.height/r.height;
                let n=0;for(let i=0;i<8;i++){
                    const a=i*Math.PI/4,px=ctx.getImageData(Math.round(x+5*Math.cos(a)),Math.round(y+5*Math.sin(a)),1,1).data;
                    if(Math.abs(px[0]-color[0])+Math.abs(px[1]-color[1])+Math.abs(px[2]-color[2])<20)n++;
                }return n;
            });
        });
        assert.ok(colors.every(n=>n>=3),'each drawn center is inside its actual rendered room color, without the bubble overlay');
        const withDoor=await page.$eval('#plan',c=>c.toDataURL());
        await page.uncheck('#show-doors');
        assert.equal(await page.$eval('#plan',c=>+c.dataset.doors),0);
        assert.notEqual(await page.$eval('#plan',c=>c.toDataURL()),withDoor,'the door is visible geometry, not only a result counter');
        await page.check('#show-doors');await page.locator('#show-aligned').check();
        assert.equal(await page.locator('#program-status').getAttribute('data-state'),'complete',await page.locator('#program-status').textContent());
        assert.equal(await page.$eval('#plan',c=>+c.dataset.doors),1,'aligned shared walls preserve the required door');
        await mkdir(new URL('../../.agents/cache/screens/',import.meta.url),{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('../../.agents/cache/screens/bubbles-result-top.png',import.meta.url))});
        await page.click('.gizmo [data-axis="+x"]');
        await page.screenshot({path:fileURLToPath(new URL('../../.agents/cache/screens/bubbles-result-3d.png',import.meta.url))});
        await page.check('#show-bubbles');
        await page.locator('.connection-row').click();await page.keyboard.press('Delete');
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        assert.equal(await page.$eval('#plan',c=>+c.dataset.doors),0,'unlinked adjacent rooms have no door');
        assert.notEqual(await page.locator('#program-status').getAttribute('data-state'),'complete','edited inputs do not inherit an old completion state');
        if(process.env.VORONOI_COVERAGE_DIR){
            const data=(await page.coverage.stopJSCoverage()).filter(e=>e.url===url||e.url.includes('/plan/'));
            await mkdir(process.env.VORONOI_COVERAGE_DIR,{recursive:true});
            await writeFile(join(process.env.VORONOI_COVERAGE_DIR,'bubbles-result.json'),JSON.stringify(data));
        }
    }finally{await browser.close();proc.kill();}
});

test('linking and deleting circles updates visible door links and keeps the remaining room identity', {timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        if(process.env.VORONOI_COVERAGE_DIR)await page.coverage.startJSCoverage({resetOnNavigation:false});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('button[data-open=rooms]');await page.click('button[data-mode=shape_a]');
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('.gizmo [data-axis="+z"]');
        await drag(page,[550,530],[585,530]);await drag(page,[780,530],[810,530]);
        assert.equal((await labels(page)).length,2);
        await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        const rooms=await labels(page);
        const strip=()=>page.$eval('#plan',async c=>{
            const image=new Image();image.src=c.toDataURL();await image.decode();
            const copy=document.createElement('canvas');copy.width=image.width;copy.height=image.height;
            const ctx=copy.getContext('2d');ctx.drawImage(image,0,0);
            const r=c.getBoundingClientRect();
            const sx=copy.width/r.width,sy=copy.height/r.height;
            const px=ctx.getImageData((610-r.left)*sx,(528-r.top)*sy,110*sx,5*sy).data;let count=0;
            for(let i=0;i<px.length;i+=4)if(px[i]<140&&px[i+1]<140&&px[i+2]<140)count++;
            return count;
        });
        const before=await strip();
        await link(page,[550,530],[780,530]);
        assert.equal(await page.locator('.connection-row').count(),1,'dragging one room\'s dot to another creates their door connection');
        assert.ok(await strip()>before+30,'the connection is drawn in the actual plan canvas');
        const edge=page.locator('.connection-row');
        assert.equal(await edge.textContent(), `Room ${rooms[0].id} / Room ${rooms[1].id}`);
        assert.equal(await edge.getAttribute('aria-label'), `Select door link Room ${rooms[0].id} to Room ${rooms[1].id}`);
        assert.equal(+await edge.getAttribute('data-from'),rooms[0].id);
        assert.equal(+await edge.getAttribute('data-to'),rooms[1].id);
        await mkdir(new URL('../../.agents/cache/screens/',import.meta.url),{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('../../.agents/cache/screens/bubbles-linked.png',import.meta.url))});
        await link(page,[550,530],[780,530]);
        assert.equal(await edge.count(),1,'the same pair cannot acquire duplicate door links');
        await edge.click();
        assert.equal(await page.locator('.connection-row.selected').count(),1,'the sidebar selects a link too');
        await page.mouse.click(665,450);
        assert.equal(await page.locator('.connection-row.selected').count(),0,'a click on empty space inside the boundary clears the selection');
        await page.mouse.click(665,530);
        assert.equal(await page.locator('.connection-row.selected').count(),1,'clicking the rendered line selects it');
        await page.keyboard.press('Delete');
        assert.equal(await edge.count(),0,'the selected connection can be deleted without deleting its rooms');
        assert.equal((await labels(page)).length,2);
        await link(page,[550,530],[780,530]);
        assert.equal(await edge.count(),1);
        const remainingColor=await page.locator(`.room-row[data-room-id="${rooms[1].id}"] .sw`).evaluate(e=>getComputedStyle(e).backgroundColor);
        await page.locator(`.room-row[data-room-id="${rooms[0].id}"]`).click();await page.keyboard.press('Delete');
        assert.equal(await edge.count(),0,'deleting a room removes all of its links');
        const remaining=await labels(page);
        assert.equal(remaining.length,1);assert.equal(remaining[0].id,rooms[1].id);
        assert.equal(remaining[0].ratio,1);
        assert.equal(await page.locator('.bubble-label:not(.draft)').textContent(),`Room ${rooms[1].id}`,'room labels do not change after deleting another room');
        assert.equal(await page.getByRole('button', { name: `Select Room ${rooms[1].id}`, exact: true }).count(), 1);
        assert.equal(await page.locator(`.room-row[data-room-id="${rooms[1].id}"] .sw`).evaluate(e=>getComputedStyle(e).backgroundColor),remainingColor);
        await page.mouse.click(...await middle(page.locator('.bubble-label:not(.draft)')));await page.keyboard.press('Delete');
        assert.equal((await labels(page)).length,0);assert.equal(await page.locator('#run').isDisabled(),true);
        assert.equal(await page.locator('#roomareas .room-entry').count(),0);
        if(process.env.VORONOI_COVERAGE_DIR){
            const data=(await page.coverage.stopJSCoverage()).filter(e=>e.url===url||e.url.includes('/plan/'));
            await mkdir(process.env.VORONOI_COVERAGE_DIR,{recursive:true});
            await writeFile(join(process.env.VORONOI_COVERAGE_DIR,'bubbles-links.json'),JSON.stringify(data));
        }
    }finally{await browser.close();proc.kill();}
});

test('bubble diagram: with Rooms open, empty space draws, a room moves and its handle resizes, with no tool buttons',{timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        assert.equal(await page.locator('[data-bubble-tool]').count(),0,'there are no bubble tool buttons');
        await page.click('button[data-open=rooms]');await page.click('button[data-mode=shape_a]');
        await page.waitForFunction(()=>{const s=document.querySelector('#status').textContent;return s.startsWith('shape_a')&&s.includes('ready');});
        await page.click('.gizmo [data-axis="+z"]');
        await page.click('button[data-open=rooms]');
        await drag(page,[600,450],[650,450]);
        assert.equal((await labels(page)).length,0,'with Rooms closed a drag draws nothing');
        await page.click('button[data-open=rooms]');
        await drag(page,[600,450],[650,450]);
        assert.equal((await labels(page)).length,1,'with Rooms open a drag on empty space draws a room');
        const first=page.locator('.bubble-label:not(.draft)').first();
        let p=await middle(first);
        const original=(await labels(page))[0];
        await drag(page,p,[p[0]+20,p[1]-20]);
        p=await middle(first);
        assert.ok(Math.hypot(p[0]-620,p[1]-430)<3,'dragging a room moves it');
        assert.equal((await labels(page)).length,1,'moving a room does not draw another one');
        const handle=await middle(page.locator('.bubble-handle'));
        await drag(page,handle,[handle[0]+30,handle[1]]);
        assert.ok((await labels(page))[0].radius>original.radius,'dragging the round handle resizes the room');
        assert.equal((await labels(page)).length,1);
    }finally{await browser.close();proc.kill();}
});

test('bubble diagram: a hovered room shows a link dot, and dragging it onto another room adds a door',{timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('button[data-open=rooms]');await page.click('button[data-mode=shape_a]');
        await page.waitForFunction(()=>{const s=document.querySelector('#status').textContent;return s.startsWith('shape_a')&&s.includes('ready');});
        await page.click('.gizmo [data-axis="+z"]');
        await drag(page,[550,530],[585,530]);await drag(page,[780,530],[810,530]);
        assert.equal((await labels(page)).length,2);
        const dot=page.locator('.bubble-link-dot[data-id="1"]');
        await page.mouse.move(550,530);
        assert.equal(await dot.count(),1,'hovering room 1 shows its link dot');
        const [x,y]=await middle(dot);
        const room=(await labels(page)).find(r=>r.id===1);
        assert.ok(y<530,'the dot sits above the room center');
        await drag(page,[x,y],[700,690]);
        assert.equal(await page.locator('.connection-row').count(),0,'dropping the dot on empty space adds no door');
        assert.deepEqual((await labels(page)).find(r=>r.id===1),room,'dragging the dot does not move the room');
        await page.mouse.move(550,530);
        await drag(page,await middle(dot),[780,530]);
        assert.equal(await page.locator('.connection-row').count(),1,'dropping the dot on room 2 adds a door');
        assert.equal((await labels(page)).length,2,'linking draws no new room');
    }finally{await browser.close();proc.kill();}
});

test('bubble diagram: an Editing Bubble Diagram badge and a crosshair show only while Rooms is open',{timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        const badge=page.locator('#bubble-editing');
        const cursor=()=>page.$eval('#plan',c=>getComputedStyle(c).cursor);
        assert.equal(await badge.isVisible(),false,'no badge before Rooms opens');
        await page.click('button[data-open=rooms]');
        assert.equal(await badge.isVisible(),true,'the badge shows while Rooms is open');
        assert.equal((await badge.textContent()).trim(),'Editing Bubble Diagram');
        assert.equal(await cursor(),'crosshair','the plan cursor shows that drawing is on');
        await page.click('button[data-open=optimize]');
        assert.equal(await badge.isVisible(),false,'the badge hides when another panel opens');
        assert.notEqual(await cursor(),'crosshair');
        await page.click('button[data-open=rooms]');await page.click('button[data-open=rooms]');
        assert.equal(await badge.isVisible(),false,'the badge hides when Rooms closes');
    }finally{await browser.close();proc.kill();}
});

test('bubble diagram: the undo shortcut and the Undo button step back through add, move and link, and redo steps forward',{timeout:120000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('button[data-open=rooms]');await page.click('button[data-mode=shape_a]');
        await page.waitForFunction(()=>{const s=document.querySelector('#status').textContent;return s.startsWith('shape_a')&&s.includes('ready');});
        await page.click('.gizmo [data-axis="+z"]');
        const presetRooms=await page.locator('#roomareas .room-entry').count();
        const count=n=>page.waitForFunction(n=>document.querySelectorAll('.bubble-label:not(.draft)').length===n,n);
        const doors=n=>page.waitForFunction(n=>document.querySelectorAll('.connection-row').length===n,n);
        assert.equal(await page.locator('#undo').isDisabled(),true,'nothing to undo at first');
        await drag(page,[550,530],[585,530]);await drag(page,[780,530],[810,530]);
        const second=(await labels(page)).find(r=>r.id===2);
        await drag(page,[780,530],[800,500]);
        const moved=(await labels(page)).find(r=>r.id===2);
        assert.notDeepEqual(moved.center,second.center);
        await page.mouse.move(550,530);
        await drag(page,await middle(page.locator('.bubble-link-dot[data-id="1"]')),[800,500]);
        await doors(1);
        await page.keyboard.press('ControlOrMeta+z');
        await doors(0);
        await page.click('#undo');
        await page.waitForFunction(c=>{const e=document.querySelector('.bubble-label[data-id="2"]');return e&&+e.dataset.x===c[0]&&+e.dataset.y===c[1];},second.center);
        await page.keyboard.press('ControlOrMeta+z');await count(1);
        await page.keyboard.press('ControlOrMeta+z');await count(0);
        await page.waitForFunction(n=>document.querySelectorAll('#roomareas .room-entry').length===n,presetRooms);
        assert.equal(await page.locator('#undo').isDisabled(),true,'back at the preset rooms with nothing left to undo');
        await page.keyboard.press('ControlOrMeta+Shift+z');await count(1);
        await page.click('#redo');await count(2);
        await page.keyboard.press('ControlOrMeta+Shift+z');
        await page.waitForFunction(c=>{const e=document.querySelector('.bubble-label[data-id="2"]');return e&&+e.dataset.x===c[0]&&+e.dataset.y===c[1];},moved.center);
        await page.keyboard.press('ControlOrMeta+z');
        await page.mouse.move(550,530);
        await drag(page,[600,450],[620,450]);await count(3);
        assert.equal(await page.locator('#redo').isDisabled(),true,'a new edit drops the undone steps');
    }finally{await browser.close();proc.kill();}
});

test('bubble diagram: the Delete key removes the selected room or door even after a panel button was clicked',{timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('button[data-open=rooms]');await page.click('button[data-mode=shape_a]');
        await page.waitForFunction(()=>{const s=document.querySelector('#status').textContent;return s.startsWith('shape_a')&&s.includes('ready');});
        await page.click('.gizmo [data-axis="+z"]');
        await drag(page,[550,530],[585,530]);await drag(page,[780,530],[810,530]);await drag(page,[600,450],[620,450]);
        await link(page,[550,530],[780,530]);
        await page.locator('.connection-row').click();
        await page.click('#undo');await page.click('#redo');
        await page.locator('.connection-row').click();await page.locator('.panel[data-panel=rooms] .label').first().click();
        await page.keyboard.press('Delete');
        assert.equal(await page.locator('.connection-row').count(),0,'Delete removes the selected door link');
        await page.mouse.click(600,450);await page.click('#undo');await page.click('#redo');
        await page.locator('.room-row[data-room-id="3"]').click();await page.locator('#wall').focus();await page.locator('.panel[data-panel=rooms] .label').first().click();
        await page.keyboard.press('Backspace');
        assert.deepEqual((await labels(page)).map(r=>r.id),[1,2],'Backspace (the Mac delete key) removes the selected room');
    }finally{await browser.close();proc.kill();}
});

test('bubble diagram: a click or drag outside the boundary changes nothing, shows no error and keeps the selection',{timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        await page.goto(url);await page.waitForFunction(()=>document.querySelector('#status').textContent.includes('ready'));
        await page.click('button[data-open=rooms]');await page.click('button[data-mode=shape_a]');
        await page.waitForFunction(()=>{const s=document.querySelector('#status').textContent;return s.startsWith('shape_a')&&s.includes('ready');});
        await page.click('.gizmo [data-axis="+z"]');
        const outside=[1100,800],message=page.locator('#bubble-message');
        const entries=page.locator('#roomareas .room-entry');
        const help=await message.textContent(),presetRooms=await entries.count();
        await page.mouse.click(...outside);
        assert.equal(await entries.count(),presetRooms,'an outside click keeps the preset rooms');
        assert.equal(await message.textContent(),help,'an outside click shows no message');
        assert.equal(await message.evaluate(m=>m.classList.contains('error')),false);
        await drag(page,[600,450],[620,450]);
        assert.equal(await page.locator('.room-row.selected').count(),1,'the new room is selected');
        await page.mouse.click(...outside);
        assert.equal(await page.locator('.room-row.selected').count(),1,'an outside click keeps the selection');
        await drag(page,outside,[outside[0]-40,outside[1]]);
        assert.equal((await labels(page)).length,1,'an outside drag draws nothing');
        assert.equal(await message.textContent(),help,'an outside drag shows no message');
    }finally{await browser.close();proc.kill();}
});
