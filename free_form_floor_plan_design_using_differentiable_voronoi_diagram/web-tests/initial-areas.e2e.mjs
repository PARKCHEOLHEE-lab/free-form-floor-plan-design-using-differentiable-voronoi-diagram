import {test} from 'node:test';
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {serve} from './serve.mjs';

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

test('resizing a circle changes actual initial room areas before Run without moving the camera',{timeout:90000},async()=>{
    const {url,proc}=await serve();const browser=await chromium.launch();
    try{
        const page=await browser.newPage({viewport:{width:1145,height:820}});
        await page.goto(url);await ready(page);
        await page.click('button[data-open=boundary]');await page.click('button[data-mode=shape_a]');await ready(page);
        await page.click('.gizmo [data-axis="+z"]');
        await page.click('button[data-open=rooms]');await page.click('button[data-bubble-tool=draw]');
        await drag(page,[550,530],[580,530]);await drag(page,[780,530],[840,530]);await ready(page);
        const pose=await camera(page),beforeTarget=await target(page);
        assert.ok(Math.abs(beforeTarget[0]-0.2)<1e-6);
        await page.click('button[data-open=display]');await page.uncheck('#show-bubbles');
        const before=await filledShares(page);
        assert.ok(before.pixels.every(n=>n>10000),'actual room fills, without circle overlays, are visible');
        assert.ok(Math.abs(before.shares[0]-beforeTarget[0])<=0.05,`initial rendered share ${before.shares[0]} must approach ${beforeTarget[0]}`);
        assert.equal(await page.locator('#run').getAttribute('aria-label'),'Run');
        assert.equal(await page.locator('.playhead .tick').textContent(),'iter 0','the preview has not optimized');
        const folder=new URL('../../.agents/cache/screens/',import.meta.url);await mkdir(folder,{recursive:true});
        await page.screenshot({path:fileURLToPath(new URL('initial-area-before.png',folder))});
        await page.check('#show-bubbles');await page.click('button[data-open=rooms]');await page.click('button[data-bubble-tool=select]');
        await page.mouse.click(550,530);
        const h=await page.locator('.bubble-handle').boundingBox();assert.ok(h);
        await drag(page,[h.x+h.width/2,h.y+h.height/2],[h.x+h.width/2+30,h.y+h.height/2]);await ready(page);
        const afterTarget=await target(page);
        await page.click('button[data-open=display]');await page.uncheck('#show-bubbles');
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
