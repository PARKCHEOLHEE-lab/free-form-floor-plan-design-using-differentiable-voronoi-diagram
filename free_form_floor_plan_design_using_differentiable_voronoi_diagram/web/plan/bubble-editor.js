import {inside} from './geom.js';

/** Left-button room editing. The view owns projection and the program owns validity. */
export function createBubbleEditor(canvas,view,{labels,colors,onChange,onStatus}) {
    let model=null,tool='select',selected=null,press=null,draft=null;
    const empty={rooms:[],links:[],ratios:[]};
    const snapshot=()=>model?model.snapshot():empty;
    const emit=commit=>onChange({commit,snapshot:snapshot()});

    function layout() {
        for(const el of labels.children){
            const p=view.screenPoint([+el.dataset.x,+el.dataset.y]);
            el.style.left=p[0]+'px';el.style.top=p[1]+'px';
        }
    }
    function refresh() {
        const s=snapshot();
        view.drawBubbles(s.rooms.map((r,i)=>({...r,color:colors[i%colors.length]})),selected,draft?{...draft,color:colors[s.rooms.length%colors.length]}:null);
        labels.replaceChildren();
        const shown=draft?[...s.rooms,{...draft,id:'draft'}]:s.rooms;
        shown.forEach((r,i)=>{
            const el=document.createElement('span');
            el.className='bubble-label'+(r.id===selected?' selected':'')+(r.id==='draft'?' draft':'');
            Object.assign(el.dataset,{id:r.id,x:r.center[0],y:r.center[1],radius:r.radius,ratio:s.ratios[i]??''});
            el.textContent=r.id==='draft'?'+':`R${i+1}`;
            labels.append(el);
        });
        const r=s.rooms.find(r=>r.id===selected);
        if(r&&tool==='select'){
            const handle=document.createElement('span');handle.className='bubble-handle';
            handle.dataset.x=r.center[0]+r.radius;handle.dataset.y=r.center[1];
            handle.setAttribute('aria-label','Resize selected room');labels.append(handle);
        }
        layout();
    }
    function down(e) {
        if(e.button!==0||!model)return;
        const p=view.pointOnPlan(e.clientX,e.clientY);if(!p)return;
        const s=snapshot();
        if(tool==='draw'){
            if(!inside(p,model.boundary)){onStatus('Start the circle inside the boundary.');return;}
            selected=null;press={kind:'draw',start:p,center:p};
        }else{
            const active=s.rooms.find(r=>r.id===selected);
            const hp=active?view.screenPoint([active.center[0]+active.radius,active.center[1]]):null;
            if(hp&&Math.hypot(e.clientX-hp[0],e.clientY-hp[1])<=9) press={kind:'resize',original:active,start:p};
            else{
                const hit=s.rooms.map(r=>({r,d:Math.hypot(p[0]-r.center[0],p[1]-r.center[1])})).filter(h=>h.d<=h.r.radius).sort((a,b)=>a.d-b.d)[0];
                selected=hit?hit.r.id:null;press=hit?{kind:'move',original:hit.r,start:p}:null;
            }
        }
        if(press){
            Object.assign(press,{id:e.pointerId,screen:[e.clientX,e.clientY],started:false});
            canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);e.preventDefault();
        }
        refresh();
    }
    function move(e) {
        if(!press||e.pointerId!==press.id)return;
        const p=view.pointOnPlan(e.clientX,e.clientY);if(!p)return;
        if(!press.started){
            if(Math.hypot(e.clientX-press.screen[0],e.clientY-press.screen[1])<3)return;
            press.started=true;emit(false);
        }
        let valid;
        if(press.kind==='draw'){
            const radius=Math.hypot(p[0]-press.center[0],p[1]-press.center[1]);
            valid=model.canPlace(press.center,radius);
            if(valid)draft={center:press.center,radius};
        }else{
            const r=press.original;
            const center=press.kind==='move'?[r.center[0]+p[0]-press.start[0],r.center[1]+p[1]-press.start[1]]:r.center;
            const radius=press.kind==='resize'?r.radius+Math.hypot(p[0]-r.center[0],p[1]-r.center[1])-Math.hypot(press.start[0]-r.center[0],press.start[1]-r.center[1]):r.radius;
            valid=model.update(r.id,center,radius);
        }
        onStatus(valid?'':model.error);
        if(valid)emit(false);
        refresh();
    }
    function finish(cancel=false) {
        if(!press)return;
        const current=press;press=null;
        if(current.started){
            if(cancel&&current.original)model.update(current.original.id,current.original.center,current.original.radius);
            if(!cancel&&current.kind==='draw'&&draft)selected=model.add(draft.center,draft.radius);
            draft=null;emit(true);
        }
        if(canvas.hasPointerCapture(current.id))canvas.releasePointerCapture(current.id);
        refresh();
    }
    const up=e=>{if(press?.id===e.pointerId)finish();};
    const cancel=e=>{if(press?.id===e.pointerId)finish(true);};
    const key=e=>{if(e.key==='Escape'){finish(true);onStatus('');}};
    canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointermove',move);
    canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',cancel);canvas.addEventListener('keydown',key);
    const unsubscribe=view.onChange(layout);
    refresh();
    return {
        get tool(){return tool;},
        get selected(){return selected;},
        setProgram(next){if(model===next)return;press=null;draft=null;selected=null;model=next;refresh();},
        setTool(next){finish(true);tool=next;canvas.style.cursor=next==='draw'?'crosshair':'default';onStatus('');refresh();},
        dispose(){unsubscribe();canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',cancel);canvas.removeEventListener('keydown',key);}
    };
}
