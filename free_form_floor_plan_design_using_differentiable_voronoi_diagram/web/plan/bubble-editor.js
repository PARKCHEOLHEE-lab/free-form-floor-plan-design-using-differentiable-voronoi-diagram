import {inside,segmentDistance} from './geom.js';

/**
 * Left-button room editing with no tool modes, active only while enabled (the Rooms panel is
 * open). What is under the pointer decides the action: the round handle resizes, the link dot on
 * a hovered room drags a door link to another room, a room moves, and empty plan draws a new room
 * (`ensureProgram` supplies a program on the first one). The view owns projection and the program
 * owns validity.
 */
export function createBubbleEditor(canvas,view,{labels,colorForRoomId,onChange,onStatus,ensureProgram}) {
    let model=null,enabled=false,selected=null,selectedLink=null,press=null,draft=null,visible=true,hover=null;
    const empty={rooms:[],links:[],ratios:[]};
    const snapshot=()=>model?model.snapshot():empty;
    const emit=commit=>onChange({commit,snapshot:snapshot()});
    const selectionChanged=()=>onChange({selectionOnly:true,snapshot:snapshot()});
    const pairKey=(a,b)=>a<b?[a,b]:[b,a];
    const roomAt=(p,s)=>s.rooms.map(r=>({r,d:Math.hypot(p[0]-r.center[0],p[1]-r.center[1])})).filter(h=>h.d<=h.r.radius).sort((a,b)=>a.d-b.d)[0]?.r;
    // The link dot sits on top of a room's circle (plan +y), on the hovered room only.
    const dotPoint=r=>view.screenPoint([r.center[0],r.center[1]+r.radius]);
    const onDot=(r,x,y)=>!!r&&Math.hypot(x-dotPoint(r)[0],y-dotPoint(r)[1])<=9;
    const svgNS='http://www.w3.org/2000/svg';
    const overlay=document.createElementNS(svgNS,'svg');
    overlay.classList.add('bubble-link-preview');
    labels.after(overlay);

    function layout() {
        for(const el of labels.children){
            const p=view.screenPoint([+el.dataset.x,+el.dataset.y]);
            el.style.left=p[0]+'px';el.style.top=p[1]+'px';
        }
    }
    function refresh() {
        canvas.toggleAttribute('data-bubble-editing',enabled&&visible);
        if(!visible){view.drawBubbles([],null,null);labels.replaceChildren();return;}
        const s=snapshot();
        const selectedEdge=s.links.findIndex(([a,b])=>selectedLink&&s.rooms[a].id===selectedLink[0]&&s.rooms[b].id===selectedLink[1]);
        view.drawBubbles(s.rooms.map(r=>({...r,color:colorForRoomId(r.id)})),selected,draft?{...draft,color:'#dddddd'}:null,s.links,selectedEdge);
        labels.replaceChildren();
        const shown=draft?[...s.rooms,{...draft,id:'draft'}]:s.rooms;
        shown.forEach((r,i)=>{
            const el=document.createElement('span');
            el.className='bubble-label'+(r.id===selected?' selected':'')+(r.id==='draft'?' draft':'');
            Object.assign(el.dataset,{id:r.id,x:r.center[0],y:r.center[1],radius:r.radius,ratio:s.ratios[i]??''});
            el.textContent=r.id==='draft'?'+':`R${r.id}`;
            labels.append(el);
        });
        const dotRoom=s.rooms.find(r=>r.id===(press?.kind==='link'?press.from.id:hover));
        if(dotRoom&&enabled){
            const dot=document.createElement('span');dot.className='bubble-link-dot';
            Object.assign(dot.dataset,{id:dotRoom.id,x:dotRoom.center[0],y:dotRoom.center[1]+dotRoom.radius});
            dot.setAttribute('aria-label','Drag to another room to add a door');labels.append(dot);
        }
        drawLinkPreview(s);
        const r=s.rooms.find(r=>r.id===selected);
        if(r&&enabled){
            const handle=document.createElement('span');handle.className='bubble-handle';
            handle.dataset.x=r.center[0]+r.radius;handle.dataset.y=r.center[1];
            handle.setAttribute('aria-label','Resize selected room');labels.append(handle);
        }
        layout();
    }
    // While a link drag runs: a line from the room to the pointer, solid once it reaches a room.
    function drawLinkPreview(s) {
        overlay.replaceChildren();
        if(press?.kind!=='link'||!press.started)return;
        const a=view.screenPoint(press.from.center);
        const target=s.rooms.find(r=>r.id===press.target);
        const b=target?view.screenPoint(target.center):press.pointer;
        const line=document.createElementNS(svgNS,'line');
        for(const [k,v] of Object.entries({x1:a[0],y1:a[1],x2:b[0],y2:b[1]}))line.setAttribute(k,v);
        line.classList.toggle('reached',!!target);
        overlay.append(line);
    }
    function down(e) {
        if(e.button!==0||!enabled||!visible)return;
        const p=view.pointOnPlan(e.clientX,e.clientY);if(!p)return;
        const s=snapshot();
        canvas.focus({preventScroll:true});
        selectedLink=null;
        const active=s.rooms.find(r=>r.id===selected);
        const hp=active?view.screenPoint([active.center[0]+active.radius,active.center[1]]):null;
        const hit=roomAt(p,s);
        const hovered=s.rooms.find(r=>r.id===hover);
        if(hp&&Math.hypot(e.clientX-hp[0],e.clientY-hp[1])<=9) press={kind:'resize',original:active,start:p};
        else if(onDot(hovered,e.clientX,e.clientY)) press={kind:'link',from:hovered,target:null,pointer:[e.clientX,e.clientY]};
        else if(hit){selected=hit.id;press={kind:'move',original:hit,start:p};}
        else{
            selected=null;
            const edge=s.links.find(([a,b])=>segmentDistance([e.clientX,e.clientY],view.screenPoint(s.rooms[a].center),view.screenPoint(s.rooms[b].center))<=6);
            if(edge)selectedLink=pairKey(s.rooms[edge[0]].id,s.rooms[edge[1]].id);
            else{
                model??=ensureProgram();
                if(model&&inside(p,model.boundary))press={kind:'draw',start:p,center:p};
                else if(model)onStatus('Start the circle inside the boundary.');
            }
        }
        if(press){
            Object.assign(press,{id:e.pointerId,screen:[e.clientX,e.clientY],started:false});
            canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);e.preventDefault();
        }
        selectionChanged();refresh();
    }
    function move(e) {
        if(!press){
            if(!enabled||!visible||!model)return;
            const s=snapshot(),p=view.pointOnPlan(e.clientX,e.clientY);
            const keep=s.rooms.find(r=>r.id===hover);
            const next=onDot(keep,e.clientX,e.clientY)?hover:(p?roomAt(p,s)?.id??null:null);
            if(next!==hover){hover=next;refresh();}
            return;
        }
        if(e.pointerId!==press.id)return;
        const p=view.pointOnPlan(e.clientX,e.clientY);if(!p)return;
        if(!press.started){
            if(Math.hypot(e.clientX-press.screen[0],e.clientY-press.screen[1])<3)return;
            press.started=true;if(press.kind!=='link')emit(false);
        }
        if(press.kind==='link'){
            const t=roomAt(p,snapshot());
            press.target=t&&t.id!==press.from.id?t.id:null;press.pointer=[e.clientX,e.clientY];
            drawLinkPreview(snapshot());return;
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
        if(current.kind==='link'){
            if(!cancel&&current.started&&current.target!==null){
                const added=model.connect(current.from.id,current.target);
                selected=null;selectedLink=pairKey(current.from.id,current.target);
                if(added)emit(true);else onStatus(model.error);
            }
        }else if(current.started){
            if(cancel&&current.original)model.update(current.original.id,current.original.center,current.original.radius);
            if(!cancel&&current.kind==='draw'&&draft)selected=model.add(draft.center,draft.radius);
            draft=null;emit(true);
        }
        if(canvas.hasPointerCapture(current.id))canvas.releasePointerCapture(current.id);
        refresh();
    }
    const up=e=>{if(press?.id===e.pointerId)finish();};
    const cancel=e=>{if(press?.id===e.pointerId)finish(true);};
    const leave=()=>{if(!press&&hover!==null){hover=null;refresh();}};
    function deleteSelection(){
        if(!model)return false;
        finish(true);
        const removed=selectedLink?model.disconnect(...selectedLink):selected!==null?model.remove(selected):false;
        if(!removed){onStatus('Select a room or door link first.');return false;}
        selected=null;selectedLink=null;emit(true);onStatus('');refresh();return true;
    }
    // Keys work from anywhere on the page while editing, except inside a form field.
    const key=e=>{
        if(!visible||!enabled||e.target.closest?.('input, textarea, select, [contenteditable]'))return;
        if(e.key==='Escape'){finish(true);selectedLink=null;onStatus('');refresh();}
        if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();deleteSelection();}
    };
    canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointermove',move);canvas.addEventListener('pointerleave',leave);
    canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',cancel);window.addEventListener('keydown',key);
    const unsubscribe=view.onChange(layout);
    refresh();
    return {
        get selected(){return selected;},
        get selectedLink(){return selectedLink?[...selectedLink]:null;},
        setVisible(next){finish(true);visible=next;refresh();},
        setProgram(next){if(model===next)return;press=null;draft=null;selected=null;selectedLink=null;model=next;refresh();},
        setEnabled(next){finish(true);enabled=next;hover=null;refresh();},
        selectRoom(id){selected=id;selectedLink=null;selectionChanged();canvas.focus({preventScroll:true});refresh();},
        selectLink(a,b){selected=null;selectedLink=pairKey(a,b);selectionChanged();canvas.focus({preventScroll:true});refresh();},
        deleteSelection,
        dispose(){unsubscribe();overlay.remove();canvas.removeEventListener('pointerleave',leave);canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',cancel);window.removeEventListener('keydown',key);}
    };
}
