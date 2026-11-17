import {inside,segmentDistance} from './geom.js';

/** Left-button room editing. The view owns projection and the program owns validity. */
export function createBubbleEditor(canvas,view,{labels,colorForRoomId,onChange,onStatus}) {
    let model=null,tool='select',selected=null,selectedLink=null,pendingLink=null,press=null,draft=null,visible=true;
    const empty={rooms:[],links:[],ratios:[]};
    const snapshot=()=>model?model.snapshot():empty;
    const emit=commit=>onChange({commit,snapshot:snapshot()});
    const selectionChanged=()=>onChange({selectionOnly:true,snapshot:snapshot()});
    const pairKey=(a,b)=>a<b?[a,b]:[b,a];
    const roomAt=(p,s)=>s.rooms.map(r=>({r,d:Math.hypot(p[0]-r.center[0],p[1]-r.center[1])})).filter(h=>h.d<=h.r.radius).sort((a,b)=>a.d-b.d)[0]?.r;

    function layout() {
        for(const el of labels.children){
            const p=view.screenPoint([+el.dataset.x,+el.dataset.y]);
            el.style.left=p[0]+'px';el.style.top=p[1]+'px';
        }
    }
    function refresh() {
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
        const r=s.rooms.find(r=>r.id===selected);
        if(r&&tool==='select'){
            const handle=document.createElement('span');handle.className='bubble-handle';
            handle.dataset.x=r.center[0]+r.radius;handle.dataset.y=r.center[1];
            handle.setAttribute('aria-label','Resize selected room');labels.append(handle);
        }
        layout();
    }
    function down(e) {
        if(e.button!==0||!model||!visible)return;
        const p=view.pointOnPlan(e.clientX,e.clientY);if(!p)return;
        const s=snapshot();
        canvas.focus({preventScroll:true});
        if(tool==='link'){
            const hit=roomAt(p,s);
            if(!hit){onStatus('Click a room circle.');return;}
            if(pendingLink===null){pendingLink=hit.id;selected=hit.id;selectedLink=null;onStatus('Click another room to add a door link.',false);}
            else if(pendingLink===hit.id){onStatus('Choose a different room.');return;}
            else{
                const a=pendingLink;pendingLink=null;
                const added=model.connect(a,hit.id);selected=null;selectedLink=pairKey(a,hit.id);
                if(added){emit(true);onStatus('Click two rooms to request a door.',false);}else onStatus(model.error);
            }
            selectionChanged();refresh();return;
        }
        selectedLink=null;
        if(tool==='draw'){
            if(!inside(p,model.boundary)){onStatus('Start the circle inside the boundary.');return;}
            selected=null;press={kind:'draw',start:p,center:p};
        }else{
            const active=s.rooms.find(r=>r.id===selected);
            const hp=active?view.screenPoint([active.center[0]+active.radius,active.center[1]]):null;
            if(hp&&Math.hypot(e.clientX-hp[0],e.clientY-hp[1])<=9) press={kind:'resize',original:active,start:p};
            else{
                const hit=roomAt(p,s);
                selected=hit?hit.id:null;press=hit?{kind:'move',original:hit,start:p}:null;
                if(!hit){
                    const edge=s.links.find(([a,b])=>segmentDistance([e.clientX,e.clientY],view.screenPoint(s.rooms[a].center),view.screenPoint(s.rooms[b].center))<=6);
                    if(edge)selectedLink=pairKey(s.rooms[edge[0]].id,s.rooms[edge[1]].id);
                }
            }
        }
        if(press){
            Object.assign(press,{id:e.pointerId,screen:[e.clientX,e.clientY],started:false});
            canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);e.preventDefault();
        }
        selectionChanged();refresh();
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
    function deleteSelection(){
        if(!model)return false;
        finish(true);
        const removed=selectedLink?model.disconnect(...selectedLink):selected!==null?model.remove(selected):false;
        if(!removed){onStatus('Select a room or door link first.');return false;}
        selected=null;selectedLink=null;pendingLink=null;emit(true);onStatus('');refresh();return true;
    }
    const key=e=>{
        if(!visible)return;
        if(e.key==='Escape'){finish(true);pendingLink=null;selectedLink=null;onStatus('');refresh();}
        if(e.key==='Delete'||e.key==='Backspace'){e.preventDefault();deleteSelection();}
    };
    canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointermove',move);
    canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',cancel);canvas.addEventListener('keydown',key);
    const unsubscribe=view.onChange(layout);
    refresh();
    return {
        get tool(){return tool;},
        get selected(){return selected;},
        get selectedLink(){return selectedLink?[...selectedLink]:null;},
        setVisible(next){finish(true);visible=next;refresh();},
        setProgram(next){if(model===next)return;press=null;draft=null;selected=null;selectedLink=null;pendingLink=null;model=next;refresh();},
        setTool(next){finish(true);pendingLink=null;tool=next;canvas.style.cursor=next==='draw'||next==='link'?'crosshair':'default';onStatus(next==='link'?'Click two rooms to request a door.':'',false);refresh();},
        selectRoom(id){selected=id;selectedLink=null;pendingLink=null;selectionChanged();canvas.focus({preventScroll:true});refresh();},
        selectLink(a,b){selected=null;selectedLink=pairKey(a,b);pendingLink=null;selectionChanged();canvas.focus({preventScroll:true});refresh();},
        deleteSelection,
        dispose(){unsubscribe();canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',cancel);canvas.removeEventListener('keydown',key);}
    };
}
