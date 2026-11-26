import { inside } from './geom.js';
import { roomRings, wallEdges } from './graph.js';
import { roomLabel } from './bubbles.js';

/** Check the actual output graph; a requested link needs one straight shared wall. */
export function evaluatePlan(g, program, width, {finished=false,running=false}={}) {
    if (!Number.isFinite(width) || width<=0) throw new RangeError('Door width must be positive and finite.');
    const issues=[], doors=[];
    const rings=g.rooms.map((_,r)=>roomRings(g,r));
    const contains=(r,p)=>(rings[r]??[]).reduce((hit,ring)=>hit!==inside(p,ring),false);
    // The panel's label: the editor's stable id, which deleting a room does not renumber.
    const name=r=>roomLabel(program.rooms[r]?.id ?? r+1);
    if (!program.rooms.length) issues.push({kind:'input',message:'Draw at least one room.'});
    if (g.rooms.length!==program.rooms.length) issues.push({kind:'rooms',message:'The generated room count does not match the bubbles.'});
    program.rooms.forEach((room,r)=>{
        if (!contains(r,room.center)) issues.push({kind:'center',room:r,message:`${name(r)} does not contain its bubble center.`});
        const parts=[];
        // A walk through a corner shared by two lobes may contain several loops.
        for(const ring of g.rooms[r]??[]){
            const path=[];
            for(const id of [...ring,ring[0]]){
                const start=path.indexOf(id);
                if(start>=0){
                    const ids=path.splice(start);
                    if(ids.length>=3)parts.push({ids,points:ids.map(id=>{const p=g.corners.get(id);return[p.x,p.y];})});
                }
                path.push(id);
            }
        }
        // Even nesting depth denotes an outer region; odd depth denotes a hole.
        // Use containment rather than winding so reversed rings keep their meaning.
        const components=parts.filter((ring,i)=>parts.filter((outer,j)=>i!==j&&ring.ids.every((id,k)=>outer.ids.includes(id)||inside(ring.points[k],outer.points))).length%2===0).length;
        if(components>1)issues.push({kind:'disconnected',room:r,message:`${name(r)} is split into ${components} separate regions.`});
    });
    const edges=wallEdges(g);
    for (const pair of program.links) {
        const [r,s]=pair;
        const candidates=edges.filter(e=>e.rooms.length===2 && e.rooms.includes(r) && e.rooms.includes(s)).map(e=>{
            const a=g.corners.get(e.a), b=g.corners.get(e.b);
            return {a,b,length:Math.hypot(b.x-a.x,b.y-a.y)};
        }).sort((a,b)=>b.length-a.length);
        const edge=candidates.find(e=>e.length>=width);
        if (!edge) {
            issues.push({kind:'connection',rooms:[...pair],message:`${name(r)} and ${name(s)} need a shared wall wide enough for a door.`});
            continue;
        }
        const dx=(edge.b.x-edge.a.x)/edge.length, dy=(edge.b.y-edge.a.y)/edge.length;
        const center=[(edge.a.x+edge.b.x)/2,(edge.a.y+edge.b.y)/2];
        let normal=[-dy,dx];
        if (!contains(r,[center[0]+normal[0]*width/4,center[1]+normal[1]*width/4])) normal=normal.map(v=>-v);
        doors.push({rooms:[...pair],a:[center[0]-dx*width/2,center[1]-dy*width/2],b:[center[0]+dx*width/2,center[1]+dy*width/2],normal});
    }
    const valid=issues.length===0;
    return {valid,doors,issues,state:finished?(valid?'complete':'unmet'):running?'running':'ready'};
}
