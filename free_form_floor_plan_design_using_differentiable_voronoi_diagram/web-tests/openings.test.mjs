import { test } from 'node:test';
import assert from 'node:assert/strict';
import { wallEdges,fromCells } from '../web/plan/graph.js';
import { evaluatePlan } from '../web/plan/openings.js';

test('disconnected regions are invalid while holes remain part of one room',()=>{
    const square=(x,y,w)=>[[x,y],[x+w,y],[x+w,y+w],[x,y+w]];
    const graph=rings=>{
        const corners=new Map();
        const ids=rings.map(ring=>ring.map(([x,y])=>{const id=corners.size;corners.set(id,{x,y});return id;}));
        return {corners,rooms:[ids]};
    };
    const program={rooms:[{center:[0.5,0.5]}],links:[]};
    for(const rings of [[square(0,0,1),square(2,0,1)],[square(0,0,1),square(1,1,1)]]){
        const result=evaluatePlan(graph(rings),program,0.1,{finished:true});
        assert.equal(result.valid,false,'separate or point-touching pieces cannot form one room');
        assert.equal(result.state,'unmet');
        assert.ok(result.issues.some(i=>i.kind==='disconnected'&&i.room===0));
    }
    const touching=fromCells([square(0,0,1),square(1,1,1)].map(ring=>({room:0,ring})),square(-1,-1,4),{roomCount:1,simplifyTol:0});
    assert.equal(evaluatePlan(touching,program,0.1).valid,false,'the actual corner graph can encode point contact inside a single repeated-corner ring');
    const hole=[square(0,0,6),square(1,1,4)];
    assert.equal(evaluatePlan(graph(hole),program,0.1,{finished:true}).state,'complete','an interior hole is not another room component');
    assert.equal(evaluatePlan(graph(hole.map(r=>[...r].reverse())),program,0.1).valid,true,'component checking is independent of ring winding');
    const island=[...hole,square(2,2,2)];
    assert.equal(evaluatePlan(graph(island),program,0.1).valid,false,'an island inside a hole is a second component');
});

test('shared corner walls make only requested doors and expose center or wall violations', () => {
    const points = [[0,0],[1,0],[2,0],[2,1],[1,1],[0,1],[3,0],[3,1]];
    const g = { corners: new Map(points.map(([x,y],i) => [i,{x,y,fixed:true}])), rooms: [[[0,1,4,5]],[[1,2,3,4]],[[2,6,7,3]]] };
    const shared = wallEdges(g).filter(e => e.rooms.length===2);
    assert.deepEqual(shared.map(e => [e.a,e.b,e.rooms]), [[1,4,[0,1]],[2,3,[1,2]]], 'shared walls retain their actual owning room pairs');
    assert.equal(wallEdges(g).length,10,'every physical edge appears once');
    const program = { rooms: points.slice(0,3).map((_,r) => ({id:10+r,center:[r+0.5,0.5],radius:0.1})), links:[[0,1]] };
    assert.equal(evaluatePlan(g,{rooms:[],links:[]},0.2).valid,false,'empty input cannot produce a successful plan');
    assert.throws(()=>evaluatePlan(g,program,0), /positive/, 'door width has an explicit contract');
    const result = evaluatePlan(g,program,0.2);
    assert.equal(result.valid,true);
    assert.deepEqual(result.issues,[]);
    assert.equal(result.doors.length,1,'unlinked adjacent rooms get no door');
    assert.deepEqual(result.doors[0].rooms,[0,1]);
    assert.deepEqual(result.doors[0].a,[1,0.4]);
    assert.deepEqual(result.doors[0].b,[1,0.6]);
    assert.deepEqual(result.doors[0].normal,[-1,0],'the door opens toward its first room');
    assert.equal(evaluatePlan(g,{...program,links:[]},0.2).doors.length,0);
    const separated=evaluatePlan(g,{...program,links:[[0,2]]},0.2);
    assert.equal(separated.valid,false);
    assert.deepEqual(separated.doors,[]);
    assert.ok(separated.issues.some(i => i.kind==='connection' && i.rooms[0]===0 && i.rooms[1]===2));
    const short=evaluatePlan(g,program,1.1);
    assert.equal(short.valid,false,'a wall shorter than the door is not a connection');
    assert.equal(short.doors.length,0);
    assert.equal(evaluatePlan(g,program,1.1,{finished:true}).state,'unmet','an iteration limit does not make an invalid plan complete');
    assert.equal(evaluatePlan(g,program,0.2,{finished:true}).state,'complete');
    assert.equal(evaluatePlan(g,program,0.2,{running:true}).state,'running');
    assert.equal(evaluatePlan(g,program,0.2).state,'ready');
    assert.equal(evaluatePlan(g,program,1).valid,true,'a door can use the full available shared wall');
    const badCenter={...program,rooms:program.rooms.map((r,i) => i===0?{...r,center:[1.5,0.5]}:r)};
    assert.ok(evaluatePlan(g,badCenter,0.2).issues.some(i => i.kind==='center' && i.room===0));
    const reversed={...g,rooms:g.rooms.map(rings => rings.map(ring => [...ring].reverse()))};
    assert.deepEqual(evaluatePlan(reversed,program,0.2).doors,result.doors,'ring winding does not change the door pose');
    const holePoints=[[0,0],[4,0],[4,4],[0,4],[1,1],[3,1],[3,3],[1,3]];
    const holes={corners:new Map(holePoints.map(([x,y],i)=>[i,{x,y}])),rooms:[[[0,1,2,3],[4,5,6,7]],[[4,5,6,7]]]};
    const inHole={rooms:[{center:[2,1.5]},{center:[2,2]}],links:[]};
    assert.ok(evaluatePlan(holes,inHole,0.2).issues.some(i=>i.kind==='center' && i.room===0),'a hole is outside the surrounding room');
    const missing={...g,rooms:g.rooms.slice(0,2)};
    assert.ok(evaluatePlan(missing,program,0.2).issues.some(i=>i.kind==='center' && i.room===2),'a missing room cannot satisfy its center');
    assert.deepEqual(g.rooms,[[[0,1,4,5]],[[1,2,3,4]],[[2,6,7,3]]],'checking does not edit the generated plan');
});
