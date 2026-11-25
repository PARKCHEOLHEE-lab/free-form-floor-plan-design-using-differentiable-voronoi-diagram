import { test } from 'node:test';
import assert from 'node:assert/strict';
import { BubbleProgram } from '../web/plan/bubbles.js';

test('bubble program enforces whole circles and area shares with stable linked rooms', () => {
    const p = new BubbleProgram([[-2,-2],[2,-2],[2,2],[-2,2]]);
    assert.deepEqual(p.snapshot(), { rooms: [], links: [], ratios: [] });
    const a = p.add([-1, 0], 0.2);
    assert.notEqual(a, null, 'a valid circle creates a room');
    const b = p.add([1, 0], 0.4);
    assert.notEqual(b, a, 'room IDs are distinct');
    const s = p.snapshot();
    assert.deepEqual(s.rooms.map(r => [r.id, r.center, r.radius]), [[a,[-1,0],0.2],[b,[1,0],0.4]]);
    assert.ok(Math.abs(s.ratios[0] - 0.2) < 1e-12 && Math.abs(s.ratios[1] - 0.8) < 1e-12, 'radii 1:2 mean room areas 1:4');
    for (const [c, r] of [[[2.1,0],0.1],[[1.9,0],0.2],[[-1,0],0.1],[[0,0],0],[[0,0],-1],[[NaN,0],0.2],[[0,0],Infinity]]) {
        assert.equal(p.add(c, r), null, 'outside, duplicate or invalid circles are rejected');
        assert.ok(p.error, 'the rejected edit has an explanation');
    }
    assert.deepEqual(p.snapshot(), s, 'rejected edits leave the last valid input intact');
    assert.equal(p.connect(b, a), true);
    assert.deepEqual(p.snapshot().links, [[0,1]], 'stable room IDs are mapped to dense engine indices');
    assert.equal(p.connect(a, b), false, 'a reverse duplicate is still the same connection');
    assert.equal(p.connect(a, a), false);
    assert.equal(p.connect(a, 999), false);
    assert.equal(p.update(b, [1,0.5], 0.2), true);
    assert.deepEqual(p.snapshot().ratios, [0.5,0.5]);
    const moved = p.snapshot();
    assert.equal(p.update(b, [1.9,0], 0.4), false);
    assert.deepEqual(p.snapshot(), moved);
    assert.equal(p.update(999, [0,0], 0.1), false);
    assert.equal(p.disconnect(a,b), true);
    assert.equal(p.disconnect(a,b), false);
    assert.equal(p.connect(a,b), true);
    assert.equal(p.remove(a), true);
    assert.deepEqual(p.snapshot().links, [], 'deleting a room deletes its connections');
    assert.equal(p.remove(a), false);
    const c = p.add([0,1], 0.2);
    assert.notEqual(c, a, 'deleted IDs are not recycled');
    assert.equal(p.connect(b,c), true);
    assert.deepEqual(p.snapshot().links, [[0,1]], 'deletion does not leave sparse solver indices');
    const detached = p.snapshot();
    detached.rooms[0].center[0] = 100;
    assert.equal(p.snapshot().rooms[0].center[0], 1, 'snapshots cannot mutate retained input');
    const l = new BubbleProgram([[0,0],[4,0],[4,1],[1,1],[1,4],[0,4],[0,0]]);
    assert.notEqual(l.add([0.5,2], 0.5), null, 'a circle can touch the boundary');
    assert.equal(l.add([0.8,0.8], 0.3), null, 'a circle inside the bounding box may cross a concave notch');
    assert.notEqual(l.add([0.8,0.8], 0.28), null, 'the nearby circle that clears the notch is valid');
    assert.equal(l.add([2,2], 0.1), null, 'the notch itself is outside');
});

test('bubble program state restores the same rooms, links and next room ID', () => {
    const p = new BubbleProgram([[-2,-2],[2,-2],[2,2],[-2,2]]);
    const a = p.add([-1, 0], 0.2), b = p.add([1, 0], 0.4);
    p.connect(a, b);
    p.remove(p.add([0, 1], 0.3));
    const copy = BubbleProgram.fromState(JSON.parse(JSON.stringify(p.state())));
    assert.deepEqual(copy.snapshot(), p.snapshot(), 'rooms and links come back unchanged');
    assert.deepEqual(copy.boundary, p.boundary);
    assert.equal(copy.add([0, -1], 0.2), p.add([0, -1], 0.2), 'a removed room ID is not reused after a restore');
    copy.update(a, [-1, 0.5], 0.2);
    assert.deepEqual(p.snapshot().rooms[0].center, [-1, 0], 'the restored program does not share rooms with the original');
});

test('a room name survives geometry edits and a state restore by room ID', () => {
    const program = new BubbleProgram([[-2,-2],[2,-2],[2,2],[-2,2]]);
    const a = program.add([-1, 0], 0.2), b = program.add([1, 0], 0.4);
    const ratios = program.snapshot().ratios;
    assert.equal(typeof program.rename, 'function', 'room names belong to the room model');
    assert.equal(program.rename(b, ' Study '), true);
    assert.deepEqual(program.snapshot().ratios, ratios);
    assert.equal(program.rename(999, 'Missing'), false);
    program.update(b, [1, 0.25], 0.3);
    program.remove(a);
    const restored = BubbleProgram.fromState(JSON.parse(JSON.stringify(program.state())));
    assert.equal(restored.snapshot().rooms[0].id, b);
    assert.equal(restored.snapshot().rooms[0].name, 'Study');
    assert.equal(restored.rename(b, '  '), true);
    assert.equal(restored.snapshot().rooms[0].name, undefined);
});
