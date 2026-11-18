import { test } from 'node:test';
import assert from 'node:assert/strict';
import { measureRawSteps, summarizeRawSteps } from './raw-wall-steps.mjs';

test('visible Raw step is measured through two connector edges and oblique arms', () => {
    const boundary = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    // The interior wall has two long runs with different directions. The
    // sideways shift turns twice between them, so a three-edge test misses it.
    const room = [[-1, -1], [-.4, -1], [-.4, -.2], [-.3, -.19],
        [-.2, -.14], [.116, 1], [-1, 1]];
    const steps = measureRawSteps([room], boundary);
    assert.equal(steps.length, 1, 'measure this visible step exactly once');
    assert.ok(steps[0].height > .18 && steps[0].height < .21);
    assert.ok(Math.abs(measureRawSteps([[...room].reverse()], boundary)[0].height - steps[0].height) < 1e-12);

    const straight = [[-1, -1], [-.4, -1], [.1, 1], [-1, 1]];
    const corner = [[-1, -1], [-.4, -1], [-.4, -.2], [-1, -.2]];
    const u = [[-.4, -1], [-.4, -.2], [.4, -.2], [.4, -1]];
    assert.deepEqual(measureRawSteps([straight, corner, u, boundary], boundary), []);
});

test('a shared step counts once and a change of neighbors marks a junction step', () => {
    const boundary = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    const [a, b, c, d, e] = [[-.4, -1], [-.4, -.2], [-.3, -.19], [-.2, -.14], [.116, 1]];
    const left = [[-1, -1], a, b, c, d, e, [-1, 1]];
    const right = [a, [1, -1], [1, 1], e, d, c, b];
    const frame = rings => ({ walls: rings, cells: rings.map((ring, room) => ({ ring, room })) });
    const onePair = summarizeRawSteps(frame([left, right]), boundary, 2);
    assert.equal(onePair.all.count, 1);
    assert.equal(onePair.pair.count, 1);

    const junction = [1, -.19];
    const lower = [a, [1, -1], junction, c, b];
    const upper = [c, junction, [1, 1], e, d];
    const withJunction = summarizeRawSteps(frame([left, lower, upper]), boundary, 3);
    const leftStep = withJunction.steps.find(step => step.wall === 0);
    assert.equal(leftStep.kind, 'junction');
    assert.deepEqual(leftStep.pairs.sort(), ['0,1', '0,2']);
    assert.ok(Math.abs(leftStep.height - onePair.all.max) < 1e-12,'ownership does not change the measured height');
});

test('a long returning wall run does not join two bends into one step', () => {
    const boundary = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
    // The westward runs enclose a vertical offset of .75, but their
    // path returns east for a full wall run. The actual step joins the
    // two northward runs on either side of the .30 horizontal connector.
    const room = [[-1, -1], [1, -1], [1, -.5], [-.3, -.5], [-.3, -.15],
        [0, -.15], [0, .25], [-1, .25]];
    const steps = measureRawSteps([room],boundary);
    assert.deepEqual(steps.map(step => step.height), [.3]);
});
