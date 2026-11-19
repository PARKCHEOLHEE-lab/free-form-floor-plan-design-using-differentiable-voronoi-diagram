import assert from 'node:assert/strict';
import { test } from 'node:test';
import { snapBoundaryPoint } from '../web/plan/boundary-editor.js';
import { isSimpleBoundary } from '../web/plan/geom.js';

test('boundary snaps use screen distance and world direction with Shift precedence', () => {
    const project = ([x,y]) => [100 * x, 100 * y];
    const vertices = [[0,0],[1,1],[2,0]];
    const reference = [[4,0],[6,0],[6,2],[4,2]];
    function snap(point, shift = false, projection = project) {
        return snapBoundaryPoint({ point, screen:projection(point), project:projection, vertices, reference, shift });
    }

    assert.deepEqual(snap([.03,.01]), {point:[0,0],kind:'close'});
    assert.deepEqual(snap([1.03,1.01]), {point:[1,1],kind:'vertex'});
    assert.deepEqual(snap([.48,.51]), {point:[.5,.5],kind:'midpoint'});
    assert.deepEqual(snap([4.04,.03]), {point:[4,0],kind:'vertex'});
    assert.deepEqual(snap([5.04,.03],true), {point:[5,0],kind:'midpoint'});
    assert.deepEqual(snap([4.4,2.03],true), {point:[4.4,0],kind:'axis'});
    const angle = snap([3.25,1.22]);
    assert.equal(angle.kind, 'parallel');
    assert.ok(Math.abs(angle.point[0] - 2 - angle.point[1]) < 1e-12);
    assert.deepEqual(snap([3.25,1.22],true), {point:[3.25,0],kind:'axis'});

    const angled = ([x,y]) => [450 + 88 * x - 52 * y, 230 - 42 * x - 60 * y];
    assert.deepEqual(snap([2.15,1.45],true,angled), {point:[2,1.45],kind:'axis'});
    assert.equal(snap([4.08,.12]).kind, 'free', 'a vertex outside the screen radius is not selected');
});

test('draft endpoints and midpoints win over closer committed boundary points', () => {
    const project = ([x, y]) => [100 * x, 100 * y];
    const vertices = [[0, 0], [1, 1], [2, 0]];
    const reference = [[0.55, 0.51], [1.05, 1.01], [4, 4]];
    const snap = point => snapBoundaryPoint({ point, screen: project(point), project, vertices, reference, shift: false });

    assert.deepEqual(snap([1.055, 1.01]), { point: [1, 1], kind: 'vertex' });
    assert.deepEqual(snap([0.555, 0.515]), { point: [0.5, 0.5], kind: 'midpoint' });
    assert.deepEqual(snap([4.03, 4.01]), { point: [4, 4], kind: 'vertex' });
});

test('near-normal cursor snaps exactly perpendicular to an oblique draft edge', () => {
    const project = ([x, y]) => [100 * x - 40 * y, -20 * x + 75 * y];
    const vertices = [[1, 1], [2, 2]];
    const reference = [[6, 6], [7.04, 5.03], [8, 8]];
    const snap = (point, shift = false, points = vertices, outline = reference) =>
        snapBoundaryPoint({ point, screen: project(point), project, vertices: points, reference: outline, shift });

    const normal = snap([3.04, 1.03]);
    assert.equal(normal.kind, 'perpendicular', 'a relevant draft direction wins over a closer old-outline direction');
    assert.ok(Math.abs(normal.point[0] - 3.005) < 1e-12);
    assert.ok(Math.abs(normal.point[1] - 0.995) < 1e-12);
    assert.equal(snap([2.2, 1.838], false, vertices, []).kind, 'free', 'the 5-degree threshold applies to perpendicular snaps');
    assert.deepEqual(snap([3.04, 1.03], true), { point: [3.04, 2], kind: 'axis' });
    assert.equal(snap([3.04, 1.03], false, [[2, 2]], [[6, 6], [7, 7]]).kind, 'perpendicular');
});

test('a simple boundary may be concave and include straight runs but cannot touch itself', () => {
    const concave = [[2,1],[4,1],[6,1],[6,3],[4,3],[4,5],[2,5]];
    assert.equal(isSimpleBoundary(concave),true);
    assert.equal(isSimpleBoundary(concave.slice().reverse()),true);
    for (const bad of [
        [[0,0],[0,0],[2,0],[1,1]],
        [[0,0],[3,0],[3,3],[1,0],[0,3]],
        [[0,0],[3,0],[3,3],[0,3],[3,0]],
        [[0,0],[3,0],[3,3],[0,3],[0,0]],
    ]) assert.equal(isSimpleBoundary(bad),false);
});
