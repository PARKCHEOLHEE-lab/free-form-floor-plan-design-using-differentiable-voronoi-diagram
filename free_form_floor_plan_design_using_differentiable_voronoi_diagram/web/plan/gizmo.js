// The view gizmo: the plan's X, Y and Z axes as seen by the camera, drawn in a small corner widget.

// Each axis end as a direction in the three.js world, where plan point (x, y) at height h is
// (x, h, -y). -Z is left out: a view from below the floor shows nothing.
const ENDS = {
    '+x': [1, 0, 0],
    '-x': [-1, 0, 0],
    '+y': [0, 0, -1],
    '-y': [0, 0, 1],
    '+z': [0, 1, 0],
};

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/**
 * Where each axis end sits in the gizmo, for a camera whose world directions are `right`, `up`
 * and `back` (back points at the viewer). Returns { axis, x, y, depth } per end, with x to the
 * right and y down in pixels from the centre (at most `radius` away) and depth growing toward
 * the viewer, ordered far to near so the nearest end is drawn last.
 */
export function axisTips(right, up, back, radius) {
    return Object.entries(ENDS)
        .map(([axis, v]) => ({ axis, x: dot(v, right) * radius, y: -dot(v, up) * radius, depth: dot(v, back) }))
        .sort((a, b) => a.depth - b.depth);
}
