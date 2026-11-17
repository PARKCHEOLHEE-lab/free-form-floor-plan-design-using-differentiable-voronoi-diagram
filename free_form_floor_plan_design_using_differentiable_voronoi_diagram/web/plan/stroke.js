// Wall bands: a closed outline stroked like a canvas line with miter joins (lineJoin 'miter').
// The plan view stacks two bands per outline, a dark one at the full wall width and a white one
// a little narrower, which leaves the two thin wall lines the 2D canvas drew; the 3D view lifts
// the same band into a wall.

const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1]];
const mul = (a, k) => [a[0] * k, a[1] * k];
const unit = (a) => {
    const l = Math.hypot(a[0], a[1]);
    return [a[0] / l, a[1] / l];
};

/**
 * Stroke a closed ring (open list of points) at half width `d`. Returns the band as a closed
 * list of [left, right] point pairs, one pair per join (two for a bevelled join); left is the
 * side to the left of the ring's direction. A join keeps its miter while the miter length stays
 * within `miterLimit` half widths, as canvas does, and is bevelled past that.
 */
export function strokeRing(ring, d, { miterLimit = 2 } = {}) {
    const pts = ring.filter((p, i) => {
        const q = ring[(i + 1) % ring.length];
        return p[0] !== q[0] || p[1] !== q[1];
    });
    const n = pts.length;
    const pairs = [];
    for (let i = 0; i < n; i++) {
        const v = pts[i];
        const u1 = unit(sub(v, pts[(i - 1 + n) % n]));
        const u2 = unit(sub(pts[(i + 1) % n], v));
        const n1 = [-u1[1], u1[0]];
        const n2 = [-u2[1], u2[0]];
        const bisector = add(n1, n2);
        const len = Math.hypot(bisector[0], bisector[1]);
        // cos of the half angle between the two edge normals; the miter reaches d / cos
        const cos = len / 2;
        if (len > 1e-12 && cos >= 1 / miterLimit) {
            const m = mul(bisector, d / cos / len);
            pairs.push([add(v, m), sub(v, m)]);
        } else {
            pairs.push([add(v, mul(n1, d)), sub(v, mul(n1, d))]);
            pairs.push([add(v, mul(n2, d)), sub(v, mul(n2, d))]);
        }
    }
    return pairs;
}

/** The band as triangles: two per step between neighbouring pairs, closing back to the first. */
export function strokeTriangles(pairs) {
    const tris = [];
    for (let i = 0; i < pairs.length; i++) {
        const [l0, r0] = pairs[i];
        const [l1, r1] = pairs[(i + 1) % pairs.length];
        tris.push([l0, r0, l1], [r0, r1, l1]);
    }
    return tris;
}
