// Wall alignment, ported from voronoi/postprocess.py (after GSDiff's test-time post-process).
import { openRing, segmentDistance } from "./geom.js";
import { wallEdges } from "./graph.js";

/** Orientation (mod 90 degrees) of the boundary segment nearest to `mid`, as the wall loss picks it. */
function frameAngle(boundary, mid) {
    const B = openRing(boundary);
    let best = Infinity;
    let angle = 0;
    B.forEach((a, i) => {
        const b = B[(i + 1) % B.length];
        const d = segmentDistance(mid, a, b);
        if (d < best) {
            best = d;
            angle = Math.atan2(b[1] - a[1], b[0] - a[0]);
        }
    });
    const q = Math.PI / 2;
    return ((angle % q) + q) % q;
}

/** Share of wall length that runs exactly parallel or perpendicular to its nearest boundary segment. */
export function alignedShare(g, boundary) {
    let ok = 0;
    let total = 0;
    for (const {a, b} of wallEdges(g)) {
        const A = g.corners.get(a);
        const Bc = g.corners.get(b);
        const phi = frameAngle(boundary, [(A.x + Bc.x) / 2, (A.y + Bc.y) / 2]);
        const dx = Bc.x - A.x;
        const dy = Bc.y - A.y;
        const du = dx * Math.cos(phi) + dy * Math.sin(phi);
        const dv = -dx * Math.sin(phi) + dy * Math.cos(phi);
        const len = Math.hypot(du, dv);
        total += len;
        if (Math.min(Math.abs(du), Math.abs(dv)) < 1e-9) ok += len;
    }
    return total > 0 ? ok / total : 1;
}

/**
 * Clean the walls in place, in two passes with one threshold `tol` (GSDiff uses 1% of its canvas):
 * 1. merge: corners closer than `tol` (Chebyshev) become one corner; a group holding two
 *    boundary corners is left alone, so the boundary never moves;
 * 2. align: in each edge's local frame (the nearest boundary segment), chains of edges whose two
 *    ends differ by at most `tol` along one axis are snapped to one shared coordinate, so walls
 *    beside a diagonal boundary straighten along the diagonal instead of along x and y.
 */
export function alignPlan(g, boundary, { tol }) {
    mergeCorners(g, tol);
    snapChains(g, boundary, tol);
}

function mergeCorners(g, tol) {
    const ids = [...g.corners.keys()];
    const parent = new Map(ids.map((id) => [id, id]));
    const find = (x) => {
        while (parent.get(x) !== x) x = parent.get(x);
        return x;
    };
    for (let i = 0; i < ids.length; i++) {
        const p = g.corners.get(ids[i]);
        for (let j = i + 1; j < ids.length; j++) {
            const q = g.corners.get(ids[j]);
            if (Math.max(Math.abs(p.x - q.x), Math.abs(p.y - q.y)) < tol) parent.set(find(ids[i]), find(ids[j]));
        }
    }
    const groups = new Map();
    for (const id of ids) groups.set(find(id), [...(groups.get(find(id)) ?? []), id]);

    const replace = new Map();
    for (const group of groups.values()) {
        const fixed = group.filter((id) => g.corners.get(id).fixed);
        if (group.length < 2 || fixed.length > 1) continue;
        const keep = fixed[0] ?? Math.min(...group);
        if (fixed.length === 0) {
            const k = g.corners.get(keep);
            k.x = group.reduce((s, id) => s + g.corners.get(id).x, 0) / group.length;
            k.y = group.reduce((s, id) => s + g.corners.get(id).y, 0) / group.length;
        }
        for (const id of group) if (id !== keep) replace.set(id, keep);
    }
    for (const id of replace.keys()) g.corners.delete(id);
    g.rooms = g.rooms.map((rings) => rings.map((ring) => cleanRing(ring.map((id) => replace.get(id) ?? id))).filter((ring) => ring.length >= 3));
}

/** Drop repeated neighbours and back-and-forth spikes (a, b, a) that merging can leave. */
function cleanRing(ring) {
    let r = ring.filter((id, i) => id !== ring[(i + 1) % ring.length]);
    for (let changed = true; changed && r.length >= 3; ) {
        changed = false;
        for (let i = 0; i < r.length; i++) {
            if (r[(i - 1 + r.length) % r.length] === r[(i + 1) % r.length]) {
                r = r.filter((_, k) => k !== i && k !== (i + 1) % r.length);
                changed = true;
                break;
            }
        }
    }
    return r;
}

function snapChains(g, boundary, tol) {
    const byFrame = new Map();
    for (const {a, b} of wallEdges(g)) {
        const A = g.corners.get(a);
        const B = g.corners.get(b);
        const phi = Math.round(frameAngle(boundary, [(A.x + B.x) / 2, (A.y + B.y) / 2]) * 1e6) / 1e6;
        byFrame.set(phi, [...(byFrame.get(phi) ?? []), [a, b]]);
    }
    // A corner on walls of two frames ends where the last frame snaps it, so the frame with the
    // most walls goes last. postprocess.py takes frames in graph order; on shape D that leaves
    // 0.63 of the wall aligned instead of 0.69.
    const frames = [...byFrame].sort((x, y) => x[1].length - y[1].length);
    for (const [phi, edges] of frames) {
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        for (const axis of [0, 1]) {
            // 0 snaps u (walls running along v), 1 snaps v (walls running along u)
            const uv = (id) => {
                const p = g.corners.get(id);
                return [p.x * c + p.y * s, -p.x * s + p.y * c];
            };
            const chain = edges.filter(([a, b]) => Math.abs(uv(a)[axis] - uv(b)[axis]) <= tol);
            for (const comp of components(chain)) {
                const nodes = [...new Set(comp.flat())];
                // boundary corners in the chain, keyed by their rounded coordinate, valued by the exact one
                const fixed = new Map(nodes.filter((id) => g.corners.get(id).fixed).map((id) => [Math.round(uv(id)[axis] * 1e9), uv(id)[axis]]));
                if (fixed.size > 1) continue; // two boundary corners disagree; moving either would move the boundary
                // GSDiff's x_bar: the mean over every edge's two ends
                const target = fixed.size === 1 ? [...fixed.values()][0] : comp.reduce((t, [a, b]) => t + uv(a)[axis] + uv(b)[axis], 0) / (2 * comp.length);
                for (const id of nodes) {
                    const p = uv(id);
                    p[axis] = target;
                    const q = g.corners.get(id);
                    q.x = p[0] * c - p[1] * s;
                    q.y = p[0] * s + p[1] * c;
                }
            }
        }
    }
}

/** Connected groups of edges (edges that share a corner). */
function components(edges) {
    const parent = new Map();
    const find = (x) => {
        if (!parent.has(x)) parent.set(x, x);
        while (parent.get(x) !== x) x = parent.get(x);
        return x;
    };
    for (const [a, b] of edges) parent.set(find(a), find(b));
    const out = new Map();
    for (const e of edges) out.set(find(e[0]), [...(out.get(find(e[0])) ?? []), e]);
    return [...out.values()];
}
