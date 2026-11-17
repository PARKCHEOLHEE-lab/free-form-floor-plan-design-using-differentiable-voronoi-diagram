// The plan's corner model: one corner per point, rooms as rings of corner ids.
// g = { corners: Map<id, {x, y, fixed}>, rooms: rooms[r] = that room's rings, each a list of corner ids }.
// A room split in pieces has several rings. Alignment (and later corner editing and openings)
// work on this one structure, so two rooms that share a wall share its corners.
import { openRing, segmentDistance, signedArea2 } from "./geom.js";

/**
 * Room outlines from optimizer cells. Cells of one room are welded by cancelling every edge
 * that two of its cells share, so the room keeps only its outline. Neighbouring cells share
 * their edge points exactly, so one point becomes one corner that every room sharing it uses.
 * Kinks between junctions are then dropped (Douglas-Peucker with `simplifyTol`); a junction is
 * a boundary corner or a corner where three or more rooms meet, and boundary corners all stay.
 */
export function fromCells(cells, boundary, { roomCount, simplifyTol }) {
    const B = openRing(boundary);
    const span = Math.max(...B.map((p) => Math.abs(p[0])), ...B.map((p) => Math.abs(p[1])), 1);
    const onBoundaryEps = 1e-9 * span;
    const corners = new Map();
    const ids = new Map();
    const cornerId = (p) => {
        const k = `${Math.round(p[0] * 1e9)},${Math.round(p[1] * 1e9)}`;
        let id = ids.get(k);
        if (id === undefined) {
            id = ids.size;
            ids.set(k, id);
            const fixed = B.some((a, i) => segmentDistance(p, a, B[(i + 1) % B.length]) < onBoundaryEps);
            corners.set(id, { x: p[0], y: p[1], fixed });
        }
        return id;
    };

    // per room: directed outline edges left after cancelling shared ones
    const edges = Array.from({ length: roomCount }, () => new Map());
    for (const c of cells) {
        if (c.room < 0 || c.room >= roomCount) continue;
        let ring = openRing(c.ring);
        if (ring.length < 3) continue;
        if (signedArea2(ring) < 0) ring = [...ring].reverse();
        const vs = ring.map(cornerId).filter((id, i, a) => id !== a[(i + 1) % a.length]);
        for (let i = 0; i < vs.length; i++) {
            const a = vs[i];
            const b = vs[(i + 1) % vs.length];
            const back = `${b},${a}`;
            if (edges[c.room].has(back)) edges[c.room].delete(back);
            else edges[c.room].set(`${a},${b}`, [a, b]);
        }
    }
    const rooms = edges.map(chainRings);
    simplifyBetweenJunctions(corners, rooms, simplifyTol);
    // keep only corners some outline still uses (cell points inside a room and dropped kinks go)
    const used = new Set(rooms.flat(2));
    for (const id of [...corners.keys()]) if (!used.has(id)) corners.delete(id);
    return { corners, rooms };
}

/** Follow directed edges into closed rings. */
function chainRings(edges) {
    const out = new Map();
    for (const [a, b] of edges.values()) out.set(a, [...(out.get(a) ?? []), b]);
    const rings = [];
    for (const [start] of out) {
        while ((out.get(start)?.length ?? 0) > 0) {
            const ring = [start];
            let cur = out.get(start).pop();
            while (cur !== start) {
                ring.push(cur);
                const nexts = out.get(cur);
                if (!nexts || nexts.length === 0) break;
                cur = nexts.pop();
            }
            if (ring.length >= 3) rings.push(ring);
        }
    }
    return rings;
}

function simplifyBetweenJunctions(corners, rooms, tol) {
    const usedBy = new Map();
    rooms.forEach((rings, r) => rings.forEach((ring) => ring.forEach((id) => usedBy.set(id, (usedBy.get(id) ?? new Set()).add(r)))));
    const junction = (id) => corners.get(id).fixed || (usedBy.get(id)?.size ?? 0) >= 3;
    const removed = new Set();
    const pt = (id) => [corners.get(id).x, corners.get(id).y];

    for (const rings of rooms) {
        for (const ring of rings) {
            const starts = ring.map((id, i) => (junction(id) ? i : -1)).filter((i) => i >= 0);
            if (starts.length === 0) continue; // a loop with no junction keeps its shape
            for (let s = 0; s < starts.length; s++) {
                const i0 = starts[s];
                const i1 = starts[(s + 1) % starts.length];
                const chain = [ring[i0]];
                for (let i = (i0 + 1) % ring.length; ; i = (i + 1) % ring.length) {
                    chain.push(ring[i]);
                    if (i === i1) break;
                }
                // the neighbouring room walks the same chain backwards; orient it one way so both agree
                const canon = chain[0] <= chain[chain.length - 1] ? chain : [...chain].reverse();
                const keep = douglasPeucker(canon.map(pt), tol);
                canon.forEach((id, k) => {
                    if (!keep[k]) removed.add(id);
                });
            }
        }
    }
    for (const rings of rooms) for (let k = 0; k < rings.length; k++) rings[k] = rings[k].filter((id) => !removed.has(id));
}

/** Which points of an open polyline Douglas-Peucker keeps; both ends always stay. */
function douglasPeucker(pts, tol) {
    const keep = pts.map((_, i) => i === 0 || i === pts.length - 1);
    const visit = (a, b) => {
        let far = -1;
        let best = tol;
        for (let i = a + 1; i < b; i++) {
            const d = segmentDistance(pts[i], pts[a], pts[b]);
            if (d > best) {
                best = d;
                far = i;
            }
        }
        if (far < 0) return;
        keep[far] = true;
        visit(a, far);
        visit(far, b);
    };
    visit(0, pts.length - 1);
    return keep;
}

/** A room's rings as point lists. */
export function roomRings(g, room) {
    return (g.rooms[room] ?? []).map((ring) => ring.map((id) => [g.corners.get(id).x, g.corners.get(id).y]));
}
