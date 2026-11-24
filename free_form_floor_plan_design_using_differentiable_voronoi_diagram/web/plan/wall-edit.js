// Moving one interior wall of the aligned plan by hand (the Adjustment panel). The plan is the
// corner graph of graph.js; a move returns a new graph and leaves the given one untouched.
import { components } from "./align.js";
import { roomRings, wallEdges } from "./graph.js";
import { isSimpleBoundary, signedArea2 } from "./geom.js";

const TOL = 1e-9;
const KINK = Math.sin(10 * Math.PI / 180); // walls within 10 degrees count as one straight line (alignment leaves small kinks)
const MIN_WALL = 0.01;  // the shortest wall a move may leave (plan units)

const point = (g, id) => [g.corners.get(id).x, g.corners.get(id).y];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
const unit = (v) => { const l = Math.hypot(v[0], v[1]); return [v[0] / l, v[1] / l]; };

/**
 * The straight wall through the interior edge a-b: every interior edge within 10 degrees of its
 * direction that connects to it, so a wall passing a T junction moves as one piece. Each corner of
 * the run slides along the other wall it meets, so neighbouring walls keep their direction; where
 * walls meet at different angles (a Y junction) it slides along the one most across the run and
 * the others turn about their far ends. A boundary corner slides along the boundary, so a run that
 * ends where the boundary bends, or where the boundary runs on in the wall's direction, is
 * `locked`. Returns null when a-b is not an interior wall.
 * The normal points right (+x), or up for a horizontal wall; a positive offset moves that way.
 */
export function wallRun(g, a, b) {
    const all = wallEdges(g);
    const interior = all.filter((e) => e.rooms.length === 2);
    const pick = interior.find((e) => (e.a === a && e.b === b) || (e.a === b && e.b === a));
    if (!pick) return null;
    const origin = point(g, pick.a);
    const u = unit(sub(point(g, pick.b), origin));
    let normal = [-u[1], u[0]];
    if (normal[0] < -TOL || (Math.abs(normal[0]) <= TOL && normal[1] < 0)) normal = [-normal[0], -normal[1]];
    const parallel = (e) => Math.abs(cross(unit(sub(point(g, e.b), point(g, e.a))), u)) < KINK;
    const line = interior.filter(parallel).map((e) => [e.a, e.b]);
    const edges = components(line).find((group) => group.some(([x, y]) => x === pick.a && y === pick.b));
    const inRun = new Set(edges.map(([x, y]) => `${x},${y}`));
    const corners = [...new Set(edges.flat())];

    const slides = new Map(); // corner -> its movement per unit of offset
    let lo = -Infinity, hi = Infinity, locked = false;
    for (const id of corners) {
        const p = point(g, id);
        const meeting = all.filter((e) => (e.a === id || e.b === id) && !inRun.has(`${e.a},${e.b}`));
        if (!meeting.length) { slides.set(id, normal); continue; } // a point inside the straight wall moves with it
        const fixed = g.corners.get(id).fixed;
        // a boundary corner slides along the boundary; an inner one along the wall most across the run
        const guides = fixed ? meeting.filter((e) => e.rooms.length === 1) : meeting;
        const ends = guides.map((e) => point(g, e.a === id ? e.b : e.a));
        const dirs = ends.map((q) => unit(sub(q, p)));
        const lead = fixed ? dirs[0] : dirs.reduce((best, d) => (Math.abs(dot(d, normal)) > Math.abs(dot(best, normal)) ? d : best));
        const onGuide = dirs.map((d) => Math.abs(cross(lead, d)) <= KINK); // the guide wall, on both sides of the corner
        if (fixed && !onGuide.every(Boolean)) { locked = true; break; } // the boundary bends here
        const others = ends.filter((_, i) => onGuide[i]);
        const w = unit(dirs.filter((_, i) => onGuide[i]).reduce((s, d) => (dot(d, lead) >= 0 ? [s[0] + d[0], s[1] + d[1]] : [s[0] - d[0], s[1] - d[1]]), [0, 0]));
        const along = dot(w, normal);
        if (Math.abs(along) < KINK) { locked = true; break; }
        const rate = 1 / along;
        slides.set(id, [w[0] * rate, w[1] * rate]);
        // stop before any wall this corner slides along gets shorter than MIN_WALL
        for (const q of others) {
            const k = dot(sub(q, p), w);
            const limit = (k > 0 ? k - MIN_WALL : k + MIN_WALL) / rate;
            if ((k > 0) === (rate > 0)) hi = Math.min(hi, limit); else lo = Math.max(lo, limit);
        }
    }
    const rooms = g.rooms.map((rings, r) => (rings.some((ring) => ring.some((id) => slides.has(id))) ? r : -1)).filter((r) => r >= 0);
    return { edges, corners, rooms, origin, normal, slides, range: [lo, hi], locked };
}

/**
 * Move a run by `offset` along its normal, clamped to its range and then to the largest offset
 * that keeps every room beside it a simple ring facing the same way. Returns { graph, offset }, or
 * null for a locked run.
 */
export function moveRun(g, run, offset) {
    if (!run || run.locked) return null;
    const moved = (t) => {
        const corners = new Map([...g.corners].map(([id, c]) => [id, { ...c }]));
        for (const [id, [dx, dy]] of run.slides) {
            const c = corners.get(id);
            c.x += dx * t;
            c.y += dy * t;
        }
        return { corners, rooms: g.rooms.map((rings) => rings.map((ring) => [...ring])) };
    };
    const sides = run.rooms.map((r) => roomRings(g, r).map((ring) => Math.sign(signedArea2(ring))));
    const valid = (h) => run.rooms.every((r, k) => roomRings(h, r).every((ring, i) => isSimpleBoundary(ring) && Math.sign(signedArea2(ring)) === sides[k][i]));
    let t = Math.min(Math.max(offset, run.range[0]), run.range[1]);
    if (!valid(moved(t))) {
        let ok = 0;
        for (let i = 0; i < 40; i++) {
            const mid = (ok + t) / 2;
            if (valid(moved(mid))) ok = mid; else t = mid;
        }
        t = ok;
    }
    return { graph: moved(t), offset: t };
}

/** Each room's share of the plan area (holes subtract). */
export function roomShares(g) {
    const areas = g.rooms.map((_, r) => roomRings(g, r).reduce((s, ring) => s + signedArea2(ring) / 2, 0));
    const total = areas.reduce((s, a) => s + a, 0);
    return areas.map((a) => a / total);
}
