// Moving one interior wall of the aligned plan by hand (the Adjustment panel). The plan is the
// corner graph of graph.js; a move returns a new graph and leaves the given one untouched.
import { components } from "./align.js";
import { SNAP_RADIUS, snapBoundaryPoint, visitSnapTargets } from "./boundary-editor.js";
import { roomRings, wallEdges } from "./graph.js";
import { segmentDistance, signedArea2 } from "./geom.js";

const TOL = 1e-9;
const KINK = Math.sin(10 * Math.PI / 180); // walls within 10 degrees count as one straight line (alignment leaves small kinks)
const MIN_WALL = 0.01;  // the shortest wall a move may leave (plan units)

const point = (g, id) => [g.corners.get(id).x, g.corners.get(id).y];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];
const cross = (a, b) => a[0] * b[1] - a[1] * b[0];
const unit = (v) => { const l = Math.hypot(v[0], v[1]); return [v[0] / l, v[1] / l]; };
const span = (g, e) => Math.hypot(...sub(point(g, e.a), point(g, e.b)));

/** The rooms with a ring that uses one of the corners `ids`. */
const usersOf = (g, ids) => g.rooms.map((rings, r) => (rings.some((ring) => ring.some((x) => ids.has(x))) ? r : -1)).filter((r) => r >= 0);

/** A corner and the corners on the same point (joined by walls of no length). */
function twinsOf(g, id, all) {
    const group = new Set([id]);
    for (const x of group) for (const e of all) if ((e.a === x || e.b === x) && span(g, e) < TOL) group.add(e.a === x ? e.b : e.a);
    return group;
}

/**
 * The straight wall through the interior edge a-b: every interior edge within 10 degrees of its
 * direction that connects to it, so a wall passing a T junction moves as one piece. Each corner of
 * the run slides along the other wall it meets, so neighbouring walls keep their direction; where
 * walls meet at different angles (a Y junction) it slides along the one most across the run and
 * the others turn about their far ends. A boundary corner slides along the boundary, so a run that
 * ends where the boundary bends, or where the boundary runs on in the wall's direction, is
 * `locked`. Alignment can snap two corners onto one point; such twins move together. Returns null
 * when a-b is not an interior wall of some length.
 * The normal points right (+x), or up for a horizontal wall; a positive offset moves that way.
 */
export function wallRun(g, a, b) {
    const all = wallEdges(g);
    const interior = all.filter((e) => e.rooms.length === 2);
    const pick = interior.find((e) => (e.a === a && e.b === b) || (e.a === b && e.b === a));
    const length = (e) => span(g, e);
    if (!pick || length(pick) < TOL) return null;
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
        if (slides.has(id)) continue; // a twin already placed
        const twins = twinsOf(g, id, all);
        const p = point(g, id);
        const meeting = all.filter((e) => (twins.has(e.a) || twins.has(e.b)) && !inRun.has(`${e.a},${e.b}`) && length(e) >= TOL);
        const place = (v) => { for (const x of twins) slides.set(x, v); };
        if (!meeting.length) { place(normal); continue; } // a point inside the straight wall moves with it
        const fixed = [...twins].some((x) => g.corners.get(x).fixed);
        // a boundary corner slides along the boundary; an inner one along the wall most across the run
        const guides = fixed ? meeting.filter((e) => e.rooms.length === 1) : meeting;
        const ends = guides.map((e) => point(g, twins.has(e.a) ? e.b : e.a));
        const dirs = ends.map((q) => unit(sub(q, p)));
        const lead = fixed ? dirs[0] : dirs.reduce((best, d) => (Math.abs(dot(d, normal)) > Math.abs(dot(best, normal)) ? d : best));
        const onGuide = dirs.map((d) => Math.abs(cross(lead, d)) <= KINK); // the guide wall, on both sides of the corner
        if (fixed && !onGuide.every(Boolean)) { locked = true; break; } // the boundary bends here
        const others = ends.filter((_, i) => onGuide[i]);
        const w = unit(dirs.filter((_, i) => onGuide[i]).reduce((s, d) => (dot(d, lead) >= 0 ? [s[0] + d[0], s[1] + d[1]] : [s[0] - d[0], s[1] - d[1]]), [0, 0]));
        const along = dot(w, normal);
        if (Math.abs(along) < KINK) { locked = true; break; }
        const rate = 1 / along;
        place([w[0] * rate, w[1] * rate]);
        // stop before any wall this corner slides along gets shorter than MIN_WALL
        for (const q of others) {
            const k = dot(sub(q, p), w);
            const limit = (k > 0 ? k - MIN_WALL : k + MIN_WALL) / rate;
            if ((k > 0) === (rate > 0)) hi = Math.min(hi, limit); else lo = Math.max(lo, limit);
        }
    }
    return { edges, corners, rooms: usersOf(g, new Set(slides.keys())), origin, normal, slides, range: [lo, hi], locked };
}

/**
 * A check that h, a changed copy of plan g, is still a plan: none of the `changed` walls (pairs of
 * corner ids, as they are in h) crosses another wall or newly touches a corner or gets shorter than
 * MIN_WALL (or than it already was), and each of the `rooms` keeps its rings and their orientation.
 * (Rings are not required to be simple: alignment can leave a ring touching itself.)
 */
function stillAPlan(g, changed, rooms) {
    const sides = rooms.map((r) => roomRings(g, r).map((ring) => Math.sign(signedArea2(ring))));
    const side = (a, b, c) => cross(sub(b, a), sub(c, a));
    const crosses = (h, e, f) => {
        if (e.a === f.a || e.a === f.b || e.b === f.a || e.b === f.b) return false;
        const [p1, p2, q1, q2] = [point(h, e.a), point(h, e.b), point(h, f.a), point(h, f.b)];
        return side(p1, p2, q1) * side(p1, p2, q2) < 0 && side(q1, q2, p1) * side(q1, q2, p2) < 0;
    };
    // a corner lying on a changed wall it does not end (corners on one point as its ends are its twins)
    const touches = (h) => {
        const found = new Set();
        for (const e of changed) {
            const a = point(h, e.a), b = point(h, e.b);
            for (const [id, c] of h.corners) {
                const p = [c.x, c.y];
                if (Math.hypot(...sub(p, a)) < TOL || Math.hypot(...sub(p, b)) < TOL) continue;
                if (segmentDistance(p, a, b) < TOL) found.add(`${id}:${e.a},${e.b}`);
            }
        }
        return found;
    };
    const touching = touches(g);
    return (h) => {
        const edges = wallEdges(h);
        return changed.every((e) => span(h, e) >= Math.min(MIN_WALL, span(g, e)) - TOL) &&
            changed.every((e) => edges.every((f) => !crosses(h, e, f))) &&
            [...touches(h)].every((k) => touching.has(k)) &&
            rooms.every((r, k) => {
                const rings = roomRings(h, r);
                return rings.length === sides[k].length && rings.every((ring, i) => Math.sign(signedArea2(ring)) === sides[k][i]);
            });
    };
}

/**
 * Move a run by `offset` along its normal, clamped to its range and then to the largest offset at
 * which no moved wall crosses or newly touches another wall or corner or gets shorter than
 * MIN_WALL (or than it already was), and every room beside it keeps its orientation (stillAPlan).
 * Returns { graph, offset }, or null for a locked run.
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
    const valid = stillAPlan(g, wallEdges(g).filter((e) => run.slides.has(e.a) || run.slides.has(e.b)), run.rooms);
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

/**
 * Drag one corner (with its twins) toward `target`. A boundary corner moves only along the boundary
 * segment it lies on; one where the boundary bends cannot move (null). The move is moveRun on a
 * one-corner run, so it stops at the same last valid position. Returns { graph, point }.
 */
export function moveCorner(g, id, target) {
    const all = wallEdges(g);
    const twins = twinsOf(g, id, all);
    const p = point(g, id);
    let d = sub(target, p);
    if ([...twins].some((x) => g.corners.get(x).fixed)) {
        const ends = all.filter((e) => (twins.has(e.a) || twins.has(e.b)) && e.rooms.length === 1 && span(g, e) >= TOL)
            .map((e) => point(g, twins.has(e.a) ? e.b : e.a));
        const dirs = ends.map((q) => unit(sub(q, p)));
        if (!dirs.length || dirs.some((v) => Math.abs(cross(dirs[0], v)) > KINK)) return null; // the boundary bends here
        const along = dot(d, dirs[0]);
        d = [dirs[0][0] * along, dirs[0][1] * along];
    }
    const run = { slides: new Map([...twins].map((x) => [x, d])), rooms: usersOf(g, twins), range: [0, 1], locked: false };
    const { graph } = moveRun(g, run, 1);
    return { graph, point: point(graph, id) };
}

/**
 * Snap a dragged corner the way Draw boundary snaps a point. With Shift it stays on the horizontal
 * or vertical line through `start` ('axis'). Otherwise, within SNAP_RADIUS screen pixels, it jumps
 * onto a corner that stays put ('vertex') or the middle of a wall that does not move with it
 * ('midpoint'); failing that, a wall to one of its neighbours turns parallel or perpendicular to a
 * boundary edge (snapBoundaryPoint with that neighbour as the last point). Returns { point, kind,
 * target?, from? }; kind 'free' and the given point when nothing is close enough.
 */
export function snapCorner(g, id, p, { project, shift, start, boundary }) {
    if (shift) {
        const keep = Math.abs(p[0] - start[0]) >= Math.abs(p[1] - start[1]) ? 1 : 0;
        const locked = [...p];
        locked[keep] = start[keep];
        return { point: locked, kind: "axis" };
    }
    const all = wallEdges(g);
    const twins = twinsOf(g, id, all);
    const touching = all.filter((e) => twins.has(e.a) || twins.has(e.b));
    const neighbours = new Set(touching.map((e) => (twins.has(e.a) ? e.b : e.a)).filter((x) => !twins.has(x)));
    const points = [...g.corners].filter(([x]) => !twins.has(x) && !neighbours.has(x)).map(([, c]) => [c.x, c.y]);
    const edges = all.filter((e) => !twins.has(e.a) && !twins.has(e.b)).map((e) => [point(g, e.a), point(g, e.b)]);
    const screen = project(p);
    const distance = (q) => Math.hypot(...sub(project(q), screen));
    let nearest = SNAP_RADIUS;
    let snapped = null;
    visitSnapTargets(points, edges, (q, kind) => {
        const d = distance(q);
        if (d < nearest) { nearest = d; snapped = { point: q, kind, target: q }; }
    });
    if (snapped) return snapped;
    nearest = SNAP_RADIUS;
    for (const x of neighbours) {
        const from = point(g, x);
        const result = snapBoundaryPoint({ point: p, screen, project, vertices: [from], reference: boundary, shift: false });
        if (result.kind === "free") continue;
        const d = distance(result.point);
        if (d < nearest) { nearest = d; snapped = { ...result, from }; }
    }
    return snapped ?? { point: p, kind: "free" };
}

/**
 * Delete a corner where exactly two walls meet: it leaves every ring, so its two walls become one
 * straight wall between its neighbours. Refused (null) where three or more walls meet, where the
 * boundary bends, for a corner sharing its point with another, and when the new wall would not
 * leave a plan (stillAPlan: crossing, touching, too short, a room turned over or gone).
 */
export function removeCorner(g, id) {
    const all = wallEdges(g);
    if (twinsOf(g, id, all).size > 1) return null;
    const walls = all.filter((e) => e.a === id || e.b === id);
    if (walls.length !== 2) return null;
    const [a, b] = walls.map((e) => (e.a === id ? e.b : e.a));
    const p = point(g, id);
    if (g.corners.get(id).fixed && Math.abs(cross(unit(sub(point(g, a), p)), unit(sub(point(g, b), p)))) > TOL) return null; // the boundary bends here
    const h = {
        corners: new Map([...g.corners].filter(([x]) => x !== id).map(([x, c]) => [x, { ...c }])),
        rooms: g.rooms.map((rings) => rings.map((ring) => ring.filter((x) => x !== id))),
    };
    return stillAPlan(g, [{ a, b }], usersOf(g, new Set([id])))(h) ? h : null;
}

/**
 * Add a corner on the wall a-b at the point nearest to p (kept MIN_WALL from both ends), in every
 * ring that runs along that wall, so the areas stay the same; on the boundary it is a boundary
 * corner. Returns { graph, id }, or null when a-b is not a wall or too short to split.
 */
export function addCorner(g, a, b, p) {
    const wall = wallEdges(g).find((e) => (e.a === a && e.b === b) || (e.a === b && e.b === a));
    if (!wall || span(g, wall) < 2 * MIN_WALL) return null;
    const A = point(g, a);
    const u = unit(sub(point(g, b), A));
    const s = Math.min(Math.max(dot(sub(p, A), u), MIN_WALL), span(g, wall) - MIN_WALL);
    const id = Math.max(...g.corners.keys()) + 1;
    const corners = new Map([...g.corners].map(([x, c]) => [x, { ...c }]));
    corners.set(id, { x: A[0] + u[0] * s, y: A[1] + u[1] * s, fixed: wall.rooms.length === 1 });
    const rooms = g.rooms.map((rings) => rings.map((ring) => ring.flatMap((x, i) => {
        const next = ring[(i + 1) % ring.length];
        return (x === a && next === b) || (x === b && next === a) ? [x, id] : [x];
    })));
    return { graph: { corners, rooms }, id };
}

/** Each room's share of the plan area (holes subtract). */
export function roomShares(g) {
    const areas = g.rooms.map((_, r) => roomRings(g, r).reduce((s, ring) => s + signedArea2(ring) / 2, 0));
    const total = areas.reduce((s, a) => s + a, 0);
    return areas.map((a) => a / total);
}

/**
 * Snap a drag offset the way Draw boundary snaps a point: when the moved wall would pass within
 * SNAP_RADIUS screen pixels of a corner that stays put ('vertex', shown as Endpoint) or the middle
 * of a wall that does not move with it ('midpoint'), the offset jumps to pass exactly through it.
 * `project` maps a plan point to screen pixels. Returns { offset, kind, target }; kind 'free' and
 * the given offset when nothing is close enough.
 */
export function snapOffset(g, run, offset, project) {
    const moving = new Set(run.slides.keys());
    const points = [...g.corners].filter(([id]) => !moving.has(id)).map(([, c]) => [c.x, c.y]);
    const edges = wallEdges(g).filter((e) => !moving.has(e.a) && !moving.has(e.b)).map((e) => [point(g, e.a), point(g, e.b)]);
    let nearest = SNAP_RADIUS;
    let snapped = { offset, kind: "free" };
    visitSnapTargets(points, edges, (q, kind) => {
        const t = dot(sub(q, run.origin), run.normal);
        const foot = [q[0] - (t - offset) * run.normal[0], q[1] - (t - offset) * run.normal[1]];
        const [x0, y0] = project(q), [x1, y1] = project(foot);
        const distance = Math.hypot(x0 - x1, y0 - y1);
        if (distance < nearest) {
            nearest = distance;
            snapped = { offset: t, kind, target: q };
        }
    });
    return snapped;
}
