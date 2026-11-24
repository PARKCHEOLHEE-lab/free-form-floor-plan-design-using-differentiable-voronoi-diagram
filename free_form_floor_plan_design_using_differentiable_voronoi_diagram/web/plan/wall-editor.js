// Pointer editing of walls while the Adjustment panel is open: hover a wall to see the straight
// run that would move, drag it to slide that run (wall-edit.js), with the Draw boundary snaps.
// Near a corner (8 px) the corner is picked instead, and a drag moves that corner.
import { segmentDistance } from './geom.js';
import { wallEdges } from './graph.js';
import { SNAP_LABELS } from './boundary-editor.js';
import { moveCorner, moveRun, snapCorner, snapOffset, wallRun } from './wall-edit.js';

const PICK = 6;        // css px from a wall that still picks it (as the bubble editor picks a door link)
const CORNER_PICK = 8; // css px from a corner that picks the corner before the walls through it
const DRAG_START = 3;  // css px a press must travel before it drags

/**
 * `plan()` is the graph on screen (the edited copy, or the live aligned plan) or null;
 * `boundary()` is the plan boundary ring (corner snaps line walls up with its edges);
 * `canEdit()` says whether walls may move now; `onChange(graph, commit)` shows a moved plan and
 * commits it when the drag ends; `onCancel()` puts back the plan from before the drag.
 */
export function createWallEditor(canvas, view, { overlay, plan, boundary, canEdit, onChange, onCancel }) {
    const path = overlay.querySelector('[data-wall]');
    const spot = overlay.querySelector('[data-corner]');
    const guide = overlay.querySelector('[data-guide]');
    const marker = overlay.querySelector('[data-cursor]');
    const label = overlay.querySelector('text');
    let enabled = false;
    let hover = null;   // { g, run } (a wall) or { g, id } (a corner) under the pointer
    let press = null;   // hover plus { start, screen, dragged, result }

    const screenOf = (g, id) => view.screenPoint([g.corners.get(id).x, g.corners.get(id).y]);

    function pick(x, y) {
        const g = plan();
        if (!g) return null;
        let corner = null, closest = CORNER_PICK;
        for (const [id] of g.corners) {
            const d = Math.hypot(...screenOf(g, id).map((v, i) => v - [x, y][i]));
            if (d < closest) { closest = d; corner = id; }
        }
        if (corner !== null && moveCorner(g, corner, [g.corners.get(corner).x, g.corners.get(corner).y])) return { g, id: corner }; // not a boundary bend
        let best = null, nearest = PICK;
        for (const e of wallEdges(g)) {
            if (e.rooms.length !== 2) continue;
            const d = segmentDistance([x, y], screenOf(g, e.a), screenOf(g, e.b));
            if (d < nearest) { nearest = d; best = e; }
        }
        const run = best && wallRun(g, best.a, best.b);
        return run && !run.locked ? { g, run } : null;
    }

    function draw(g, item, snap) {
        overlay.toggleAttribute('hidden', !item);
        if (!item) return;
        const run = item.run;
        path.toggleAttribute('hidden', !run);
        spot.toggleAttribute('hidden', !!run);
        if (run) path.setAttribute('d', run.edges.map(([a, b]) => `M${screenOf(g, a).join(',')}L${screenOf(g, b).join(',')}`).join(''));
        else {
            const [cx, cy] = screenOf(g, item.id);
            spot.setAttribute('cx', cx);
            spot.setAttribute('cy', cy);
        }
        const shown = !!snap && snap.kind !== 'free';
        for (const el of [guide, marker, label]) el.toggleAttribute('hidden', !shown);
        if (!shown) return;
        const [x, y] = view.screenPoint(snap.target ?? snap.point);
        marker.setAttribute('cx', x);
        marker.setAttribute('cy', y);
        label.textContent = SNAP_LABELS[snap.kind];
        label.setAttribute('x', x + 12);
        label.setAttribute('y', y - 8);
        // a dashed guide from the snap target to the nearest end of the moved wall, or for a
        // corner from the neighbour it lines up with (or its start, for X/Y)
        if (!run) {
            const from = snap.from ?? (snap.kind === 'axis' ? press?.start : null);
            guide.toggleAttribute('hidden', !from);
            if (from) {
                const [fx, fy] = view.screenPoint(from);
                guide.setAttribute('x1', x); guide.setAttribute('y1', y);
                guide.setAttribute('x2', fx); guide.setAttribute('y2', fy);
            }
            return;
        }
        const ends = run.corners.map((id) => screenOf(g, id));
        const near = ends.reduce((a, b) => (Math.hypot(a[0] - x, a[1] - y) <= Math.hypot(b[0] - x, b[1] - y) ? a : b));
        guide.setAttribute('x1', x); guide.setAttribute('y1', y);
        guide.setAttribute('x2', near[0]); guide.setAttribute('y2', near[1]);
    }

    function setHover(next) {
        hover = next;
        if (hover?.run) canvas.dataset.wallHover = ''; else delete canvas.dataset.wallHover;
        if (hover && !hover.run) {
            canvas.dataset.cornerHover = hover.id;
            canvas.toggleAttribute('data-corner-boundary', hover.g.corners.get(hover.id).fixed);
        } else { delete canvas.dataset.cornerHover; canvas.removeAttribute('data-corner-boundary'); }
        draw(hover?.g, hover, null);
    }

    function down(e) {
        if (!enabled || e.button !== 0 || !canEdit()) return;
        const hit = pick(e.clientX, e.clientY);
        if (!hit) return;
        const start = view.pointOnPlan(e.clientX, e.clientY);
        if (!start) return;
        e.preventDefault();
        e.stopImmediatePropagation();
        canvas.setPointerCapture?.(e.pointerId);
        if (!hit.run) { const c = hit.g.corners.get(hit.id); start[0] = c.x; start[1] = c.y; } // a corner moves from where it is
        press = { ...hit, start, screen: [e.clientX, e.clientY], dragged: false, result: null };
    }

    function move(e) {
        if (!enabled) return;
        if (!press) { setHover(canEdit() ? pick(e.clientX, e.clientY) : null); return; }
        if (!press.dragged && Math.hypot(e.clientX - press.screen[0], e.clientY - press.screen[1]) < DRAG_START) return;
        press.dragged = true;
        const p = view.pointOnPlan(e.clientX, e.clientY);
        if (!p) return;
        if (!press.run) {
            const snap = snapCorner(press.g, press.id, p, { project: view.screenPoint, shift: e.shiftKey, start: press.start, boundary: boundary() });
            press.result = moveCorner(press.g, press.id, snap.point);
            onChange(press.result.graph, false);
            const reached = Math.hypot(press.result.point[0] - snap.point[0], press.result.point[1] - snap.point[1]) < 1e-12;
            draw(press.result.graph, press, reached ? snap : null);
            return;
        }
        const { normal } = press.run;
        const raw = (p[0] - press.start[0]) * normal[0] + (p[1] - press.start[1]) * normal[1];
        const snap = snapOffset(press.g, press.run, raw, view.screenPoint);
        press.result = moveRun(press.g, press.run, snap.offset);
        onChange(press.result.graph, false);
        draw(press.result.graph, press, Math.abs(press.result.offset - snap.offset) < 1e-12 ? snap : null);
    }

    function up() {
        if (!press) return;
        const { dragged, result } = press;
        press = null;
        if (dragged && result) onChange(result.graph, true);
        setHover(null);
    }

    function key(e) {
        if (e.key !== 'Escape' || !press) return;
        press = null;
        onCancel();
        setHover(null);
    }

    canvas.addEventListener('pointerdown', down, true);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);
    window.addEventListener('keydown', key);
    const offView = view.onChange(() => { if (!press) draw(hover?.g, hover, null); });

    return {
        setEnabled(next) {
            if (press) { press = null; onCancel(); }
            enabled = next;
            setHover(null);
        },
        dispose() {
            canvas.removeEventListener('pointerdown', down, true);
            canvas.removeEventListener('pointermove', move);
            canvas.removeEventListener('pointerup', up);
            canvas.removeEventListener('pointercancel', up);
            window.removeEventListener('keydown', key);
            offView();
        },
    };
}
