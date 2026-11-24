// Pointer editing of walls while the Adjustment panel is open: hover a wall to see the straight
// run that would move, drag it to slide that run (wall-edit.js), with the Draw boundary snaps.
import { segmentDistance } from './geom.js';
import { wallEdges } from './graph.js';
import { SNAP_LABELS } from './boundary-editor.js';
import { moveRun, snapOffset, wallRun } from './wall-edit.js';

const PICK = 6;        // css px from a wall that still picks it (as the bubble editor picks a door link)
const DRAG_START = 3;  // css px a press must travel before it drags

/**
 * `plan()` is the graph on screen (the edited copy, or the live aligned plan) or null;
 * `canEdit()` says whether walls may move now; `onChange(graph, commit)` shows a moved plan and
 * commits it when the drag ends; `onCancel()` puts back the plan from before the drag.
 */
export function createWallEditor(canvas, view, { overlay, plan, canEdit, onChange, onCancel }) {
    const path = overlay.querySelector('[data-wall]');
    const guide = overlay.querySelector('[data-guide]');
    const marker = overlay.querySelector('[data-cursor]');
    const label = overlay.querySelector('text');
    let enabled = false;
    let hover = null;   // { g, run } under the pointer
    let press = null;   // { g, run, start, screen, dragged, result }

    const screenOf = (g, id) => view.screenPoint([g.corners.get(id).x, g.corners.get(id).y]);

    function pick(x, y) {
        const g = plan();
        if (!g) return null;
        let best = null, nearest = PICK;
        for (const e of wallEdges(g)) {
            if (e.rooms.length !== 2) continue;
            const d = segmentDistance([x, y], screenOf(g, e.a), screenOf(g, e.b));
            if (d < nearest) { nearest = d; best = e; }
        }
        const run = best && wallRun(g, best.a, best.b);
        return run && !run.locked ? { g, run } : null;
    }

    function draw(g, run, snap) {
        overlay.toggleAttribute('hidden', !run);
        if (!run) return;
        path.setAttribute('d', run.edges.map(([a, b]) => `M${screenOf(g, a).join(',')}L${screenOf(g, b).join(',')}`).join(''));
        const shown = !!snap && snap.kind !== 'free';
        for (const el of [guide, marker, label]) el.toggleAttribute('hidden', !shown);
        if (!shown) return;
        const [x, y] = view.screenPoint(snap.target);
        marker.setAttribute('cx', x);
        marker.setAttribute('cy', y);
        label.textContent = SNAP_LABELS[snap.kind];
        label.setAttribute('x', x + 12);
        label.setAttribute('y', y - 8);
        // a dashed guide from the snap target to the nearest end of the moved wall
        const ends = run.corners.map((id) => screenOf(g, id));
        const near = ends.reduce((a, b) => (Math.hypot(a[0] - x, a[1] - y) <= Math.hypot(b[0] - x, b[1] - y) ? a : b));
        guide.setAttribute('x1', x); guide.setAttribute('y1', y);
        guide.setAttribute('x2', near[0]); guide.setAttribute('y2', near[1]);
    }

    function setHover(next) {
        hover = next;
        if (hover) canvas.dataset.wallHover = ''; else delete canvas.dataset.wallHover;
        draw(hover?.g, hover?.run, null);
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
        press = { ...hit, start, screen: [e.clientX, e.clientY], dragged: false, result: null };
    }

    function move(e) {
        if (!enabled) return;
        if (!press) { setHover(canEdit() ? pick(e.clientX, e.clientY) : null); return; }
        if (!press.dragged && Math.hypot(e.clientX - press.screen[0], e.clientY - press.screen[1]) < DRAG_START) return;
        press.dragged = true;
        const p = view.pointOnPlan(e.clientX, e.clientY);
        if (!p) return;
        const { normal } = press.run;
        const raw = (p[0] - press.start[0]) * normal[0] + (p[1] - press.start[1]) * normal[1];
        const snap = snapOffset(press.g, press.run, raw, view.screenPoint);
        press.result = moveRun(press.g, press.run, snap.offset);
        onChange(press.result.graph, false);
        draw(press.result.graph, press.run, Math.abs(press.result.offset - snap.offset) < 1e-12 ? snap : null);
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
    const offView = view.onChange(() => { if (!press) draw(hover?.g, hover?.run, null); });

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
