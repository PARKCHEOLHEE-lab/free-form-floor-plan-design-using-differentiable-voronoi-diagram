// Pointer editing of walls while the Adjustment panel is open: hover a wall to see the straight
// run that would move, drag it to slide that run (wall-edit.js), with the Draw boundary snaps.
// Near a corner (8 px) the corner is picked instead, and a drag moves that corner. A click selects
// a corner, Delete or Backspace removes it, and a double click on a wall adds a corner there. A click
// on a wall selects it and shows a round handle; dragging the handle turns the wall about its middle.
import { segmentDistance } from './geom.js';
import { wallEdges } from './graph.js';
import { SNAP_LABELS } from './boundary-editor.js';
import { CELL_LINE, SITE } from './view3d.js';
import { addCorner, movableParts, moveCorner, moveRun, removeCorner, rotateRun, snapAngle, snapCorner, snapOffset, wallRun } from './wall-edit.js';

const PICK = 6;        // css px from a wall that still picks it (as the bubble editor picks a door link)
const CORNER_PICK = 8; // css px from a corner that picks the corner before the walls through it
const DRAG_START = 3;  // css px a press must travel before it drags
const HANDLE = 28;     // css px from the middle of a selected wall to its rotate handle, along its normal
const HANDLE_HIT = 9;  // css px around the handle that grab it (as the bubble editor's resize handle)

/**
 * `plan()` is the graph on screen (the edited copy, or the live aligned plan) or null;
 * `boundary()` is the plan boundary ring (corner snaps line walls up with its edges);
 * `canEdit()` says whether walls may move now; `onChange(graph, commit)` shows a moved plan and
 * commits it when the drag ends; `onCancel()` puts back the plan from before the drag;
 * `onStatus(text)` shows why an edit was refused ('' clears it).
 */
const REFUSED = 'This corner cannot be deleted. Only a corner where two walls meet can go, and the joined wall must stay clear of other walls.';

export function createWallEditor(canvas, view, { overlay, plan, boundary, canEdit, onChange, onCancel, onStatus }) {
    const path = overlay.querySelector('[data-wall]');
    const spot = overlay.querySelector('[data-corner]');
    const chosen = overlay.querySelector('[data-selected]');
    const chosenWall = overlay.querySelector('[data-chosen]');
    const knob = overlay.querySelector('[data-handle]');
    const guide = overlay.querySelector('[data-guide]');
    const marker = overlay.querySelector('[data-cursor]');
    const label = overlay.querySelector('text');
    const marks = overlay.querySelector('[data-movable]');
    // the marks look like the Raw view's Voronoi: walls as its cell lines, corners as its sites
    marks.style.setProperty('--mark-line', CELL_LINE.color);
    marks.style.setProperty('--mark-line-opacity', CELL_LINE.opacity);
    marks.style.setProperty('--mark-site', SITE.color);
    marks.style.setProperty('--mark-site-opacity', SITE.opacity);
    let marked = null;  // { g, corners: [[id, circle]], walls: [[a, b, line]] } marked on plan g
    let enabled = false;
    let hover = null;   // { g, run } (a wall) or { g, id } (a corner) under the pointer
    let press = null;   // hover plus { start, screen, dragged, result }
    let selected = null; // { id } of the selected corner, or { wall: [a, b] } of the selected wall

    const screenOf = (g, id) => view.screenPoint([g.corners.get(id).x, g.corners.get(id).y]);

    /** The interior wall nearest to screen point (x, y), within PICK. */
    function wallAt(g, x, y) {
        let best = null, nearest = PICK;
        for (const e of wallEdges(g)) {
            if (e.rooms.length !== 2) continue;
            const d = segmentDistance([x, y], screenOf(g, e.a), screenOf(g, e.b));
            if (d < nearest) { nearest = d; best = e; }
        }
        return best;
    }

    function pick(x, y) {
        const g = plan();
        if (!g) return null;
        let corner = null, closest = CORNER_PICK;
        for (const [id] of g.corners) {
            const d = Math.hypot(...screenOf(g, id).map((v, i) => v - [x, y][i]));
            if (d < closest) { closest = d; corner = id; }
        }
        if (corner !== null && moveCorner(g, corner, [g.corners.get(corner).x, g.corners.get(corner).y])) return { g, id: corner }; // not a boundary bend
        const best = wallAt(g, x, y);
        const run = best && wallRun(g, best.a, best.b);
        return run && !run.locked ? { g, run, a: best.a, b: best.b } : null;
    }

    const runPath = (g, run) => run.edges.map(([a, b]) => `M${screenOf(g, a).join(',')}L${screenOf(g, b).join(',')}`).join('');
    /** The rotate handle of a run, HANDLE px from its middle along its normal, on screen. */
    function handleAt(run) {
        const m = view.screenPoint(run.mid);
        const q = view.screenPoint([run.mid[0] + run.normal[0], run.mid[1] + run.normal[1]]);
        const l = Math.hypot(q[0] - m[0], q[1] - m[1]) || 1;
        return [m[0] + (q[0] - m[0]) / l * HANDLE, m[1] + (q[1] - m[1]) / l * HANDLE];
    }
    /** The selected wall's run on the plan now, or null. */
    function selectedRun() {
        const g = plan();
        const run = selected?.wall && g && wallRun(g, ...selected.wall);
        return run && !run.locked ? run : null;
    }

    function drawSelection() {
        const g = plan();
        const run = selectedRun();
        if (selected?.wall ? !run : selected && !g?.corners.has(selected.id)) selected = null; // undone or rebuilt
        const corner = !!selected && !selected.wall;
        if (corner) canvas.dataset.selectedCorner = selected.id; else delete canvas.dataset.selectedCorner;
        if (selected?.wall) canvas.dataset.selectedWall = selected.wall.join(','); else delete canvas.dataset.selectedWall;
        chosen.toggleAttribute('hidden', !corner);
        chosenWall.toggleAttribute('hidden', !selected?.wall);
        knob.toggleAttribute('hidden', !selected?.wall);
        if (corner) {
            const [cx, cy] = screenOf(g, selected.id);
            chosen.setAttribute('cx', cx);
            chosen.setAttribute('cy', cy);
        } else if (selected) {
            chosenWall.setAttribute('d', runPath(g, run));
            const [hx, hy] = handleAt(run);
            knob.setAttribute('cx', hx);
            knob.setAttribute('cy', hy);
        }
    }

    function select(next) {
        selected = next;
        onStatus('');
        drawSelection();
    }

    /**
     * Mark every corner and wall a drag can move, so the panel shows what to grab before any
     * hover; listed once per plan, placed again on every redraw. Returns whether marks show.
     */
    function drawMarks() {
        const g = plan();
        const shown = enabled && !!g && canEdit();
        marks.toggleAttribute('hidden', !shown);
        if (!shown) return false;
        marks.style.setProperty('--mark-line-width', `${1 / (window.devicePixelRatio || 1)}px`); // WebGL lines are one device pixel
        if (marked?.g !== g) {
            const make = (tag) => document.createElementNS(marks.namespaceURI, tag);
            const { corners, walls } = movableParts(g);
            marked = { g, corners: corners.map((id) => [id, make('circle')]), walls: walls.map(([a, b]) => [a, b, make('line')]) };
            for (const [, c] of marked.corners) c.setAttribute('r', SITE.size / 2);
            marks.replaceChildren(...marked.walls.map((w) => w[2]), ...marked.corners.map((c) => c[1]));
        }
        for (const [id, c] of marked.corners) {
            const [x, y] = screenOf(g, id);
            c.setAttribute('cx', x);
            c.setAttribute('cy', y);
        }
        for (const [a, b, l] of marked.walls) {
            const [[x1, y1], [x2, y2]] = [screenOf(g, a), screenOf(g, b)];
            for (const [k, v] of Object.entries({ x1, y1, x2, y2 })) l.setAttribute(k, v);
        }
        return true;
    }

    function draw(g, item, snap) {
        drawSelection();
        overlay.toggleAttribute('hidden', !drawMarks() && !item && !selected);
        path.toggleAttribute('hidden', !item?.run);
        spot.toggleAttribute('hidden', !item || !!item.run);
        if (!item) { for (const el of [guide, marker, label]) el.toggleAttribute('hidden', true); return; }
        const run = item.run;
        if (run) path.setAttribute('d', runPath(g, run));
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

    const onKnob = (e) => {
        const run = selectedRun();
        return run && Math.hypot(...handleAt(run).map((v, i) => v - [e.clientX, e.clientY][i])) <= HANDLE_HIT ? run : null;
    };

    function down(e) {
        if (!enabled || e.button !== 0 || !canEdit()) return;
        const turning = onKnob(e);
        if (turning) {
            const start = view.pointOnPlan(e.clientX, e.clientY);
            if (!start) return;
            e.preventDefault();
            e.stopImmediatePropagation();
            canvas.setPointerCapture?.(e.pointerId);
            press = { g: plan(), run: turning, turn: true, wall: selected.wall, start, screen: [e.clientX, e.clientY], dragged: false, result: null };
            return;
        }
        const hit = pick(e.clientX, e.clientY);
        if (!hit) { if (selected) select(null); return; }
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
        if (!press) {
            const hit = canEdit() ? pick(e.clientX, e.clientY) : null;
            setHover(hit);
            // a wall that cannot move (it ends where the boundary bends) shows a not-allowed cursor
            const g = plan(), wall = !hit && canEdit() && g && wallAt(g, e.clientX, e.clientY);
            canvas.toggleAttribute('data-wall-locked', !!wall && !!wallRun(g, wall.a, wall.b)?.locked);
            canvas.toggleAttribute('data-handle-hover', canEdit() && !!onKnob(e));
            return;
        }
        if (!press.dragged && Math.hypot(e.clientX - press.screen[0], e.clientY - press.screen[1]) < DRAG_START) return;
        press.dragged = true;
        const p = view.pointOnPlan(e.clientX, e.clientY);
        if (!p) return;
        if (press.turn) {
            const { run } = press;
            const from = Math.atan2(press.start[1] - run.mid[1], press.start[0] - run.mid[0]);
            const turn = Math.atan2(p[1] - run.mid[1], p[0] - run.mid[0]) - from;
            const snap = snapAngle(run, Math.atan2(Math.sin(turn), Math.cos(turn)), { boundary: boundary(), shift: e.shiftKey });
            press.result = rotateRun(press.g, run, snap.angle);
            onChange(press.result.graph, false);
            draw(press.result.graph, null, null);
            // the turned wall and its handle are drawn as the selection; the snap names the line-up
            const text = Math.abs(press.result.angle - snap.angle) < 1e-12 ? SNAP_LABELS[snap.kind] : null;
            label.toggleAttribute('hidden', !text);
            if (text) {
                label.textContent = text;
                label.setAttribute('x', +knob.getAttribute('cx') + 12);
                label.setAttribute('y', +knob.getAttribute('cy') - 8);
            }
            return;
        }
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
        const { dragged, result, run, id, a, b, turn, wall } = press;
        press = null;
        if (dragged && result) onChange(result.graph, true);
        select(turn ? { wall } : run ? { wall: [a, b] } : { id });   // a click or drag selects what it pressed
        setHover(null);
    }

    // a double click on a wall adds a corner there (and keeps the view where it is)
    function addAt(e) {
        if (!enabled || !canEdit()) return;
        const g = plan();
        const wall = g && wallAt(g, e.clientX, e.clientY);
        const p = wall && view.pointOnPlan(e.clientX, e.clientY);
        const added = p && addCorner(g, wall.a, wall.b, p);
        if (!added) return;
        e.preventDefault();
        e.stopImmediatePropagation();
        onChange(added.graph, true);
        select({ id: added.id });
        setHover({ g: added.graph, id: added.id });
    }

    function key(e) {
        if (!enabled) return;
        if (e.key === 'Escape') {
            if (press) { press = null; onCancel(); setHover(null); } else if (selected) select(null);
            return;
        }
        if ((e.key !== 'Delete' && e.key !== 'Backspace') || !selected || selected.wall || press || !canEdit()) return;
        if (e.target.closest?.('input, textarea, select, [contenteditable]')) return;
        e.preventDefault();
        const h = removeCorner(plan(), selected.id);
        if (!h) { onStatus(REFUSED); return; }
        onChange(h, true);
        select(null);
        setHover(null);
    }

    canvas.addEventListener('pointerdown', down, true);
    canvas.addEventListener('dblclick', addAt, true);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointercancel', up);
    window.addEventListener('keydown', key);
    const offView = view.onChange(() => { if (!press) draw(hover?.g, hover, null); });

    return {
        setEnabled(next) {
            if (press) { press = null; onCancel(); }
            enabled = next;
            if (!next) selected = null;
            setHover(null);
        },
        /** Redraw after the plan changed outside the editor (undo, redo, discard). */
        sync() { if (!press) setHover(null); },
        dispose() {
            canvas.removeEventListener('pointerdown', down, true);
            canvas.removeEventListener('dblclick', addAt, true);
            canvas.removeEventListener('pointermove', move);
            canvas.removeEventListener('pointerup', up);
            canvas.removeEventListener('pointercancel', up);
            window.removeEventListener('keydown', key);
            offView();
        },
    };
}
