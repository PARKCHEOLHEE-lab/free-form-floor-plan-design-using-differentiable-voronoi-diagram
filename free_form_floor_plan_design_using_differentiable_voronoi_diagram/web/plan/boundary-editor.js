import { isSimpleBoundary, openRing } from './geom.js';

const SNAP_RADIUS = 10; // CSS pixels, independent of zoom.
const SNAP_ANGLE_SINE = Math.sin(5 * Math.PI / 180);
const samePoint = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-9;
const segments = points => points.slice(1).map((end, i) => [points[i], end]);

function visitSnapTargets(points, edges, visit) {
    for (const p of points) visit(p, 'vertex');
    for (const [a, b] of edges) {
        visit([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], 'midpoint');
    }
}

export function snapBoundaryPoint({ point, screen, project, vertices, reference, shift }) {
    const last = vertices.at(-1);
    const dx = last ? point[0] - last[0] : 0;
    const dy = last ? point[1] - last[1] : 0;
    const fixedAxis = shift && last ? (Math.abs(dx) >= Math.abs(dy) ? 1 : 0) : null;
    const onAxis = p => fixedAxis === null || Math.abs(p[fixedAxis] - last[fixedAxis]) < 1e-9;
    const screenDistance = (p, cursor) => {
        const [x, y] = project(p);
        return Math.hypot(x - cursor[0], y - cursor[1]);
    };

    if (vertices.length >= 3 && onAxis(vertices[0]) && screenDistance(vertices[0], screen) <= SNAP_RADIUS) {
        return { point: vertices[0], kind: 'close' };
    }

    const ring = openRing(reference);
    const draftEdges = segments(vertices);
    const referenceEdges = segments(ring.length ? [...ring, ring[0]] : []);
    let nearest = SNAP_RADIUS;
    let candidate = null;
    function consider(p, kind, cursor = screen, trackingFrom = null) {
        const distance = screenDistance(p, cursor);
        if (distance < nearest) {
            nearest = distance;
            candidate = trackingFrom ? { point: p, kind, trackingFrom } : { point: p, kind };
        }
    }
    for (const [points, edges] of [[vertices, draftEdges], [ring, referenceEdges]]) {
        visitSnapTargets(points, edges, (p, kind) => {
            if ((!last || !samePoint(p, last)) && onAxis(p)) consider(p, kind);
        });
        if (candidate) return candidate;
    }

    if (fixedAxis !== null) {
        const lock = p => {
            const aligned = [...p];
            aligned[fixedAxis] = last[fixedAxis];
            return aligned;
        };
        const constrained = lock(point);
        const constrainedScreen = project(constrained);
        visitSnapTargets(vertices, draftEdges, (from, kind) => {
            const target = lock(from);
            if (samePoint(target, from) || samePoint(target, last) || samePoint(target, vertices[0])) return;
            consider(target, kind, constrainedScreen, from);
        });
        return candidate ?? { point: constrained, kind: 'axis' };
    }

    const length = Math.hypot(dx, dy);
    if (last && length > 0) {
        for (const edges of [draftEdges, referenceEdges]) {
            for (const [a, b] of edges) {
                const ex = b[0] - a[0], ey = b[1] - a[1];
                const edgeLength = Math.hypot(ex, ey);
                if (edgeLength === 0) continue;
                const limit = length * edgeLength * SNAP_ANGLE_SINE;
                const parallel = Math.abs(dx * ey - dy * ex) <= limit;
                const perpendicular = Math.abs(dx * ex + dy * ey) <= limit;
                if (!parallel && !perpendicular) continue;

                const [ux, uy] = parallel ? [ex, ey] : [-ey, ex];
                const along = (dx * ux + dy * uy) / (edgeLength * edgeLength);
                consider([last[0] + along * ux, last[1] + along * uy], parallel ? 'parallel' : 'perpendicular');
            }
            if (candidate) return candidate;
        }
    }
    return { point, kind: 'free' };
}

function drawGuide(line, start, end) {
    line.toggleAttribute('hidden', !start || !end);
    if (!start || !end) return;
    line.setAttribute('x1', start[0]);
    line.setAttribute('y1', start[1]);
    line.setAttribute('x2', end[0]);
    line.setAttribute('y2', end[1]);
}

/** A temporary sketch in the existing camera's plan coordinates. */
export function createBoundaryEditor(canvas, view, { overlay, reference, isBuilding, onActive, onClose, onError, onPoints = () => {} }) {
    const placed = overlay.querySelector('polyline');
    const guide = overlay.querySelector('[data-edge-guide]');
    const trackingGuide = overlay.querySelector('[data-tracking-guide]');
    const start = overlay.querySelector('[data-first]');
    const marker = overlay.querySelector('[data-cursor]');
    const label = overlay.querySelector('text');
    const snapLabels = { close: 'Close', vertex: 'Endpoint', midpoint: 'Midpoint', axis: 'X/Y', parallel: 'Parallel', perpendicular: 'Perpendicular' };
    let active = false;
    let vertices = [];
    let undone = []; // points taken back by undo, newest last, until a new point is placed
    let pointer = null;

    function snapped() {
        if (!pointer) return null;
        const { screen, shift } = pointer;
        const point = view.pointOnPlan(...screen);
        return point ? snapBoundaryPoint({ point, screen, shift, vertices, reference: reference(), project: view.screenPoint }) : null;
    }

    function render() {
        if (!active) return;
        const points = vertices.map(view.screenPoint);
        placed.setAttribute('points', points.map(p => p.join(',')).join(' '));
        start.toggleAttribute('hidden', !points.length);
        if (points.length) {
            start.setAttribute('cx', points[0][0]);
            start.setAttribute('cy', points[0][1]);
        }
        const result = snapped();
        overlay.dataset.snap = result?.kind ?? 'free';
        marker.toggleAttribute('hidden', !result);
        label.toggleAttribute('hidden', !snapLabels[result?.kind]);
        const end = result ? view.screenPoint(result.point) : null;
        const trackingStart = result?.trackingFrom && view.screenPoint(result.trackingFrom);
        drawGuide(guide, points.at(-1), end);
        drawGuide(trackingGuide, trackingStart, end);
        if (end) {
            const [x, y] = end;
            marker.setAttribute('cx', x);
            marker.setAttribute('cy', y);
            label.textContent = snapLabels[result.kind] ?? '';
            label.setAttribute('x', x + 12);
            label.setAttribute('y', y - 8);
        }
    }

    function setActive(next) {
        if (active === next) return;
        active = next;
        vertices = [];
        undone = [];
        pointer = null;
        canvas.toggleAttribute('data-boundary-drawing', active);
        overlay.toggleAttribute('hidden', !active);
        if (active) {
            canvas.focus({ preventScroll: true });
            render();
        } else {
            delete overlay.dataset.snap;
        }
        onActive(active);
    }

    function move(e) {
        if (!active) return;
        pointer = { screen: [e.clientX, e.clientY], shift: e.shiftKey };
        render();
    }

    function down(e) {
        if (e.button !== 0 || (!active && !isBuilding())) return;
        e.preventDefault();
        e.stopImmediatePropagation(); // The boundary tool owns left clicks while sketching or submitting.
        if (!active) return;
        canvas.focus({ preventScroll: true });
        pointer = { screen: [e.clientX, e.clientY], shift: e.shiftKey };
        const result = snapped();
        if (!result) return;
        if (result.kind === 'close') {
            if (!isSimpleBoundary(vertices)) {
                onError('Boundary crosses or retraces itself, or has no area. Press Backspace to undo.');
                return;
            }
            const boundary = vertices;
            setActive(false);
            onClose(boundary);
            return;
        }
        if (vertices.length && samePoint(result.point, vertices.at(-1))) return;
        vertices.push(result.point);
        undone = [];
        onError('');
        render();
        onPoints();
    }

    function undoPoint() {
        if (!active || !vertices.length) return;
        undone.push(vertices.pop());
        onError('');
        render();
        onPoints();
    }

    function redoPoint() {
        if (!active || !undone.length) return;
        vertices.push(undone.pop());
        onError('');
        render();
        onPoints();
    }

    function key(e) {
        if (isBuilding() && e.target === canvas && (e.key === 'Backspace' || e.key === 'Delete')) {
            e.preventDefault();
            e.stopImmediatePropagation();
        }
        if (!active) return;
        if (e.key === 'Shift' && pointer) {
            pointer.shift = e.type === 'keydown';
            render();
        }
        if (e.type !== 'keydown') return;
        const key = e.key.toLowerCase();
        if ((e.metaKey || e.ctrlKey) && !e.altKey && (key === 'z' || key === 'y')) {
            e.preventDefault();
            e.stopImmediatePropagation();
            if (key === 'y' || e.shiftKey) redoPoint();
            else undoPoint();
            return;
        }
        if (e.key === 'Escape' || e.key === 'Backspace') {
            e.preventDefault();
            e.stopImmediatePropagation();
            if (e.key === 'Escape') setActive(false);
            else undoPoint();
        }
    }

    const leave = () => {
        pointer = null;
        render();
    };
    const unsubscribe = view.onChange(render);
    canvas.addEventListener('pointerdown', down, true);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerleave', leave);
    window.addEventListener('keydown', key, true);
    window.addEventListener('keyup', key, true);
    return {
        get active() { return active; },
        get canUndo() { return active && vertices.length > 0; },
        get canRedo() { return active && undone.length > 0; },
        undo: undoPoint,
        redo: redoPoint,
        start() { setActive(true); },
        cancel() { setActive(false); },
        dispose() {
            unsubscribe();
            canvas.removeEventListener('pointerdown', down, true);
            canvas.removeEventListener('pointermove', move);
            canvas.removeEventListener('pointerleave', leave);
            window.removeEventListener('keydown', key, true);
            window.removeEventListener('keyup', key, true);
        },
    };
}
