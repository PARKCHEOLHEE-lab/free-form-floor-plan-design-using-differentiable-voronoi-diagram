import { isSimpleBoundary, openRing } from './geom.js';

const SNAP_RADIUS = 10; // CSS pixels, independent of zoom.
const SNAP_ANGLE_SINE = Math.sin(5 * Math.PI / 180);
const samePoint = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]) < 1e-9;
const segments = points => points.slice(1).map((end, i) => [points[i], end]);

export function snapBoundaryPoint({ point, screen, project, vertices, reference, shift }) {
    const last = vertices.at(-1);
    const dx = last ? point[0] - last[0] : 0;
    const dy = last ? point[1] - last[1] : 0;
    const fixedAxis = shift && last ? (Math.abs(dx) >= Math.abs(dy) ? 1 : 0) : null;
    const onAxis = p => fixedAxis === null || Math.abs(p[fixedAxis] - last[fixedAxis]) < 1e-9;
    const screenDistance = p => {
        const [x, y] = project(p);
        return Math.hypot(x - screen[0], y - screen[1]);
    };

    if (vertices.length >= 3 && onAxis(vertices[0]) && screenDistance(vertices[0]) <= SNAP_RADIUS) {
        return { point: vertices[0], kind: 'close' };
    }

    const ring = openRing(reference);
    const draftEdges = segments(vertices);
    const referenceEdges = segments(ring.length ? [...ring, ring[0]] : []);
    let nearest = SNAP_RADIUS;
    let candidate = null;
    function consider(p, kind) {
        const distance = screenDistance(p);
        if (distance < nearest) {
            nearest = distance;
            candidate = { point: p, kind };
        }
    }
    for (const [points, edges] of [[vertices, draftEdges], [ring, referenceEdges]]) {
        for (const p of points) {
            if ((!last || !samePoint(p, last)) && onAxis(p)) consider(p, 'vertex');
        }
        for (const [a, b] of edges) {
            const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
            if (onAxis(mid)) consider(mid, 'midpoint');
        }
        if (candidate) return candidate;
    }

    if (fixedAxis !== null) {
        const constrained = [...point];
        constrained[fixedAxis] = last[fixedAxis];
        return { point: constrained, kind: 'axis' };
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

/** A temporary sketch in the existing camera's plan coordinates. */
export function createBoundaryEditor(canvas, view, { overlay, reference, isBuilding, onActive, onClose, onError }) {
    const placed = overlay.querySelector('polyline');
    const guide = overlay.querySelector('line');
    const start = overlay.querySelector('[data-first]');
    const marker = overlay.querySelector('[data-cursor]');
    const label = overlay.querySelector('text');
    const snapLabels = { close: 'Close', vertex: 'Vertex', midpoint: 'Midpoint', axis: 'X/Y', parallel: 'Parallel', perpendicular: 'Perpendicular' };
    let active = false;
    let vertices = [];
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
        guide.toggleAttribute('hidden', !result || !points.length);
        if (result) {
            const [x, y] = view.screenPoint(result.point);
            marker.setAttribute('cx', x);
            marker.setAttribute('cy', y);
            label.textContent = snapLabels[result.kind] ?? '';
            label.setAttribute('x', x + 12);
            label.setAttribute('y', y - 8);
            const last = points.at(-1);
            if (last) {
                guide.setAttribute('x1', last[0]);
                guide.setAttribute('y1', last[1]);
                guide.setAttribute('x2', x);
                guide.setAttribute('y2', y);
            }
        }
    }

    function setActive(next) {
        if (active === next) return;
        active = next;
        vertices = [];
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
        onError('');
        render();
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
        if (e.key === 'Escape' || e.key === 'Backspace') {
            e.preventDefault();
            e.stopImmediatePropagation();
            if (e.key === 'Escape') setActive(false);
            else {
                vertices.pop();
                onError('');
                render();
            }
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
