// The plan view, drawn with three.js: a top view that keeps the 2D canvas's look, and an
// isometric 3D view with the walls lifted. One orthographic camera serves both; OrbitControls
// zooms (0.5x to 8x of the fitted plan), pans, and in 3D rotates. Plan point (x, y) at height h
// sits at world (x, h, -y), so the top view shows plan +y as screen up.
import * as THREE from 'three';
import { OrbitControls } from '../vendor/three/OrbitControls.js';
import { strokeRing, strokeTriangles } from './stroke.js';
import { signedArea2 } from './geom.js';

const MIN_ZOOM = 0.5, MAX_ZOOM = 8;
const FIT = 0.80;                               // zoom 1 fits the plan's extent into 80% of the shorter window side
const ISO_ELEVATION = Math.atan(1 / Math.SQRT2); // 35.26 degrees: the true isometric angle
const ISO_AZIMUTH = -Math.PI / 4;               // camera to the south-west, looking north-east
const TOP_POLAR = 1e-6;                         // straight down (OrbitControls keeps the polar angle above 0)
// Wall widths keep the 2D canvas's fixed reference scale, 1 unit = 0.70 * shorter side / 1.45 px
// at zoom 1, so one slider value looks equally thick on every boundary.
const WALL_UNIT = 0.70 / (1.45 * FIT);          // wall unit in plan extents
const WALL_LINE = 0.005;                        // each wall line's thickness, in wall units
const WALL_HEIGHT = 0.1;                        // 3D wall height, in plan extents
const CELL_LINE = { color: '#969696', opacity: 0.55 };
const SITE = { color: '#000000', opacity: 0.4, size: 2.8 };

/**
 * Create the view on `canvas`. draw(plan) shows one plan:
 *   outline ring               the boundary the camera fits (optional; else the rooms)
 *   rooms  [{ rings, color }]  filled even-odd, as the 2D canvas filled them
 *   walls  [ring]              wall centerlines, drawn as double-line walls
 *   lines  [ring]              faint cell outlines (optional)
 *   points [[x, y]]            Voronoi sites (optional)
 *   wall   half wall width, in the 2D canvas's wall units
 * The camera fits the plan when its outline changes, and otherwise keeps the user's view.
 * The canvas carries data-view, data-zoom, data-azimuth and data-elevation (degrees).
 */
export function createPlanView(canvas) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.setClearColor(0xffffff, 1);
    const scene = new THREE.Scene();
    const content = new THREE.Group();
    scene.add(content);
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 10);
    const controls = new OrbitControls(camera, canvas);
    Object.assign(controls, { minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM, zoomToCursor: true, screenSpacePanning: true });

    const meshMat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    const lineMat = new THREE.LineBasicMaterial({ color: CELL_LINE.color, transparent: true, opacity: CELL_LINE.opacity });
    const pointMat = new THREE.PointsMaterial({ color: SITE.color, size: SITE.size, sizeAttenuation: false, transparent: true, opacity: SITE.opacity });

    let mode = 'plan';
    let width = 1, height = 1;
    let fitted = null;                          // { cx, cy, extent } of the plan the camera is fitted to
    let lastPlan = null;

    const render = () => {
        renderer.render(scene, camera);
        const d = canvas.dataset;
        d.view = mode;
        d.zoom = String(+camera.zoom.toFixed(4));
        d.azimuth = String(+THREE.MathUtils.radToDeg(controls.getAzimuthalAngle()).toFixed(2));
        d.elevation = String(+(90 - THREE.MathUtils.radToDeg(controls.getPolarAngle())).toFixed(2));
    };
    controls.addEventListener('change', render);

    const extent = () => fitted?.extent ?? 2;
    // zoom 1 shows the fitted extent across FIT of the shorter side
    function frustum() {
        const perPx = extent() / (FIT * Math.min(width, height));
        Object.assign(camera, { left: -width / 2 * perPx, right: width / 2 * perPx, top: height / 2 * perPx, bottom: -height / 2 * perPx });
        camera.near = 0;
        camera.far = 10 * extent();
        camera.updateProjectionMatrix();
    }
    // Point the camera along this mode's view direction, keeping the target and the zoom.
    function orient() {
        const top = mode === 'plan';
        controls.enableRotate = !top;
        controls.maxPolarAngle = top ? TOP_POLAR : THREE.MathUtils.degToRad(85);
        const drag = top ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
        controls.mouseButtons = { LEFT: drag, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
        controls.touches = { ONE: top ? THREE.TOUCH.PAN : THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN };
        const phi = top ? TOP_POLAR : Math.PI / 2 - ISO_ELEVATION;
        const theta = top ? 0 : ISO_AZIMUTH;
        camera.position.copy(controls.target).add(new THREE.Vector3().setFromSphericalCoords(5 * extent(), phi, theta));
        camera.lookAt(controls.target);
        controls.update();
    }
    // The home view of the current mode: centred on the plan, zoom 1.
    function home() {
        const { cx = 0, cy = 0 } = fitted ?? {};
        controls.target.set(cx, 0, -cy);
        camera.zoom = 1;
        frustum();
        orient();
        render();
    }
    canvas.addEventListener('dblclick', home);

    function resize() {
        const r = canvas.getBoundingClientRect();
        width = Math.max(1, Math.round(r.width));
        height = Math.max(1, Math.round(r.height));
        renderer.setPixelRatio(window.devicePixelRatio || 1);
        renderer.setSize(width, height, false);
        frustum();
        render();
    }

    function setMode(next) {
        mode = next;
        if (lastPlan) build(lastPlan);
        orient();
        render();
    }

    function clear() {
        dispose();
        render();
    }

    function dispose() {
        for (const o of content.children) o.geometry.dispose();
        content.clear();
    }

    function draw(plan) {
        lastPlan = plan;
        const box = bounds(plan.outline ? [plan.outline] : plan.rooms.flatMap((r) => r.rings));
        if (box && !sameFit(fitted, box)) {
            fitted = box;
            home();
        }
        build(plan);
        render();
    }

    function build(plan) {
        dispose();
        const e = extent();
        const lift = 1e-3 * e;                  // layer gap; invisible from above, keeps the depth order
        const hw = plan.wall * WALL_UNIT * e;
        const line = WALL_LINE * WALL_UNIT * e;
        const iso = mode === 'iso';
        const wallTop = iso ? WALL_HEIGHT * e : 0;
        const tris = new TriangleBuffer();

        const fill = new THREE.Color();
        for (const room of plan.rooms) {
            fill.set(room.color);
            for (const t of fillTriangles(room.rings)) tris.add(t, 0, fill);
        }
        const bands = plan.walls.map((ring) => strokeRing(ring, hw));
        if (iso) {
            const side = new THREE.Color();
            for (const pairs of bands)
                for (let i = 0; i < pairs.length; i++) {
                    const [l0, r0] = pairs[i];
                    const [l1, r1] = pairs[(i + 1) % pairs.length];
                    // each face over a-b faces right of a->b: outward is left of the ring on the
                    // left edge and right of it on the right edge
                    tris.addWall(l1, l0, wallTop, sideShade(side, l1, l0));
                    tris.addWall(r0, r1, wallTop, sideShade(side, r0, r1));
                }
        }
        if (hw > 0) {
            const black = new THREE.Color('#111111');
            for (const pairs of bands) for (const t of strokeTriangles(pairs)) tris.add(t, wallTop + lift, black);
        }
        if (hw - line > 0) {
            const white = new THREE.Color('#ffffff');
            for (const ring of plan.walls) for (const t of strokeTriangles(strokeRing(ring, hw - line))) tris.add(t, wallTop + 2 * lift, white);
        }
        content.add(new THREE.Mesh(tris.geometry(), meshMat));

        if (plan.lines?.length) {
            const seg = [];
            for (const ring of plan.lines)
                ring.forEach(([x, y], i) => {
                    const [x1, y1] = ring[(i + 1) % ring.length];
                    seg.push(x, 3 * lift, -y, x1, 3 * lift, -y1);
                });
            content.add(new THREE.LineSegments(positions(seg), lineMat));
        }
        if (plan.points?.length) content.add(new THREE.Points(positions(plan.points.flatMap(([x, y]) => [x, 4 * lift, -y])), pointMat));
    }

    return { draw, clear, resize, setMode, home, get mode() { return mode; } };
}

/** Triangles with one colour each, gathered into one geometry. */
class TriangleBuffer {
    pos = [];
    col = [];
    /** A flat triangle of plan points at height h. */
    add(tri, h, c) {
        for (const [x, y] of tri) {
            this.pos.push(x, h, -y);
            this.col.push(c.r, c.g, c.b);
        }
    }
    /** An upright wall face over the plan segment a-b, from the floor to height h. */
    addWall(a, b, h, c) {
        const [a0, b0, a1, b1] = [[a, 0], [b, 0], [a, h], [b, h]].map(([[x, y], z]) => [x, z, -y]);
        for (const p of [a0, b0, b1, a0, b1, a1]) {
            this.pos.push(...p);
            this.col.push(c.r, c.g, c.b);
        }
    }
    geometry() {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
        g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
        return g;
    }
}

const positions = (flat) => new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));

// Wall sides have no lights to shade them, so each face gets a grey from where it faces:
// lightest to the south, darkest to the north. The face over a-b faces right of a->b.
const LIGHT = new THREE.Vector2(0.3, -1).normalize();
function sideShade(c, a, b) {
    const n = new THREE.Vector2(b[1] - a[1], a[0] - b[0]).normalize();
    const v = 0.86 + 0.1 * n.dot(LIGHT);
    return c.setRGB(v, v, v, THREE.SRGBColorSpace);
}

/**
 * Triangles that fill a room's rings even-odd. Rings wound like the largest one are outlines,
 * the others are holes, and each hole goes to the outline around its first point.
 */
function fillTriangles(rings) {
    const sized = rings.filter((r) => r.length >= 3).map((r) => ({ r, a: signedArea2(r) }));
    if (!sized.length) return [];
    const sign = Math.sign(sized.reduce((m, s) => (Math.abs(s.a) > Math.abs(m.a) ? s : m)).a);
    const outers = sized.filter((s) => Math.sign(s.a) === sign).map((s) => ({ ring: s.r, holes: [] }));
    for (const s of sized.filter((s) => Math.sign(s.a) !== sign))
        (outers.find((o) => inside(s.r[0], o.ring)) ?? outers[0]).holes.push(s.r);
    const tris = [];
    for (const { ring, holes } of outers) {
        const all = [ring, ...holes].flat();
        const v = (p) => new THREE.Vector2(p[0], p[1]);
        for (const [i, j, k] of THREE.ShapeUtils.triangulateShape(ring.map(v), holes.map((h) => h.map(v))))
            tris.push([all[i], all[j], all[k]]);
    }
    return tris;
}

/** Whether point p lies inside the ring (ray casting). */
function inside([px, py], ring) {
    let hit = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i];
        const [xj, yj] = ring[j];
        if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
}

/** Centre and extent (the longer side) of the rings' bounding box, or null when empty. */
function bounds(rings) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const ring of rings)
        for (const [x, y] of ring) {
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        }
    if (!(minX <= maxX)) return null;
    return { cx: (minX + maxX) / 2, cy: (minY + maxY) / 2, extent: Math.max(maxX - minX, maxY - minY) || 1 };
}

const sameFit = (a, b) =>
    !!a && Math.abs(a.cx - b.cx) + Math.abs(a.cy - b.cy) + Math.abs(a.extent - b.extent) <= 1e-6 * b.extent;
