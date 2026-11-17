// The plan view, drawn with three.js in 3D: it opens isometric, with the plan flat on the floor
// in the 2D canvas's look (walls stay flat until a user extrudes them). One orthographic camera;
// OrbitControls zooms (0.5x to 8x of the fitted plan), pans and rotates. Plan point (x, y) at
// height h sits at world (x, h, -y), so looking straight down shows plan +y as screen up.
import * as THREE from 'three';
import { OrbitControls } from '../vendor/three/OrbitControls.js';
import { strokeRing, strokeTriangles } from './stroke.js';
import { signedArea2 } from './geom.js';
import { createGizmo } from './gizmo.js';

const MIN_ZOOM = 0.5, MAX_ZOOM = 8;
const SIDE_AZIMUTH = { '+x': 90, '-x': -90, '+y': 180, '-y': 0 }; // degrees: the camera east, west, north or south
const FIT = 0.80;                               // zoom 1 fits the plan's extent into 80% of the shorter window side
const ISO_ELEVATION = Math.atan(1 / Math.SQRT2); // 35.26 degrees: the true isometric angle
const ISO_AZIMUTH = -Math.PI / 4;               // camera to the south-west, looking north-east
const TOP_POLAR = 1e-6;                         // straight down (OrbitControls keeps the polar angle above 0)
// Wall widths keep the 2D canvas's fixed reference scale, 1 unit = 0.70 * shorter side / 1.45 px
// at zoom 1, so one slider value looks equally thick on every boundary.
const WALL_UNIT = 0.70 / (1.45 * FIT);          // wall unit in plan extents
const WALL_LINE = 0.005;                        // each wall line's thickness, in wall units
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
 * The canvas carries data-zoom, data-azimuth and data-elevation (degrees).
 * `gizmo`, an empty box, becomes the view gizmo.
 */
export function createPlanView(canvas, { gizmo } = {}) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.setClearColor(0xffffff, 1);
    const scene = new THREE.Scene();
    const content = new THREE.Group();
    scene.add(content);
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 10);
    const controls = new OrbitControls(camera, canvas);
    Object.assign(controls, { minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM, zoomToCursor: true, screenSpacePanning: true });
    controls.maxPolarAngle = THREE.MathUtils.degToRad(85); // the camera stays above the floor
    // mouse as in Rhino: right drag rotates, Shift (or Ctrl, Cmd) and right drag pans
    // (OrbitControls turns a rotate drag with those keys into a pan), and the left button is
    // left free for drawing and picking tools
    controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE };

    const meshMat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    const lineMat = new THREE.LineBasicMaterial({ color: CELL_LINE.color, transparent: true, opacity: CELL_LINE.opacity });
    const pointMat = new THREE.PointsMaterial({ color: SITE.color, size: SITE.size, sizeAttenuation: false, transparent: true, opacity: SITE.opacity });

    let width = 1, height = 1;
    let fitted = null;                          // { cx, cy, extent } of the plan the camera is fitted to
    let lastPlan = null;
    const axes = gizmo ? createGizmo(gizmo, { onPick: lookFrom }) : null;

    const render = () => {
        renderer.render(scene, camera);
        if (axes) {
            // the camera's right, up and back directions are the first three columns of its world matrix
            const m = camera.matrixWorld.elements;
            axes.update([m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]);
        }
        const d = canvas.dataset;
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
    // Turn the camera to look from `azimuth` (OrbitControls' angle: 0 is the south side) and
    // `polar` (0 is straight down), keeping the target and the zoom; isometric by default.
    function orient(azimuth = ISO_AZIMUTH, polar = Math.PI / 2 - ISO_ELEVATION) {
        camera.position.copy(controls.target).add(new THREE.Vector3().setFromSphericalCoords(5 * extent(), polar, azimuth));
        camera.lookAt(controls.target);
        controls.update();
    }
    // The home view: isometric, centred on the plan, zoom 1.
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

    // A gizmo click turns the view to that side, keeping the zoom: +Z looks straight down, the
    // other ends look from the east, west, north or south at the isometric height.
    function lookFrom(axis) {
        if (axis === '+z') orient(0, TOP_POLAR);
        else orient(THREE.MathUtils.degToRad(SIDE_AZIMUTH[axis]));
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
        const tris = new TriangleBuffer();

        const fill = new THREE.Color();
        for (const room of plan.rooms) {
            fill.set(room.color);
            for (const t of fillTriangles(room.rings)) tris.add(t, 0, fill);
        }
        if (hw > 0) {
            const black = new THREE.Color('#111111');
            for (const ring of plan.walls) for (const t of strokeTriangles(strokeRing(ring, hw))) tris.add(t, lift, black);
        }
        if (hw - line > 0) {
            const white = new THREE.Color('#ffffff');
            for (const ring of plan.walls) for (const t of strokeTriangles(strokeRing(ring, hw - line))) tris.add(t, 2 * lift, white);
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

    return { draw, clear, resize, home };
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
    geometry() {
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
        g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
        return g;
    }
}

const positions = (flat) => new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));

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
