// The plan view, drawn with three.js in 3D: it opens isometric, with the plan flat on the floor
// in the 2D canvas's look (walls stay flat until a user extrudes them). One orthographic camera;
// OrbitControls zooms (0.5x to 8x of the fitted plan), pans and rotates. Plan point (x, y) at
// height h sits at world (x, h, -y), so looking straight down shows plan +y as screen up.
import * as THREE from 'three';
import { OrbitControls } from '../vendor/three/OrbitControls.js';
import { strokeRing, strokeTriangles } from './stroke.js';
import { signedArea2, inside, roomTagAnchor } from './geom.js';
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
 *   rooms  [{ rings, color, tag? }]  filled even-odd, with an optional { name, share } tag
 *   walls  [ring]              wall centerlines, drawn as double-line walls
 *   lines  [ring]              faint cell outlines (optional)
 *   points [[x, y]]            Voronoi sites (optional)
 *   wall   half wall width, in the 2D canvas's wall units
 * The camera fits the plan when its outline changes, and otherwise keeps the user's view.
 * The canvas carries data-zoom, data-azimuth and data-elevation (degrees).
 * `gizmo`, an empty box, becomes the view gizmo.
 */
export function createPlanView(canvas, { gizmo, tags } = {}) {
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    renderer.setClearColor(0xffffff, 1);
    const scene = new THREE.Scene();
    const content = new THREE.Group();
    scene.add(content);
    const input = new THREE.Group();
    scene.add(input);
    const listeners = new Set();
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 10);
    const controls = new OrbitControls(camera, canvas);
    Object.assign(controls, { minZoom: MIN_ZOOM, maxZoom: MAX_ZOOM, zoomToCursor: true, screenSpacePanning: true });
    controls.maxPolarAngle = THREE.MathUtils.degToRad(85); // the camera stays above the floor
    // mouse as in Rhino: right drag rotates, Shift (or Ctrl, Cmd) and right drag pans
    // (OrbitControls turns a rotate drag with those keys into a pan), and the left button is
    // left free for drawing and picking tools; a middle drag pans as in CAD (with those keys it rotates)
    controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE };

    const meshMat = new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide });
    const lineMat = new THREE.LineBasicMaterial({ color: CELL_LINE.color, transparent: true, opacity: CELL_LINE.opacity });
    const pointMat = new THREE.PointsMaterial({ color: SITE.color, size: SITE.size, sizeAttenuation: false, transparent: true, opacity: SITE.opacity });
    const doorMat = new THREE.LineBasicMaterial({color:'#111111'});

    let width = 1, height = 1;
    let fitted = null;                          // { cx, cy, extent } of the plan the camera is fitted to
    let lastPlan = null;
    let tagAnchors = [];
    const axes = gizmo ? createGizmo(gizmo, { onPick: lookFrom, onDrag: turn }) : null;

    const render = () => {
        renderer.render(scene, camera);
        for (const [tag, anchor] of tagAnchors) {
            const [x, y] = screenPoint(anchor);
            tag.style.left = x + 'px';
            tag.style.top = y + 'px';
        }
        if (axes) {
            // the camera's right, up and back directions are the first three columns of its world matrix
            const m = camera.matrixWorld.elements;
            axes.update([m[0], m[1], m[2]], [m[4], m[5], m[6]], [m[8], m[9], m[10]]);
        }
        const d = canvas.dataset;
        d.zoom = String(+camera.zoom.toFixed(4));
        d.azimuth = String(+THREE.MathUtils.radToDeg(controls.getAzimuthalAngle()).toFixed(2));
        d.elevation = String(+(90 - THREE.MathUtils.radToDeg(controls.getPolarAngle())).toFixed(2));
        for (const fn of listeners) fn();
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

    // A drag in the gizmo rotates the view as a right drag on the canvas does (OrbitControls'
    // rate: a full turn per canvas height of pointer movement).
    function turn(dx, dy) {
        controls.rotateLeft((2 * Math.PI * dx) / canvas.clientHeight);
        controls.rotateUp((2 * Math.PI * dy) / canvas.clientHeight);
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
        tagAnchors = [];
        tags?.replaceChildren();
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
        tagAnchors = [];
        for (const room of plan.rooms) {
            fill.set(room.color);
            const triangles = fillTriangles(room.rings);
            for (const t of triangles) tris.add(t, 0, fill);
            if (tags && room.tag) {
                const anchor = roomTagAnchor(room.rings, triangles);
                if (anchor) tagAnchors.push([makeTag(room.tag), anchor]);
            }
        }
        tags?.replaceChildren(...tagAnchors.map(([tag]) => tag));
        if (hw > 0) {
            const black = new THREE.Color('#111111');
            for (const ring of plan.walls) for (const t of strokeTriangles(strokeRing(ring, hw))) tris.add(t, lift, black);
        }
        if (hw - line > 0) {
            const white = new THREE.Color('#ffffff');
            for (const ring of plan.walls) for (const t of strokeTriangles(strokeRing(ring, hw - line))) tris.add(t, 2 * lift, white);
        }
        const doorLines=[];
        for(const {a,b,normal:n} of plan.doors??[]){
            const offset=(p,d)=>[p[0]+n[0]*d,p[1]+n[1]*d];
            const gap=[offset(a,hw+line),offset(b,hw+line),offset(b,-hw-line),offset(a,-hw-line)];
            const white=new THREE.Color('#ffffff');
            tris.add([gap[0],gap[1],gap[2]],5*lift,white);
            tris.add([gap[0],gap[2],gap[3]],5*lift,white);
            const dx=b[0]-a[0],dy=b[1]-a[1],w=Math.hypot(dx,dy);
            const end=offset(a,w);
            doorLines.push(a[0],6*lift,-a[1],end[0],6*lift,-end[1]);
            for(let i=0;i<24;i++){
                for(const k of [i,i+1]){
                    const t=k*Math.PI/48;
                    doorLines.push(a[0]+dx*Math.cos(t)+n[0]*w*Math.sin(t),6*lift,-a[1]-dy*Math.cos(t)-n[1]*w*Math.sin(t));
                }
            }
        }
        canvas.dataset.doors=String(plan.doors?.length??0);
        content.add(new THREE.Mesh(tris.geometry(), meshMat));
        if(doorLines.length)content.add(new THREE.LineSegments(positions(doorLines),doorMat));

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

    function screenPoint([x,y]) {
        camera.updateMatrixWorld();
        const p=new THREE.Vector3(x,0,-y).project(camera), r=canvas.getBoundingClientRect();
        return [r.left+(p.x+1)*r.width/2,r.top+(1-p.y)*r.height/2];
    }
    function pointOnPlan(x,y) {
        const r=canvas.getBoundingClientRect(), ray=new THREE.Raycaster();
        camera.updateMatrixWorld();
        ray.setFromCamera(new THREE.Vector2((x-r.left)/r.width*2-1,1-(y-r.top)/r.height*2),camera);
        const p=ray.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0,1,0),0),new THREE.Vector3());
        return p?[p.x,-p.z]:null;
    }
    // Input is retained separately from frame geometry and never changes camera fit.
    function drawBubbles(rooms,selected,draft,links=[],selectedEdge=-1) {
        for(const o of input.children){o.geometry.dispose();o.material.dispose();}
        input.clear();
        for(const r of draft?[...rooms,draft]:rooms){
            const fill=new THREE.MeshBasicMaterial({color:r.color,transparent:true,opacity:0.28,side:THREE.DoubleSide,depthTest:false,depthWrite:false});
            const circle=new THREE.Mesh(new THREE.CircleGeometry(r.radius,64).rotateX(-Math.PI/2),fill);
            circle.position.set(r.center[0],0,-r.center[1]);circle.renderOrder=20;input.add(circle);
            const points=Array.from({length:64},(_,i)=>new THREE.Vector3(r.center[0]+r.radius*Math.cos(i*Math.PI/32),0,-r.center[1]-r.radius*Math.sin(i*Math.PI/32)));
            const line=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:r.id===selected?'#111111':'#555555',transparent:true,depthTest:false,depthWrite:false}));
            line.renderOrder=21;input.add(line);
        }
        links.forEach(([a,b],i)=>{
            const points=[rooms[a].center,rooms[b].center].map(([x,y])=>new THREE.Vector3(x,0,-y));
            const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints(points),new THREE.LineBasicMaterial({color:i===selectedEdge?'#111111':'#555555',transparent:true,depthTest:false,depthWrite:false}));
            line.renderOrder=22;input.add(line);
        });
        render();
    }
    return { draw, clear, resize, home, screenPoint, pointOnPlan, drawBubbles, onChange(fn){listeners.add(fn);return()=>listeners.delete(fn);} };
}

function makeTag({ name, share }) {
    const tag = document.createElement('div');
    const label = document.createElement('span');
    const area = document.createElement('span');
    tag.className = 'room-tag';
    label.className = 'room-tag-name';
    area.className = 'room-tag-share';
    label.textContent = name;
    area.textContent = share;
    tag.append(label, area);
    return tag;
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
