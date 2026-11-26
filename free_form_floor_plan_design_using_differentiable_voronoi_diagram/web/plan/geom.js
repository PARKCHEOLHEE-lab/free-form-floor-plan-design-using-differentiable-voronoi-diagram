// Plain polygon helpers. Points are [x, y]; rings are open (the first point is not repeated).

/** Twice the signed area: positive for a counter-clockwise ring. */
export function signedArea2(ring) {
    let s = 0;
    for (let i = 0; i < ring.length; i++) {
        const [x0, y0] = ring[i];
        const [x1, y1] = ring[(i + 1) % ring.length];
        s += x0 * y1 - x1 * y0;
    }
    return s;
}

export function ringArea(ring) {
    return Math.abs(signedArea2(ring)) / 2;
}

/** Distance from p to segment a-b. */
export function segmentDistance(p, a, b) {
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len2 = dx * dx + dy * dy;
    const t = len2 > 0 ? Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2)) : 0;
    return Math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy);
}

/** Drop a repeated closing point, if the ring has one. */
export function openRing(ring) {
    const n = ring.length;
    return n > 1 && ring[0][0] === ring[n - 1][0] && ring[0][1] === ring[n - 1][1] ? ring.slice(0, -1) : ring;
}

/** Whether point p lies inside the ring (ray casting). Shared with the plan renderer. */
export function inside([px, py], ring) {
    let hit = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
        const [xi, yi] = ring[i];
        const [xj, yj] = ring[j];
        if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) hit = !hit;
    }
    return hit;
}

/** Use the area center, or a point in the largest filled triangle if it lies outside. */
export function roomTagAnchor(rings, triangles) {
    let area = 0, x = 0, y = 0, largestArea = 0, largestCenter = null;
    for (const triangle of triangles) {
        const weight = Math.abs(signedArea2(triangle));
        if (weight === 0) continue;
        const [a, b, c] = triangle;
        const center = [(a[0] + b[0] + c[0]) / 3, (a[1] + b[1] + c[1]) / 3];
        area += weight;
        x += center[0] * weight;
        y += center[1] * weight;
        if (weight > largestArea) {
            largestArea = weight;
            largestCenter = center;
        }
    }
    if (!largestCenter) return null;
    const center = [x / area, y / area];
    const inFill = rings.reduce((hit, ring) => hit !== inside(center, ring), false);
    return inFill ? center : largestCenter;
}

/** A finite, open ring enclosing area without crossing or retracing its edges. */
export function isSimpleBoundary(ring) {
    if (!Array.isArray(ring) || ring.length < 3 ||
        !ring.every(p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite))) return false;

    const [x, y] = ring[0];
    const local = ring.map(([px, py]) => [px - x, py - y]);
    const span = local.reduce((max, [dx, dy]) => Math.max(max, Math.abs(dx), Math.abs(dy)), 0);
    if (!(span > 0) || !Number.isFinite(span)) return false;

    const points = local.map(([dx, dy]) => [dx / span, dy / span]);
    const tolerance = 1e-9;
    if (ringArea(points) <= tolerance) return false;
    const side = (a, b, p) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    const n = points.length;

    for (let i = 0; i < n; i++) {
        const a = points[i], b = points[(i + 1) % n], next = points[(i + 2) % n];
        if (Math.hypot(a[0] - b[0], a[1] - b[1]) <= tolerance ||
            segmentDistance(next, a, b) <= tolerance || segmentDistance(a, b, next) <= tolerance) return false;

        for (let j = i + 2; j < n; j++) {
            if (i === 0 && j === n - 1) continue;
            const c = points[j], d = points[(j + 1) % n];
            if (side(a, b, c) * side(a, b, d) < 0 && side(c, d, a) * side(c, d, b) < 0) return false;
            if (segmentDistance(a, c, d) <= tolerance || segmentDistance(b, c, d) <= tolerance ||
                segmentDistance(c, a, b) <= tolerance || segmentDistance(d, a, b) <= tolerance) return false;
        }
    }
    return true;
}
