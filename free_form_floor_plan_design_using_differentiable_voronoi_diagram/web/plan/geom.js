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
