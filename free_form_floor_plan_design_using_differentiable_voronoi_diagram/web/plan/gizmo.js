// The view gizmo: the plan's X, Y and Z axes as seen by the camera, drawn in a small corner widget.

// Each axis end as a direction in the three.js world, where plan point (x, y) at height h is
// (x, h, -y). -Z is left out: a view from below the floor shows nothing.
const ENDS = {
    '+x': [1, 0, 0],
    '-x': [-1, 0, 0],
    '+y': [0, 0, -1],
    '-y': [0, 0, 1],
    '+z': [0, 1, 0],
};

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const COLOURS = { x: '#e5484d', y: '#30a46c', z: '#3e63dd' };
const NAMES = {
    '+x': 'View from east',
    '-x': 'View from west',
    '+y': 'View from north',
    '-y': 'View from south',
    '+z': 'View from top',
};
const RADIUS = 28; // px from the gizmo's centre to an axis end

/**
 * Fill `el`, an empty box, with the gizmo: a line for each positive axis and a button for each
 * axis end (lettered for the positive ones). update(right, up, back) turns it to a camera, and a
 * click on an end calls onPick(axis).
 */
export function createGizmo(el, { onPick }) {
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    const half = el.clientWidth / 2;
    svg.setAttribute('viewBox', `${-half} ${-half} ${2 * half} ${2 * half}`);
    svg.setAttribute('aria-hidden', 'true');
    const lines = {};
    const ends = {};
    for (const axis of Object.keys(ENDS)) {
        const colour = COLOURS[axis[1]];
        if (axis[0] === '+') {
            lines[axis] = svg.appendChild(document.createElementNS(svgNS, 'line'));
            lines[axis].setAttribute('stroke', colour);
        }
        const end = document.createElement('button');
        end.className = 'axis';
        end.dataset.axis = axis;
        end.setAttribute('aria-label', NAMES[axis]);
        end.textContent = axis[0] === '+' ? axis[1].toUpperCase() : '';
        end.style.setProperty('--axis', colour);
        end.addEventListener('click', () => onPick(axis));
        ends[axis] = end;
    }
    el.append(svg, ...Object.values(ends));
    return {
        update(right, up, back) {
            axisTips(right, up, back, RADIUS).forEach((t, i) => {
                lines[t.axis]?.setAttribute('x2', t.x);
                lines[t.axis]?.setAttribute('y2', t.y);
                ends[t.axis].style.transform = `translate(${t.x}px, ${t.y}px)`;
                ends[t.axis].style.setProperty('--z', String(i + 1)); // nearer ends on top
            });
        },
    };
}

/**
 * Where each axis end sits in the gizmo, for a camera whose world directions are `right`, `up`
 * and `back` (back points at the viewer). Returns { axis, x, y, depth } per end, with x to the
 * right and y down in pixels from the centre (at most `radius` away) and depth growing toward
 * the viewer, ordered far to near so the nearest end is drawn last.
 */
export function axisTips(right, up, back, radius) {
    return Object.entries(ENDS)
        .map(([axis, v]) => ({ axis, x: dot(v, right) * radius, y: -dot(v, up) * radius, depth: dot(v, back) }))
        .sort((a, b) => a.depth - b.depth);
}
