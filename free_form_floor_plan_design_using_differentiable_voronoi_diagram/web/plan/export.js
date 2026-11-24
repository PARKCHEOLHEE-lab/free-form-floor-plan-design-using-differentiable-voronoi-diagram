// The state on screen as one JSON document: the boundary, the room program, the settings, the last
// optimizer frame and the plan (with the wall edits, or the live aligned plan). Pure, so the page
// only adds the download.
import { roomShares } from './wall-edit.js';

const FORMAT = 'voronoi-plan';
const VERSION = 1;

/** A deep copy that JSON keeps as it is (typed arrays become plain arrays). */
const plain = (v) => JSON.parse(JSON.stringify(v, (_, x) => (ArrayBuffer.isView(x) ? Array.from(x) : x)));

/**
 * `mode` and `boundary` (its ring) name the outline; `program` is RoomProgram.snapshot() or null;
 * `settings` holds the panel values as the user set them; `frame` is the last optimizer frame and
 * `finished` says whether the run ended; `plan` is the corner graph on screen and `edited` says
 * whether it holds wall edits. Rooms are named by their program id; rings list corner ids.
 */
export function exportPlan({ mode, boundary, program, settings, frame, finished, plan, edited, now }) {
    const ids = program ? program.rooms.map((r) => r.id) : null;
    return plain({
        format: FORMAT,
        version: VERSION,
        exportedAt: now.toISOString(),
        boundary: { mode, ring: boundary },
        program: program && {
            rooms: program.rooms.map((r, i) => ({ id: r.id, center: r.center, radius: r.radius, ratio: program.ratios[i] })),
            links: program.links.map((pair) => pair.map((i) => ids[i])),
        },
        settings,
        frame: { iteration: frame.iteration, finished, sites: frame.sites, cells: frame.cells },
        plan: {
            source: edited ? 'edited' : 'aligned',
            corners: [...plan.corners].sort(([a], [b]) => a - b).map(([id, c]) => ({ id, x: c.x, y: c.y, fixed: c.fixed })),
            rooms: plan.rooms.map((rings, index) => ({ index, id: ids ? ids[index] : null, rings })),
            areaShares: roomShares(plan),
        },
    });
}

const two = (n) => String(n).padStart(2, '0');

/** plan-<mode>-<YYYYMMDD-HHMMSS>.json, in local time. */
export function exportName(mode, now) {
    const day = `${now.getFullYear()}${two(now.getMonth() + 1)}${two(now.getDate())}`;
    const time = `${two(now.getHours())}${two(now.getMinutes())}${two(now.getSeconds())}`;
    return `plan-${mode}-${day}-${time}.json`;
}
