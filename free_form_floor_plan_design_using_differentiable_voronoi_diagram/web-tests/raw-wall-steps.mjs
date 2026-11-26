import { openRing, ringArea, segmentDistance } from '../web/plan/geom.js';
import { fromCells, wallEdges } from '../web/plan/graph.js';

const subtract = ([x, y], [u, v]) => [x - u, y - v];
const dot = ([x, y], [u, v]) => x * u + y * v;
const cross = ([x, y], [u, v]) => x * v - y * u;
const direction = ([x, y]) => {
    const length = Math.hypot(x, y);
    return length > 0 ? [x / length, y / length] : [0, 0];
};

// A step moves sideways between two wall runs that resume the same heading.
// Follow the full connector, even if several short wall edges make the turn.
export function measureRawSteps(walls, boundary) {
    const outline = openRing(boundary);
    const scale = Math.sqrt(ringArea(outline));
    const lengthLimit = .02 * scale;
    const onBoundary = (a, b) => outline.some((p, i) => {
        const q = outline[(i + 1) % outline.length];
        return segmentDistance(a, p, q) < 1e-8 * scale && segmentDistance(b, p, q) < 1e-8 * scale;
    });
    const steps = [];
    for (const [wall, ring] of walls.entries()) {
        const points = openRing(ring);
        const runs = points.map((a, i) => {
            const b = points[(i + 1) % points.length];
            const d = subtract(b, a);
            return { a, b, length: Math.hypot(...d), heading: direction(d), outer: onBoundary(a, b) };
        }).filter(run => run.length > 1e-10 * scale);
        for (const [i, entry] of runs.entries()) {
            if (entry.outer || entry.length < lengthLimit) continue;
            const first = runs[(i + 1) % runs.length];
            const turnIn = cross(entry.heading, first.heading);
            if (first.outer || Math.abs(turnIn) < Math.SQRT1_2) continue;
            const connector = [first];
            for (let j = 2; j < runs.length; j++) {
                const exit = runs[(i + j) % runs.length];
                if (exit.outer) break;
                const last = connector.at(-1);
                const turnOut = cross(last.heading, exit.heading);
                if (exit.length >= lengthLimit && dot(entry.heading, exit.heading) >= Math.cos(35 * Math.PI / 180)
                    && turnIn * turnOut < 0 && Math.abs(turnOut) >= Math.SQRT1_2) {
                    const heading = direction(entry.heading.map((x, axis) => x + exit.heading[axis]));
                    const shift = subtract(exit.a, entry.b);
                    const height = cross(heading, shift);
                    if (Math.abs(height) >= .005 * scale && Math.abs(dot(heading, shift)) <= .75 * Math.abs(height)
                        && connector.every(run => {
                            // Each connector edge must go mainly sideways.
                            // A returning wall run is a separate part of the room.
                            const sideways = cross(heading, run.heading);
                            return sideways * height >= 0
                                && Math.abs(sideways) >= Math.abs(dot(heading, run.heading));
                        })) {
                        steps.push({
                            wall,
                            height: Math.abs(height),
                            endpoints: [entry.b, exit.a],
                            path: [entry.a, ...connector.map(run => run.a), exit.a, exit.b],
                        });
                        break;
                    }
                }
                connector.push(exit);
            }
        }
    }
    return steps;
}

// Use the exact cell graph only for ownership. The step height comes from the
// simplified wall ring that the browser draws.
export function summarizeRawSteps(frame, boundary, roomCount) {
    const graph = fromCells(frame.cells, boundary, { roomCount, simplifyTol: 0 });
    const edges = wallEdges(graph).filter(edge => edge.rooms.length === 2).map(edge => ({
        endpoints: [edge.a, edge.b].map(id => {
            const { x, y } = graph.corners.get(id);
            return [x, y];
        }),
        pair: edge.rooms.join(','),
    }));
    const ownerAt = point => {
        // The drawn wall simplification can move an edge by up to .004.
        let best = .006;
        let owner;
        for (const edge of edges) {
            const distance = segmentDistance(point, ...edge.endpoints);
            if (distance < best) {
                best = distance;
                owner = edge.pair;
            }
        }
        return owner;
    };
    const near = (a, b) => Math.hypot(...subtract(a, b)) < .006;
    const unique = [];
    for (const step of measureRawSteps(frame.walls, boundary).sort((a, b) => b.height - a.height)) {
        const [a, b] = step.endpoints;
        if (unique.some(({ endpoints: [p, q] }) => (near(a, p) && near(b, q)) || (near(a, q) && near(b, p)))) continue;
        const owners = new Set(step.path.slice(1).map((p, i) => ownerAt(p.map((x, axis) => (x + step.path[i][axis]) / 2))));
        const unknown = owners.delete(undefined);
        const kind = owners.size > 1 ? 'junction' : unknown ? 'unmatched' : 'pair';
        unique.push({ ...step, kind, pairs: [...owners] });
    }
    const totals = steps => ({
        count: steps.length,
        max: Math.max(0, ...steps.map(step => step.height)),
        sum: steps.reduce((sum, step) => sum + step.height, 0),
    });
    return {
        all: totals(unique),
        pair: totals(unique.filter(step => step.kind === 'pair')),
        junction: totals(unique.filter(step => step.kind === 'junction')),
        unmatched: totals(unique.filter(step => step.kind === 'unmatched')),
        steps: unique,
    };
}
