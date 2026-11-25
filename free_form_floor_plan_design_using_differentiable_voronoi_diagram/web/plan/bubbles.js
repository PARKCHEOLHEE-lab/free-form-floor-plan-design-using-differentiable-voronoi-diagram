import { inside, openRing, segmentDistance } from './geom.js';

/** Retained room intentions; snapshots map stable editor IDs to dense engine indices. */
export class BubbleProgram {
    #rooms = [];
    #links = [];
    #nextId = 1;
    error = '';

    constructor(boundary) { this.boundary = openRing(boundary).map(p => [...p]); }

    canPlace(center, radius, id) { return this.#valid(center, radius, id); }

    #valid(center, radius, id) {
        if (!Array.isArray(center) || center.length !== 2 || !center.every(Number.isFinite) || !Number.isFinite(radius) || radius <= 0) {
            this.error = 'Draw a circle with a positive radius.';
            return false;
        }
        if (this.#rooms.some(r => r.id !== id && r.center[0] === center[0] && r.center[1] === center[1])) {
            this.error = 'Room centers must be different.';
            return false;
        }
        const b = this.boundary;
        if (!inside(center, b) || b.some((a, i) => segmentDistance(center, a, b[(i + 1) % b.length]) < radius)) {
            this.error = 'Keep the whole circle inside the boundary.';
            return false;
        }
        this.error = '';
        return true;
    }

    add(center, radius) {
        if (!this.#valid(center, radius)) return null;
        const id = this.#nextId++;
        this.#rooms.push({ id, center: [...center], radius });
        return id;
    }

    update(id, center, radius) {
        const room = this.#rooms.find(r => r.id === id);
        if (!room) { this.error = 'Room not found.'; return false; }
        if (!this.#valid(center, radius, id)) return false;
        Object.assign(room, { center: [...center], radius });
        return true;
    }

    rename(id, name) {
        const room = this.#rooms.find(r => r.id === id);
        if (!room) { this.error = 'Room not found.'; return false; }
        const trimmed = name.trim();
        if (trimmed) room.name = trimmed;
        else delete room.name;
        this.error = '';
        return true;
    }

    remove(id) {
        const i = this.#rooms.findIndex(r => r.id === id);
        if (i < 0) { this.error = 'Room not found.'; return false; }
        this.#rooms.splice(i, 1);
        this.#links = this.#links.filter(pair => !pair.includes(id));
        this.error = '';
        return true;
    }

    connect(a, b) {
        if (a === b || ![a,b].every(id => this.#rooms.some(r => r.id === id))) {
            this.error = 'Choose two different rooms.';
            return false;
        }
        const pair = a < b ? [a,b] : [b,a];
        if (this.#links.some(p => p[0] === pair[0] && p[1] === pair[1])) {
            this.error = 'These rooms are already linked.';
            return false;
        }
        this.#links.push(pair);
        this.error = '';
        return true;
    }

    disconnect(a, b) {
        const pair = a < b ? [a,b] : [b,a];
        const i = this.#links.findIndex(p => p[0] === pair[0] && p[1] === pair[1]);
        if (i < 0) { this.error = 'Connection not found.'; return false; }
        this.#links.splice(i, 1);
        this.error = '';
        return true;
    }

    /** Plain data for undo: room IDs, ID pairs for links, and the next ID so removed IDs stay unused. */
    state() {
        return { boundary: this.boundary.map(p => [...p]), rooms: this.#rooms.map(r => ({ ...r, center: [...r.center] })), links: this.#links.map(pair => [...pair]), nextId: this.#nextId };
    }

    static fromState({ boundary, rooms, links, nextId }) {
        const program = new BubbleProgram(boundary);
        program.#rooms = rooms.map(r => ({ ...r, center: [...r.center] }));
        program.#links = links.map(pair => [...pair]);
        program.#nextId = nextId;
        return program;
    }

    snapshot() {
        const indices = new Map(this.#rooms.map((r,i) => [r.id,i]));
        const areas = this.#rooms.map(r => r.radius * r.radius);
        const total = areas.reduce((a,b) => a+b, 0);
        return {
            rooms: this.#rooms.map(r => ({ ...r, center: [...r.center] })),
            links: this.#links.map(pair => pair.map(id => indices.get(id))),
            ratios: areas.map(a => a / total),
        };
    }
}
