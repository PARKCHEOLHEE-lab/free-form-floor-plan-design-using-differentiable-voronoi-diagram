/**
 * Undo and redo over whole saved states. `commit` records a change as one step; `replace` moves the
 * current state without a step (for actions that are not undone on their own, such as choosing a
 * preset), so a later undo still returns to the state before the last recorded change.
 */
export class History {
    #past = [];
    #future = [];
    #current;
    constructor(state) { this.#current = state; }
    commit(state) {
        if (JSON.stringify(state) === JSON.stringify(this.#current)) return;
        this.#past.push(this.#current);
        this.#future = [];
        this.#current = state;
    }
    replace(state) { this.#current = state; }
    undo() {
        if (!this.#past.length) return null;
        this.#future.push(this.#current);
        return this.#current = this.#past.pop();
    }
    redo() {
        if (!this.#future.length) return null;
        this.#past.push(this.#current);
        return this.#current = this.#future.pop();
    }
    get canUndo() { return this.#past.length > 0; }
    get canRedo() { return this.#future.length > 0; }
}
