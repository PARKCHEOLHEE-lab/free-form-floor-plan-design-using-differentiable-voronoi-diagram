import {test} from 'node:test';
import assert from 'node:assert/strict';
import {History} from '../web/plan/history.js';

test('history: undo returns the state before each change and redo replays it', () => {
    const h = new History({rooms: 0});
    assert.equal(h.canUndo, false);
    h.commit({rooms: 1});
    h.commit({rooms: 2});
    assert.deepEqual(h.undo(), {rooms: 1});
    assert.deepEqual(h.undo(), {rooms: 0});
    assert.equal(h.undo(), null, 'nothing left to undo');
    assert.deepEqual(h.redo(), {rooms: 1});
    assert.equal(h.canRedo, true);
    h.commit({rooms: 5});
    assert.equal(h.canRedo, false, 'a new change drops the undone future');
    assert.deepEqual(h.undo(), {rooms: 1});
});

test('history: an unchanged state is no step, and replace moves the current state without one', () => {
    const h = new History({mode: 'a'});
    h.commit({mode: 'a'});
    assert.equal(h.canUndo, false, 'committing the same state adds no step');
    h.commit({mode: 'a', rooms: 1});
    h.replace({mode: 'b', rooms: 1});
    assert.deepEqual(h.undo(), {mode: 'a'}, 'undo goes back past the replaced state');
    assert.deepEqual(h.redo(), {mode: 'b', rooms: 1}, 'redo returns the replaced state');
});
