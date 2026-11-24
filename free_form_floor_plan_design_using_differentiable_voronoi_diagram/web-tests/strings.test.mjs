import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

test("details: status messages end with the ellipsis character, never three dots", () => {
    for (const file of ["../web/index.html", "../web/worker.js", "../web/plan/bubble-editor.js", "../web/plan/boundary-editor.js"]) {
        const text = readFileSync(new URL(file, import.meta.url), "utf8");
        // A word followed by three dots and a closing quote is a message, not the spread operator.
        assert.deepEqual(text.match(/[A-Za-z]\.\.\.['"`]/g) ?? [], [], `${file} writes "..." in a message`);
    }
});
