// Serve web/ with its own serve.py (COOP/COEP headers) on a free port, for the browser tests.
// web/ must hold a wasm-pack build in web/pkg.
import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { fileURLToPath } from "node:url";

const freePort = () =>
    new Promise((resolve) => {
        const s = createServer();
        s.listen(0, () => {
            const { port } = s.address();
            s.close(() => resolve(port));
        });
    });

/** Start the server; returns its url and the process to kill when done. */
export async function serve() {
    const port = await freePort();
    const proc = spawn("python3", [fileURLToPath(new URL("../web/serve.py", import.meta.url)), String(port)], { stdio: "ignore" });
    const url = `http://localhost:${port}/`;
    for (let i = 0; i < 50; i++) {
        try {
            if ((await fetch(url)).ok) return { url, proc };
        } catch {}
        await new Promise((r) => setTimeout(r, 100));
    }
    proc.kill();
    throw new Error("serve.py did not start");
}
