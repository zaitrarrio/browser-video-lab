// Minimal dependency-free static server for the built `dist/` directory.
//
// It exists so the production host (Railway) can serve the WebGPU app with the
// cross-origin isolation headers that threaded WASM / SharedArrayBuffer need —
// the same COOP/COEP headers Vite's dev server sets in vite.config.ts. It also
// sends the correct `application/wasm` MIME type and an SPA fallback.
import { createServer } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { join, normalize, extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../dist", import.meta.url));
const port = process.env.PORT ? Number(process.env.PORT) : 4173;
const host = process.env.HOST ?? "0.0.0.0";

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".ico": "image/x-icon",
  ".webp": "image/webp",
  ".woff2": "font/woff2",
  ".map": "application/json; charset=utf-8",
};

async function statOrNull(p) {
  try {
    return await stat(p);
  } catch {
    return null;
  }
}

const server = createServer(async (req, res) => {
  // Cross-origin isolation: required by ONNX Runtime Web's threaded WASM backend.
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
  res.setHeader("Cross-Origin-Resource-Policy", "same-origin");

  try {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    const pathname = decodeURIComponent(url.pathname);
    // Resolve within root and reject traversal outside it.
    let filePath = resolve(join(root, normalize(pathname)));
    if (filePath !== root && !filePath.startsWith(root + sep)) {
      res.statusCode = 403;
      res.end("Forbidden");
      return;
    }

    let info = await statOrNull(filePath);
    if (info?.isDirectory()) {
      filePath = join(filePath, "index.html");
      info = await statOrNull(filePath);
    }
    if (!info) {
      // Single-page-app fallback.
      filePath = join(root, "index.html");
      info = await statOrNull(filePath);
      if (!info) {
        res.statusCode = 404;
        res.end("Not found");
        return;
      }
    }

    const body = await readFile(filePath);
    res.setHeader("Content-Type", TYPES[extname(filePath)] ?? "application/octet-stream");
    // Hashed build assets are immutable; everything else must revalidate.
    res.setHeader(
      "Cache-Control",
      filePath.includes(`${sep}assets${sep}`)
        ? "public, max-age=31536000, immutable"
        : "no-cache",
    );
    res.statusCode = 200;
    res.end(body);
  } catch (err) {
    res.statusCode = 500;
    res.end("Internal Server Error");
    console.error(err);
  }
});

server.listen(port, host, () => {
  console.log(`browser-video-lab serving dist/ on http://${host}:${port} (cross-origin isolated)`);
});
