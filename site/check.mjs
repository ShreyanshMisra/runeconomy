import { access, readFile, readdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const siteDir = resolve(fileURLToPath(new URL(".", import.meta.url)));
const output = join(siteDir, "dist");

async function htmlFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? htmlFiles(path) : path.endsWith(".html") ? [path] : [];
  }));
  return nested.flat();
}

const files = await htmlFiles(output);
const routes = new Set(files.map((path) => {
  const relative = path.slice(output.length).replace(/\\/g, "/").replace(/\/index\.html$/, "/");
  return relative || "/";
}));
const failures = [];

for (const file of files) {
  const html = await readFile(file, "utf8");
  for (const match of html.matchAll(/href="([^"]+)"/g)) {
    const href = match[1].split("#", 1)[0];
    if (!href || href.startsWith("http") || href.startsWith("mailto:") || href.startsWith("/assets/")) continue;
    if (href.startsWith("/") && !routes.has(href)) {
      try {
        await access(join(output, href.slice(1)));
      } catch {
        failures.push(`${file}: missing ${href}`);
      }
    }
  }
  if (/docs\/internal|artifacts\/|modal\.direct|\/Users\//.test(html)) {
    failures.push(`${file}: private path or endpoint marker`);
  }
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`Checked ${files.length} pages and internal links`);
