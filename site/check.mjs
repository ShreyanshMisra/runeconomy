import { access, readFile, readdir } from "node:fs/promises";
import { join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { articles } from "./articles.mjs";
import { siteConfig } from "./config.mjs";

const siteDir = resolve(fileURLToPath(new URL(".", import.meta.url)));
const root = resolve(siteDir, "..");
const output = join(siteDir, "dist");
const failures = [];
const retiredTerm = new RegExp("negative" + "[ _-]?split|negative" + "split", "i");

async function htmlFiles(directory) {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? htmlFiles(path) : path.endsWith(".html") ? [path] : [];
  }));
  return nested.flat();
}

function visibleProse(html) {
  return html
    .replace(/<(style|script|pre|code|svg)\b[^>]*>[\s\S]*?<\/\1>/gi, " ")
    .replace(/&(?:[a-z]+|#\d+|#x[a-f\d]+);/gi, " ")
    .replace(/<[^>]+>/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function routeFor(file) {
  const value = relative(output, file).replace(/\\/g, "/").replace(/index\.html$/, "");
  return value ? `/${value}` : "/";
}

const files = await htmlFiles(output);
const routes = new Set(files.map(routeFor));
const requiredRoutes = [
  "/", "/start/", "/doctor/", "/agents/", "/proof/", "/experiments/",
  "/compare/", "/traces/", "/cli/", "/troubleshooting/", "/alpha/",
  "/articles/0-2-0a2/", "/articles/0-2-0a3/",
  "/articles/how-runeconomy-works/", "/getting-started/", "/skill/",
  "/how-it-works/",
];
for (const route of requiredRoutes) {
  if (!routes.has(route)) failures.push(`missing required route ${route}`);
}

for (const file of files) {
  const html = await readFile(file, "utf8");
  const route = routeFor(file);
  const prose = visibleProse(html);
  const h1Count = (html.match(/<h1\b/g) || []).length;
  if (h1Count !== 1) failures.push(`${route}: expected one h1, found ${h1Count}`);
  if (retiredTerm.test(html)) failures.push(`${route}: retired terminology`);
  if (prose.includes("—")) failures.push(`${route}: em dash in visible prose`);
  if (prose.includes(";")) failures.push(`${route}: semicolon in visible prose`);
  if (/\b(INVALID|VERIFIED|REGRESSION|INCONCLUSIVE)\b/.test(prose)) failures.push(`${route}: all-caps verdict`);
  if (/docs\/internal|artifacts\/|modal\.direct|\/Users\/|Authorization:\s*Bearer/i.test(html)) failures.push(`${route}: private path or endpoint marker`);
  if (/source-note|Repository source:/.test(html)) failures.push(`${route}: obsolete global source note`);
  if (/<(?:script|link|img)[^>]+(?:src|href)="https?:/i.test(html)) failures.push(`${route}: remote runtime asset`);
  if (/@import\s+url|googletagmanager|analytics/i.test(html)) failures.push(`${route}: remote font or tracker`);
  const primary = html.match(/<nav class="primary-nav"[\s\S]*?<\/nav>/)?.[0];
  if (!route.includes("getting-started") && !route.includes("skill/") && !route.includes("how-it-works")) {
    const labels = [...(primary || "").matchAll(/<a[^>]*>([^<]+)<\/a>/g)].map((match) => match[1]);
    if (labels.join("|") !== "Home|Start|Doctor|Agents") failures.push(`${route}: primary navigation differs`);
  }
  for (const match of html.matchAll(/(?:href|src)="([^"]+)"/g)) {
    const target = match[1].split("#", 1)[0];
    if (!target || target.startsWith("http") || target.startsWith("mailto:") || target.startsWith("data:")) continue;
    if (!target.startsWith("/")) continue;
    if (target.startsWith("/assets/") || target === "/favicon.svg") {
      try {
        await access(join(output, target.slice(1)));
      } catch {
        failures.push(`${route}: missing asset ${target}`);
      }
    } else if (!routes.has(target)) {
      failures.push(`${route}: missing internal route ${target}`);
    }
  }
  for (const match of html.matchAll(/<link rel="canonical" href="([^"]+)"/g)) {
    if (!/^https:\/\//.test(match[1])) failures.push(`${route}: canonical URL is not absolute`);
  }
}

const home = await readFile(join(output, "index.html"), "utf8");
for (const target of [siteConfig.repository, siteConfig.pypi, siteConfig.owner.x, siteConfig.owner.github, siteConfig.owner.linkedin]) {
  if (!home.includes(`href="${target}"`)) failures.push(`home: missing verified external target ${target}`);
}
for (const marker of ["data-menu-button", "data-copy", "article-rail", "Maintained by Shrey"]) {
  if (!home.includes(marker)) failures.push(`home: missing interaction marker ${marker}`);
}

const proof = await readFile(join(output, "proof", "index.html"), "utf8");
for (const marker of ["79.61%", "0.8308031360405823", "negative 12.55 percentage points", "+105", "9 ms lower", "62 ms lower", "Result not verified", "Verdict: Invalid", "consumed"]) {
  if (!proof.includes(marker)) failures.push(`proof: missing frozen marker ${marker}`);
}

const builtSkill = await readFile(join(output, "assets", "runeconomy-skill.md"));
const sourceSkill = await readFile(join(root, "skills", "runeconomy", "SKILL.md"));
if (!builtSkill.equals(sourceSkill)) failures.push("skill download differs from canonical repository source");

const siteCss = await readFile(join(output, "assets", "site.css"), "utf8");
for (const marker of [
  "@media (prefers-reduced-motion: reduce)",
  "scroll-snap-type: inline proximity",
  "touch-action: pan-x",
  ":focus-visible",
]) {
  if (!siteCss.includes(marker)) failures.push(`site.css: missing accessibility marker ${marker}`);
}

const release = articles.find((article) => article.slug === "release-0-2-0a2");
if (release?.date !== "2026-11-14" || release?.dateLabel !== "November 14, 2026") {
  failures.push("0.2.0a2 release date differs from the GitHub release");
}
const upcoming = articles.find((article) => article.slug === "release-0-2-0a3");
if (upcoming?.date || upcoming?.status !== "Upcoming, not published") {
  failures.push("0.2.0a3 must remain undated and upcoming");
}

for (const [route, target] of [["getting-started", "/start/"], ["skill", "/agents/"], ["how-it-works", "/articles/how-runeconomy-works/"]]) {
  const html = await readFile(join(output, route, "index.html"), "utf8");
  if (!html.includes(`url=${target}`) || !html.includes(`href="${target}"`)) failures.push(`${route}: redirect target mismatch`);
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`Checked ${files.length} pages, routes, links, prose, assets and privacy boundaries`);
