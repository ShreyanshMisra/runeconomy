import { readFile, writeFile, mkdir, rm, cp } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const siteDir = dirname(fileURLToPath(import.meta.url));
const root = resolve(siteDir, "..");
const output = join(siteDir, "dist");
const repository = "https://github.com/ShreyanshMisra/runeconomy";

const pages = [
  { route: "", source: "site/content/index.md", title: "RunEconomy" },
  { route: "getting-started", source: "docs/getting-started.md", title: "Getting started" },
  { route: "how-it-works", source: "docs/how-it-works.md", title: "How it works" },
  { route: "proof", source: "docs/public-proof.md", title: "The cache optimization that looked faster—and why RunEconomy rejected it" },
  { route: "doctor", source: "docs/doctor.md", title: "vLLM compatibility" },
  { route: "cli", source: "docs/cli-reference.md", title: "CLI reference" },
  { route: "experiments", source: "docs/experiment-format.md", title: "Static experiments" },
  { route: "compare", source: "docs/closed-loop-comparison.md", title: "Closed-loop comparison" },
  { route: "traces", source: "docs/trace-format.md", title: "Trace format" },
  { route: "skill", source: "docs/agent-skill.md", title: "Agent skill" },
  { route: "troubleshooting", source: "docs/troubleshooting.md", title: "Troubleshooting" },
  { route: "alpha", source: "docs/alpha-release.md", title: "Alpha boundaries" },
];

const routeBySource = new Map(pages.map((page) => [page.source.replace(/^docs\//, ""), `/${page.route}/`]));
routeBySource.set("public-proof.md", "/proof/");
routeBySource.set("doctor.md", "/doctor/");
routeBySource.set("experiment-format.md", "/experiments/");
routeBySource.set("closed-loop-comparison.md", "/compare/");
routeBySource.set("trace-format.md", "/traces/");

function rewriteLinks(markdown) {
  return markdown.replace(/\]\(([^)#]+\.md)(#[^)]+)?\)/g, (match, source, hash = "") => {
    const target = routeBySource.get(source.replace(/^\.\//, ""));
    return target ? `](${target}${hash})` : match;
  });
}

function layout(page, content) {
  const canonical = page.route ? `/${page.route}/` : "/";
  const sourceUrl = `${repository}/blob/main/${page.source}`;
  const skillDownload = page.route === "skill"
    ? `<p class="download"><a href="/runeconomy-skill.md" download>Download the canonical SKILL.md</a> · <a href="${repository}/tree/main/skills/runeconomy">View skill source</a></p>`
    : "";
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="RunEconomy verifies whether inference optimizations improve successful agent workflows.">
  <title>${page.title} · RunEconomy</title>
  <link rel="canonical" href="${canonical}">
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/assets/site.css">
</head>
<body>
  <header class="site-header">
    <a class="wordmark" href="/">RunEconomy</a>
    <nav aria-label="Primary">
      <a href="/getting-started/">Start</a>
      <a href="/proof/">Proof</a>
      <a href="/doctor/">Doctor</a>
      <a href="/skill/">Agent skill</a>
      <a href="${repository}">GitHub</a>
    </nav>
  </header>
  <main>
    <article>${skillDownload}${content}</article>
    <aside class="source-note">Repository source: <a href="${sourceUrl}">${page.source}</a></aside>
  </main>
  <footer>Alpha software. Inspect evidence and quality gates before acting on performance measurements.</footer>
</body>
</html>`;
}

await rm(output, { recursive: true, force: true });
await mkdir(join(output, "assets"), { recursive: true });
await cp(join(siteDir, "site.css"), join(output, "assets", "site.css"));
await cp(join(siteDir, "favicon.svg"), join(output, "favicon.svg"));
await cp(join(root, "skills", "runeconomy", "SKILL.md"), join(output, "runeconomy-skill.md"));

for (const page of pages) {
  const markdown = await readFile(join(root, page.source), "utf8");
  const content = await marked.parse(rewriteLinks(markdown), { gfm: true });
  const directory = page.route ? join(output, page.route) : output;
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, "index.html"), layout(page, content));
}

console.log(`Built ${pages.length} static pages in ${output}`);
