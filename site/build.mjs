import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { dirname, join, posix, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { marked } from "marked";

import { articles } from "./articles.mjs";
import { siteConfig } from "./config.mjs";
import { fetchReleases, formatReleaseDate } from "./releases.mjs";

const siteDir = dirname(fileURLToPath(import.meta.url));
const root = resolve(siteDir, "..");
const output = join(siteDir, "dist");
const skillDirectory = join(root, "skills", "runeconomy");
const skillSource = await readFile(join(skillDirectory, "SKILL.md"), "utf8");
const skillChecksum = createHash("sha256").update(skillSource).digest("hex");
const skillInstallCommand = "npx skills add ShreyanshMisra/runeconomy --skill runeconomy --all";
const skillBootstrapPrompt = `Install the official RunEconomy skill for this project by running:
${skillInstallCommand}

Review the installed files, read the RunEconomy SKILL.md completely, and follow it for this task. Begin read-only. Inspect the repository and available inference evidence. Locate the real agent entry point and existing tracing, then explain the smallest integration before editing. Use from runeconomy.agent import Attempt in a runner-only wrapper or guard it with RUNECONOMY_STATUS_PATH in a shared entry point. Preserve normal execution, exit behavior, and tracing. Map only failure classes the repository can establish truthfully, and run a free compatibility trial with a local stub covering both entry-point paths before any benchmark. Initialize an agent-process workspace, run experiment validate, then run the read-only experiment plan check. Do not start paid compute, change a backend, consume held-out data, publish, or deploy without my explicit approval. Report what still requires my approval.`;

const primaryNavigation = [
  { route: "", label: "Home", key: "home" },
  { route: "start", label: "Start", key: "start" },
  { route: "doctor", label: "Doctor", key: "doctor" },
  { route: "agents", label: "Agents", key: "agents" },
];

const referencePages = [
  {
    route: "proofs",
    source: "docs/proofs-and-studies.md",
    title: "Workflow latency and task success in Proofs 3 and 4",
    description: "Frozen experiments comparing online FP8 with BF16 and testing the effect of removing a search tool.",
    active: "",
    kind: "reference",
  },
  {
    route: "start",
    source: "docs/getting-started.md",
    title: "Start with RunEconomy",
    description: "Install RunEconomy, run the offline proof, inspect vLLM and prepare a controlled experiment.",
    active: "start",
    kind: "guide",
  },
  {
    route: "proxy",
    source: "docs/proxy.md",
    title: "Capture inference through the local proxy",
    description: "Record OpenAI-compatible exchanges and prepare a controlled replay.",
    active: "start",
    kind: "guide",
  },
  {
    route: "cli",
    source: "docs/cli-reference.md",
    title: "CLI reference",
    description: "Commands, exit behavior and error contracts.",
    active: "start",
    kind: "reference",
  },
  {
    route: "experiments",
    source: "docs/experiment-format.md",
    title: "Experiment contract",
    description: "Workspaces, modes, validation, controls, execution and verdict semantics.",
    active: "start",
    kind: "reference",
  },
  {
    route: "compare",
    source: "docs/closed-loop-comparison.md",
    title: "Closed-loop comparison",
    description: "Run complete baseline and candidate agent processes with durable evidence.",
    active: "start",
    kind: "reference",
  },
  {
    route: "traces",
    source: "docs/trace-format.md",
    title: "Trace format",
    description: "Normalized requests, spans, dependencies and evidence quality.",
    active: "start",
    kind: "reference",
  },
  {
    route: "evidence",
    source: "docs/evidence-formats.md",
    title: "Evidence formats",
    description: "Capability grading and privacy boundaries for supported evidence sources.",
    active: "start",
    kind: "reference",
  },
  {
    route: "connectors",
    source: "docs/connectors.md",
    title: "Connector support",
    description: "Supported connectors, evidence boundaries and deferred integrations.",
    active: "start",
    kind: "reference",
  },
  {
    route: "troubleshooting",
    source: "docs/troubleshooting.md",
    title: "Troubleshooting",
    description: "Correct common setup, execution and evidence failures.",
    active: "start",
    kind: "reference",
  },
  {
    route: "privacy",
    source: "docs/privacy.md",
    title: "Privacy and support bundles",
    description: "Create a local diagnostic archive without copying private experiment content.",
    active: "start",
    kind: "reference",
  },
  {
    route: "compatibility",
    source: "docs/compatibility.md",
    title: "Live vLLM compatibility",
    description: "A dated compatibility contract containing only combinations tested live.",
    active: "start",
    kind: "reference",
  },
  {
    route: "alpha",
    source: "docs/alpha-release.md",
    title: "Alpha boundaries",
    description: "Current capabilities, migration notes, limitations and privacy boundaries.",
    active: "start",
    kind: "reference",
  },
];

const homePage = {
  route: "",
  source: "site/content/index.md",
  title: "Verify serving changes without breaking your agent",
  description: "Verify that a serving change or model migration makes your agent cheaper or faster without breaking it.",
  active: "home",
  kind: "home",
};

const notFoundPage = {
  route: "404",
  outputFile: "404.html",
  markdown: "# Page not found\n\nThis page does not exist or has moved. [Start with RunEconomy](/start/), read the [public proof](/proof/) or return to the [home page](/).",
  title: "Page not found",
  description: "This RunEconomy page does not exist or has moved.",
  active: "",
  kind: "not-found",
};

const releases = await fetchReleases(siteConfig.repository);
const currentVersion = releases.find((release) => (
  !release.prerelease && release.version === siteConfig.currentVersion
))?.version || siteConfig.fallbackStableVersion;

// A release-notes article takes its date from the matching GitHub release when
// one exists, so a newly published version is dated without editing this site.
const resolvedArticles = articles.map((article) => {
  const date = releases.find((release) => release.version === article.releaseVersion)?.date || article.date;
  return date ? { ...article, date, dateLabel: formatReleaseDate(date) } : article;
});

const articlePages = resolvedArticles.map((article) => ({
  ...article,
  active: article.primaryPage ? article.slug : "",
  kind: article.primaryPage ? "primary" : "article",
}));

// The scoreboard is built from docs/scoreboard.md but stays unpublished until
// the owner sets RUNECONOMY_PUBLISH_SCOREBOARD=1 for the production build.
const scoreboardPages = process.env.RUNECONOMY_PUBLISH_SCOREBOARD === "1" ? [{
  route: "scoreboard",
  source: "docs/scoreboard.md",
  title: "Verified decision cards",
  description: "Frozen RunEconomy decision records, one card each, with verdicts, gates and claim boundaries.",
  active: "start",
  kind: "reference",
}] : [];

const pages = [homePage, ...referencePages, ...scoreboardPages, ...articlePages];
const routeBySource = new Map(pages.map((page) => [page.source.replace(/^docs\//, ""), `/${page.route}/`]));
routeBySource.set("public-proof.md", "/proof/");
routeBySource.set("doctor.md", "/doctor/");
routeBySource.set("experiment-format.md", "/experiments/");
routeBySource.set("closed-loop-comparison.md", "/compare/");
routeBySource.set("trace-format.md", "/traces/");
routeBySource.set("evidence-formats.md", "/evidence/");
routeBySource.set("connectors.md", "/connectors/");
routeBySource.set("privacy.md", "/privacy/");
routeBySource.set("compatibility.md", "/compatibility/");
routeBySource.set("agent-skill.md", "/agents/");
routeBySource.set("getting-started.md", "/start/");
routeBySource.set("how-it-works.md", "/articles/how-runeconomy-works/");

function absoluteUrl(path) {
  if (!siteConfig.siteUrl) return null;
  return new URL(path, `${siteConfig.siteUrl.replace(/\/$/, "")}/`).toString();
}

// Links between published documents become site routes. Every other relative
// link points at a repository file that the site does not host, so it is
// rewritten to that file on GitHub instead of resolving to a missing page.
function rewriteLinks(markdown, source) {
  const sourceDirectory = posix.dirname(source);
  return markdown.replace(/\]\(([^)\s]+)\)/g, (match, link) => {
    if (/^(?:[a-z][a-z\d+.-]*:|\/|#)/i.test(link)) return match;
    const [target, hash = ""] = link.split("#", 2);
    const anchor = hash ? `#${hash}` : "";
    const route = routeBySource.get(target.replace(/^\.\//, ""));
    if (route) return `](${route}${anchor})`;
    const repositoryPath = posix.normalize(posix.join(sourceDirectory, target));
    if (repositoryPath.startsWith("..")) return match;
    return `](${siteConfig.repository}/blob/main/${repositoryPath}${anchor})`;
  });
}

function icon(name) {
  const paths = {
    copy: '<rect x="5" y="5" width="10" height="10" rx="1"/><path d="M9 2h8a1 1 0 0 1 1 1v8"/>',
    github: '<path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.87c-2.78.6-3.37-1.18-3.37-1.18-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.61.07-.61 1 .07 1.53 1.03 1.53 1.03.9 1.53 2.34 1.09 2.91.83.09-.65.35-1.09.64-1.34-2.22-.25-4.56-1.11-4.56-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02A9.6 9.6 0 0 1 12 6.82a9.6 9.6 0 0 1 2.5.34c1.91-1.29 2.75-1.02 2.75-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.85v2.77c0 .27.18.58.69.48A10 10 0 0 0 12 2Z"/>',
    package: '<path d="m4 7 8-4 8 4-8 4-8-4Z"/><path d="m4 7 8 4 8-4v10l-8 4-8-4V7Z"/><path d="M12 11v10"/>',
    download: '<path d="M12 3v12m0 0 5-5m-5 5-5-5"/><path d="M4 20h16"/>',
    twitter: '<path d="M22 4.5a9.6 9.6 0 0 1-2.8 1.36 4 4 0 0 0-7 2.68v.9A9.5 9.5 0 0 1 4 5.4s-3.6 8 4.5 11.6A10.4 10.4 0 0 1 2 18.8c8.1 4.5 18 0 18-10.3a4 4 0 0 0-.07-.74A6.9 6.9 0 0 0 22 4.5Z"/>',
    linkedin: '<path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-4 0v7h-4v-7a6 6 0 0 1 6-6Z"/><rect x="2" y="9" width="4" height="12"/><circle cx="4" cy="4" r="2"/>',
    arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  };
  return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${paths[name]}</svg>`;
}

function escapeAttribute(value) {
  return value.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;");
}

function sidebar(active) {
  const links = primaryNavigation.map((item) => {
    const current = item.key === active;
    return `<a href="/${item.route ? `${item.route}/` : ""}"${current ? ' aria-current="page"' : ""}>${item.label}</a>`;
  }).join("");
  return `<aside class="sidebar">
    <a class="wordmark" href="/" aria-label="RunEconomy home"><span class="wordmark-mark">R/E</span><span>RunEconomy</span></a>
    <nav class="primary-nav" aria-label="Primary">${links}</nav>
    <div class="sidebar-secondary">
      <a href="${siteConfig.repository}">GitHub</a>
      <a href="${siteConfig.pypi}">PyPI</a>
      <span>v${siteConfig.currentVersion}</span>
    </div>
  </aside>`;
}

function mobileHeader(active) {
  const links = primaryNavigation.map((item) => {
    const current = item.key === active;
    return `<a href="/${item.route ? `${item.route}/` : ""}"${current ? ' aria-current="page"' : ""}>${item.label}</a>`;
  }).join("");
  return `<header class="mobile-header">
    <a class="mobile-wordmark" href="/">RunEconomy</a>
    <button class="menu-button" type="button" aria-expanded="false" aria-controls="mobile-menu" data-menu-button>
      <span>Menu</span><span class="menu-lines" aria-hidden="true"></span>
    </button>
    <nav id="mobile-menu" class="mobile-menu" aria-label="Primary" data-mobile-menu hidden>${links}</nav>
  </header>`;
}

function metadata(page) {
  if (!page.type && !page.status && !page.dateLabel) return "";
  const date = page.date ? `<time datetime="${page.date}">${page.dateLabel}</time>` : `<span>${page.status}</span>`;
  return `<div class="article-meta"><span>${page.type || "Documentation"}</span>${date}</div>`;
}

function enhanceCodeBlocks(html) {
  return html.replace(/<pre><code([^>]*)>([\s\S]*?)<\/code><\/pre>/g, (_match, attrs, body) => (
    `<div class="code-block"><button type="button" class="copy-button copy-button-code" data-copy data-copy-code aria-label="Copy command">${icon("copy")}<span data-copy-label aria-live="polite">Copy</span></button><pre><code${attrs}>${body}</code></pre></div>`
  ));
}

function articleRail() {
  const cards = resolvedArticles.filter((article) => article.rail).map((article) => {
    const date = article.date ? `<time datetime="${article.date}">${article.dateLabel}</time>` : "";
    return `<article class="rail-card">
      <a href="/${article.route}/" aria-label="${article.title}">
        <img src="${article.thumbnail}" alt="${article.thumbnailAlt}" width="640" height="400">
        <div class="rail-card-body">
          <div class="rail-card-meta"><span>${article.type}</span>${date}</div>
          <h3>${article.title}</h3>
          <p>${article.description}</p>
          <span class="text-link">Read article ${icon("arrow")}</span>
        </div>
      </a>
    </article>`;
  }).join("");
  return `<section class="rail-section" aria-labelledby="from-runeconomy">
    <div class="section-label-row">
      <h2 id="from-runeconomy">From RunEconomy</h2>
      <div class="rail-controls" data-rail-controls="article-rail" aria-label="Article rail controls">
        <button type="button" data-direction="previous" aria-label="Previous articles">${icon("arrow")}</button>
        <button type="button" data-direction="next" aria-label="Next articles">${icon("arrow")}</button>
      </div>
    </div>
    <div class="article-rail" id="article-rail" tabindex="0" aria-label="RunEconomy articles">${cards}</div>
  </section>`;
}

function installStrip() {
  return `<section class="install-section" aria-labelledby="install-heading">
    <div class="install-copy">
      <p class="eyebrow" id="install-heading">Install</p>
      <p>Install RunEconomy ${currentVersion}, then inspect the bundled report or start a controlled experiment.</p>
    </div>
    <div class="install-options">
      <div class="install-command">
        <span class="command-label">Current release ${currentVersion}</span>
        <code>python -m pip install runeconomy==${currentVersion}</code>
        <button type="button" class="copy-button" data-copy data-copy-text="python -m pip install runeconomy==${currentVersion}" aria-label="Copy current release install command">${icon("copy")}<span data-copy-label aria-live="polite">Copy</span></button>
      </div>
    </div>
    <div class="install-links" aria-label="Package links">
      <a href="${siteConfig.pypi}">${icon("package")}<span>PyPI</span></a>
      <a href="${siteConfig.repository}">${icon("github")}<span>GitHub</span></a>
    </div>
  </section>`;
}

function homeFooter() {
  return `<footer class="home-footer">
    <p>Maintained by Shrey</p>
    <nav aria-label="Shrey's profiles">
      <a href="${siteConfig.owner.x}">${icon("twitter")}<span>Twitter</span></a>
      <a href="${siteConfig.owner.github}">${icon("github")}<span>GitHub</span></a>
      <a href="${siteConfig.owner.linkedin}">${icon("linkedin")}<span>LinkedIn</span></a>
    </nav>
  </footer>`;
}

function agentActions() {
  return `<section class="resource-actions" aria-label="Agent skill actions">
    <div><p class="eyebrow">Canonical source</p><p>The install, download and checksum are generated from the repository's complete <code>skills/runeconomy</code> directory.</p></div>
    <div class="action-row">
      <button type="button" class="action-button" data-copy data-copy-text="${escapeAttribute(skillInstallCommand)}">${icon("copy")}<span data-copy-label aria-live="polite">Copy install command</span></button>
      <button type="button" class="action-button" data-copy data-copy-text="${escapeAttribute(skillBootstrapPrompt)}">${icon("copy")}<span data-copy-label aria-live="polite">Copy bootstrap prompt</span></button>
      <a class="action-button" href="/skills/latest/SKILL.md">${icon("arrow")}<span>View skill</span></a>
      <a class="action-button" href="/skills/latest/SKILL.md" download>${icon("download")}<span>Download skill</span></a>
      <a class="action-button" href="${siteConfig.repository}/tree/main/skills/runeconomy">${icon("github")}<span>View source</span></a>
    </div>
    <p class="skill-checksum"><a href="/skills/latest/SHA256SUMS">Verify SHA-256</a> <code>${skillChecksum}</code></p>
  </section>`;
}

function articleLead(page) {
  if (!page.thumbnail) return "";
  return `<figure class="article-lead">
    <img src="${page.thumbnail}" alt="${page.thumbnailAlt}" width="640" height="400">
    <figcaption>${page.description}</figcaption>
  </figure>`;
}

function sourceLink(page) {
  if (page.kind !== "article") return "";
  return `<p class="view-source"><a href="${siteConfig.repository}/blob/main/${page.source}">View article source</a></p>`;
}

function head(page) {
  const notFound = page.kind === "not-found";
  const canonical = notFound ? null : absoluteUrl(`/${page.route ? `${page.route}/` : ""}`);
  const socialImage = absoluteUrl(page.thumbnail || siteConfig.defaultSocialImage);
  return `<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${page.description}">
  <meta name="theme-color" content="#f4f1ea">
  <meta property="og:type" content="${page.kind === "article" ? "article" : "website"}">
  <meta property="og:site_name" content="RunEconomy">
  <meta property="og:title" content="${page.title}">
  <meta property="og:description" content="${page.description}">
  ${canonical ? `<link rel="canonical" href="${canonical}"><meta property="og:url" content="${canonical}">` : ""}
  ${notFound ? '<meta name="robots" content="noindex">' : ""}
  ${socialImage ? `<meta property="og:image" content="${socialImage}"><meta property="og:image:width" content="1280"><meta property="og:image:height" content="800">` : ""}
  <meta name="twitter:card" content="${socialImage ? "summary_large_image" : "summary"}">
  <meta name="twitter:title" content="${page.title}">
  <meta name="twitter:description" content="${page.description}">
  ${socialImage ? `<meta name="twitter:image" content="${socialImage}">` : ""}
  <title>${page.title} · RunEconomy</title>
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/assets/site.css">
</head>`;
}

function layout(page, content) {
  const isHome = page.kind === "home";
  const bodyClass = isHome ? "home-page" : page.kind === "article" ? "article-page" : "documentation-page";
  const extras = isHome
    ? `${articleRail()}${homeFooter()}`
    : page.route === "agents"
      ? agentActions()
      : "";
  let renderedContent = page.thumbnail
    ? content.replace("</h1>", `</h1>${articleLead(page)}`)
    : content;
  if (isHome) {
    if (!renderedContent.includes("<!-- INSTALL -->")) {
      throw new Error("home page is missing its install marker");
    }
    renderedContent = renderedContent.replace("<!-- INSTALL -->", installStrip());
  }
  return `<!doctype html>
<html lang="en">
${head(page)}
<body class="${bodyClass}">
  <a class="skip-link" href="#main-content">Skip to content</a>
  ${sidebar(page.active)}
  ${mobileHeader(page.active)}
  <main id="main-content" class="site-main">
    <article class="content ${page.kind === "article" ? "article-content" : ""}">
      ${metadata(page)}
      ${renderedContent}
      ${sourceLink(page)}
    </article>
    ${extras}
  </main>
  <script src="/assets/site.js" defer></script>
</body>
</html>`;
}

function redirectPage(target) {
  const absolute = absoluteUrl(target);
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="0; url=${target}">${absolute ? `<link rel="canonical" href="${absolute}">` : ""}<title>Moved · RunEconomy</title></head><body><main><h1>This page moved</h1><p><a href="${target}">Continue to the new page</a>.</p></main></body></html>`;
}

await rm(output, { recursive: true, force: true });
await mkdir(join(output, "assets", "thumbnails"), { recursive: true });
await mkdir(join(output, "skills", "latest"), { recursive: true });
await mkdir(join(output, "skills", siteConfig.currentVersion), { recursive: true });
await cp(join(siteDir, "site.css"), join(output, "assets", "site.css"));
await cp(join(siteDir, "site.js"), join(output, "assets", "site.js"));
await cp(join(siteDir, "favicon.svg"), join(output, "favicon.svg"));
await cp(join(root, "skills", "runeconomy", "SKILL.md"), join(output, "assets", "runeconomy-skill.md"));
await cp(skillDirectory, join(output, "skills", "latest"), { recursive: true });
await cp(skillDirectory, join(output, "skills", siteConfig.currentVersion), { recursive: true });
await writeFile(join(output, "skills", "latest", "SHA256SUMS"), `${skillChecksum}  SKILL.md\n`);
await writeFile(join(output, "skills", siteConfig.currentVersion, "SHA256SUMS"), `${skillChecksum}  SKILL.md\n`);
await writeFile(join(output, "llms.txt"), `# RunEconomy\n\n- Start: /start/\n- Evidence: /evidence/\n- Experiments: /experiments/\n- Connectors: /connectors/\n- Agent skill: /skills/latest/SKILL.md\n`);
const skillReferences = ["evidence-imports.md", "experiment-contract.md", "verdicts.md", "privacy.md", "troubleshooting.md"];
const fullSkill = [skillSource, ...await Promise.all(skillReferences.map(async (name) => `\n\n${await readFile(join(skillDirectory, "references", name), "utf8")}`))].join("");
await writeFile(join(output, "llms-full.txt"), fullSkill);
await cp(join(siteDir, "assets", "thumbnails"), join(output, "assets", "thumbnails"), { recursive: true });
await mkdir(join(output, "demo"), { recursive: true });
await cp(join(siteDir, "static", "demo", "index.html"), join(output, "demo", "index.html"));

for (const page of pages) {
  const markdown = await readFile(join(root, page.source), "utf8");
  const rendered = await marked.parse(rewriteLinks(markdown, page.source), { gfm: true });
  const content = enhanceCodeBlocks(rendered);
  const directory = page.route ? join(output, page.route) : output;
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, "index.html"), layout(page, content));
}

// Vercel serves a root 404.html for any path that does not match a page.
await writeFile(
  join(output, notFoundPage.outputFile),
  layout(notFoundPage, await marked.parse(notFoundPage.markdown, { gfm: true })),
);

const redirects = [
  ["getting-started", "/start/"],
  ["skill", "/agents/"],
  ["how-it-works", "/articles/how-runeconomy-works/"],
];
for (const [from, target] of redirects) {
  const directory = join(output, from);
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, "index.html"), redirectPage(target));
}

console.log(`Built ${pages.length} pages, a 404 page and ${redirects.length} redirects in ${output}`);
