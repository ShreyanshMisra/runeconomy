import { cp, mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { marked } from "marked";

import { articles } from "./articles.mjs";
import { siteConfig } from "./config.mjs";

const siteDir = dirname(fileURLToPath(import.meta.url));
const root = resolve(siteDir, "..");
const output = join(siteDir, "dist");

const primaryNavigation = [
  { route: "", label: "Home", key: "home" },
  { route: "start", label: "Start", key: "start" },
  { route: "doctor", label: "Doctor", key: "doctor" },
  { route: "agents", label: "Agents", key: "agents" },
];

const referencePages = [
  {
    route: "start",
    source: "docs/getting-started.md",
    title: "Start with RunEconomy",
    description: "Install RunEconomy, run the offline proof, inspect vLLM and prepare a controlled experiment.",
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
    title: "Static experiments",
    description: "Manifest structure, controls, execution and verdict semantics.",
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
    route: "troubleshooting",
    source: "docs/troubleshooting.md",
    title: "Troubleshooting",
    description: "Correct common setup, execution and evidence failures.",
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
  title: "Know if the optimization worked",
  description: "RunEconomy connects inference evidence to complete workflow latency and task success.",
  active: "home",
  kind: "home",
};

const articlePages = articles.map((article) => ({
  ...article,
  active: article.primaryPage ? article.slug : "",
  kind: article.primaryPage ? "primary" : "article",
}));

const pages = [homePage, ...referencePages, ...articlePages];
const routeBySource = new Map(pages.map((page) => [page.source.replace(/^docs\//, ""), `/${page.route}/`]));
routeBySource.set("public-proof.md", "/proof/");
routeBySource.set("doctor.md", "/doctor/");
routeBySource.set("experiment-format.md", "/experiments/");
routeBySource.set("closed-loop-comparison.md", "/compare/");
routeBySource.set("trace-format.md", "/traces/");
routeBySource.set("agent-skill.md", "/agents/");
routeBySource.set("getting-started.md", "/start/");
routeBySource.set("how-it-works.md", "/articles/how-runeconomy-works/");

function absoluteUrl(path) {
  if (!siteConfig.siteUrl) return null;
  return new URL(path, `${siteConfig.siteUrl.replace(/\/$/, "")}/`).toString();
}

function rewriteLinks(markdown) {
  return markdown.replace(/\]\(([^)#]+\.md)(#[^)]+)?\)/g, (match, source, hash = "") => {
    const target = routeBySource.get(source.replace(/^\.\//, ""));
    return target ? `](${target}${hash})` : match;
  });
}

function icon(name) {
  const paths = {
    copy: '<rect x="5" y="5" width="10" height="10" rx="1"/><path d="M9 2h8a1 1 0 0 1 1 1v8"/>',
    github: '<path d="M12 2a10 10 0 0 0-3.16 19.49c.5.09.68-.22.68-.48v-1.87c-2.78.6-3.37-1.18-3.37-1.18-.45-1.16-1.11-1.47-1.11-1.47-.91-.62.07-.61.07-.61 1 .07 1.53 1.03 1.53 1.03.9 1.53 2.34 1.09 2.91.83.09-.65.35-1.09.64-1.34-2.22-.25-4.56-1.11-4.56-4.94 0-1.09.39-1.98 1.03-2.68-.1-.25-.45-1.27.1-2.64 0 0 .84-.27 2.75 1.02A9.6 9.6 0 0 1 12 6.82a9.6 9.6 0 0 1 2.5.34c1.91-1.29 2.75-1.02 2.75-1.02.55 1.37.2 2.39.1 2.64.64.7 1.03 1.59 1.03 2.68 0 3.84-2.34 4.68-4.57 4.93.36.31.68.92.68 1.85v2.77c0 .27.18.58.69.48A10 10 0 0 0 12 2Z"/>',
    package: '<path d="m4 7 8-4 8 4-8 4-8-4Z"/><path d="m4 7 8 4 8-4v10l-8 4-8-4V7Z"/><path d="M12 11v10"/>',
    download: '<path d="M12 3v12m0 0 5-5m-5 5-5-5"/><path d="M4 20h16"/>',
    x: '<path d="M5 4l14 16M19 4 5 20"/>',
    linkedin: '<rect x="4" y="9" width="4" height="11"/><path d="M6 4v.1M12 20v-6a5 5 0 0 1 8 0v6M12 9v11"/>',
    arrow: '<path d="M5 12h14m-5-5 5 5-5 5"/>',
  };
  return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">${paths[name]}</svg>`;
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
      <span>${siteConfig.upcomingVersion} upcoming</span>
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
  const cards = articles.filter((article) => article.rail).map((article) => {
    const marker = article.dateLabel || article.status;
    return `<article class="rail-card">
      <a href="/${article.route}/" aria-label="${article.title}">
        <img src="${article.thumbnail}" alt="${article.thumbnailAlt}" width="640" height="400">
        <div class="rail-card-body">
          <div class="rail-card-meta"><span>${article.type}</span><span>${marker}</span></div>
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
      <p>Production PyPI currently serves stable ${siteConfig.stableVersion}. The upcoming alpha remains local until it is published.</p>
    </div>
    <div class="install-options">
      <div class="install-command">
        <span class="command-label">Stable ${siteConfig.stableVersion}</span>
        <code>pip install runeconomy</code>
        <button type="button" class="copy-button" data-copy data-copy-text="pip install runeconomy" aria-label="Copy stable install command">${icon("copy")}<span data-copy-label aria-live="polite">Copy</span></button>
      </div>
      <div class="install-command install-command-upcoming">
        <span class="command-label">Alpha ${siteConfig.upcomingVersion}, upcoming</span>
        <code>python -m pip install --pre runeconomy==${siteConfig.upcomingVersion}</code>
        <button type="button" class="copy-button" data-copy data-copy-text="python -m pip install --pre runeconomy==${siteConfig.upcomingVersion}" aria-label="Copy upcoming alpha install command">${icon("copy")}<span data-copy-label aria-live="polite">Copy</span></button>
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
      <a href="${siteConfig.owner.x}">${icon("x")}<span>X</span></a>
      <a href="${siteConfig.owner.github}">${icon("github")}<span>GitHub</span></a>
      <a href="${siteConfig.owner.linkedin}">${icon("linkedin")}<span>LinkedIn</span></a>
    </nav>
  </footer>`;
}

function agentActions() {
  return `<section class="resource-actions" aria-label="Agent skill actions">
    <div><p class="eyebrow">Canonical source</p><p>The downloaded and copied skill is produced from the repository's current <code>SKILL.md</code>.</p></div>
    <div class="action-row">
      <button type="button" class="action-button" data-copy data-copy-url="/assets/runeconomy-skill.md">${icon("copy")}<span data-copy-label aria-live="polite">Copy skill</span></button>
      <a class="action-button" href="/assets/runeconomy-skill.md" download>${icon("download")}<span>Download skill</span></a>
      <a class="action-button" href="${siteConfig.repository}/tree/main/skills/runeconomy">${icon("github")}<span>View source</span></a>
    </div>
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
  const canonical = absoluteUrl(`/${page.route ? `${page.route}/` : ""}`);
  const ogImage = page.thumbnail ? absoluteUrl(page.thumbnail) : null;
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
  ${ogImage ? `<meta property="og:image" content="${ogImage}">` : ""}
  <title>${page.title} · RunEconomy</title>
  <link rel="icon" href="/favicon.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/assets/site.css">
</head>`;
}

function layout(page, content) {
  const isHome = page.kind === "home";
  const bodyClass = isHome ? "home-page" : page.kind === "article" ? "article-page" : "documentation-page";
  const extras = isHome
    ? `${installStrip()}${articleRail()}${homeFooter()}`
    : page.route === "agents"
      ? agentActions()
      : "";
  const renderedContent = page.thumbnail
    ? content.replace("</h1>", `</h1>${articleLead(page)}`)
    : content;
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

function redirectPage(from, target) {
  const absolute = absoluteUrl(target);
  return `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="0; url=${target}">${absolute ? `<link rel="canonical" href="${absolute}">` : ""}<title>Moved · RunEconomy</title></head><body><main><h1>This page moved</h1><p><a href="${target}">Continue to the new page</a>.</p></main></body></html>`;
}

await rm(output, { recursive: true, force: true });
await mkdir(join(output, "assets", "thumbnails"), { recursive: true });
await cp(join(siteDir, "site.css"), join(output, "assets", "site.css"));
await cp(join(siteDir, "site.js"), join(output, "assets", "site.js"));
await cp(join(siteDir, "favicon.svg"), join(output, "favicon.svg"));
await cp(join(root, "skills", "runeconomy", "SKILL.md"), join(output, "assets", "runeconomy-skill.md"));
await cp(join(siteDir, "assets", "thumbnails"), join(output, "assets", "thumbnails"), { recursive: true });

for (const page of pages) {
  const markdown = await readFile(join(root, page.source), "utf8");
  const rendered = await marked.parse(rewriteLinks(markdown), { gfm: true });
  const content = enhanceCodeBlocks(rendered);
  const directory = page.route ? join(output, page.route) : output;
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, "index.html"), layout(page, content));
}

for (const [from, target] of [
  ["getting-started", "/start/"],
  ["skill", "/agents/"],
  ["how-it-works", "/articles/how-runeconomy-works/"],
]) {
  const directory = join(output, from);
  await mkdir(directory, { recursive: true });
  await writeFile(join(directory, "index.html"), redirectPage(from, target));
}

console.log(`Built ${pages.length} pages and 3 redirects in ${output}`);
