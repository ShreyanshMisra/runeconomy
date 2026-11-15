import { readFileSync } from "node:fs";

// The alpha version shown on the site is the package version, so a release
// only needs the version bump in pyproject.toml.
const pyproject = readFileSync(new URL("../pyproject.toml", import.meta.url), "utf8");
const packageVersion = pyproject.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
if (!packageVersion) throw new Error("pyproject.toml does not declare a project version");

export const siteConfig = {
  name: "RunEconomy",
  siteUrl: process.env.RUNECONOMY_SITE_URL || "https://runeconomy.vercel.app",
  repository: "https://github.com/ShreyanshMisra/runeconomy",
  pypi: "https://pypi.org/project/runeconomy/",
  // Used only when the GitHub release list cannot be read during the build.
  fallbackStableVersion: "0.1.0",
  alphaVersion: packageVersion,
  defaultSocialImage: "/assets/thumbnails/workflow.png",
  owner: {
    name: "Shrey",
    x: "https://x.com/shreyanshmisra6",
    github: "https://github.com/shreyanshmisra",
    linkedin: "https://www.linkedin.com/in/shreyanshmisra/",
  },
};
