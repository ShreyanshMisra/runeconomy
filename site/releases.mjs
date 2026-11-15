// Reads published releases from GitHub so the site can show the current stable
// version and real release dates without a manual edit. The build never fails
// because of this request. It falls back to the values in the repository.
export async function fetchReleases(repository) {
  if (process.env.RUNECONOMY_SKIP_RELEASE_LOOKUP) return [];
  const slug = new URL(repository).pathname.replace(/^\/|\/$/g, "");
  try {
    const response = await fetch(`https://api.github.com/repos/${slug}/releases?per_page=30`, {
      headers: { Accept: "application/vnd.github+json" },
      signal: AbortSignal.timeout(5000),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const releases = await response.json();
    return releases
      .filter((release) => !release.draft && release.published_at)
      .map((release) => ({
        version: String(release.tag_name).replace(/^v/, ""),
        prerelease: Boolean(release.prerelease),
        date: release.published_at.slice(0, 10),
      }))
      .sort((left, right) => right.date.localeCompare(left.date));
  } catch (error) {
    console.warn(`Release lookup skipped (${error.message}). Using repository values.`);
    return [];
  }
}

export function formatReleaseDate(date) {
  return new Intl.DateTimeFormat("en-US", { dateStyle: "long", timeZone: "UTC" })
    .format(new Date(`${date}T00:00:00Z`));
}
