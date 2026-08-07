// Ported directly from the old app/index.html's scoreColor()/formatSize()
// -- same thresholds, same behavior, just typed.

export function scoreColor(score: number | null | undefined): string {
  if (score === null || score === undefined) return "#8b8b8b"; // dim/muted
  if (score >= 80) return "#0ca30c";
  if (score >= 60) return "#fab219";
  if (score >= 40) return "#ec835a";
  return "#d03b3b";
}

export function formatSize(mb: number): string {
  if (mb >= 1024) return `${(mb / 1024).toFixed(2)} GB`;
  return `${mb.toFixed(1)} MB`;
}
