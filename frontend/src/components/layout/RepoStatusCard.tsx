import { useRepo } from "@/state/RepoContext";
import { cn } from "@/lib/utils";

function repoLabel(repoUrl: string): string {
  // "https://github.com/owner/repo" -> "owner/repo"
  const match = repoUrl.match(/github\.com\/([^/]+\/[^/]+?)(?:\.git)?\/?$/i);
  return match ? match[1] : repoUrl;
}

export function RepoStatusCard() {
  const repo = useRepo();

  if (!repo.active) {
    return (
      <div className="rounded-lg border border-border bg-card p-3">
        <div className="flex items-center gap-2">
          <span className="size-2 rounded-full bg-dim" aria-hidden="true" />
          <span className="text-caption text-muted-foreground">No repository loaded</span>
        </div>
      </div>
    );
  }

  const primaryLanguage =
    repo.health?.code_health.available && repo.health.code_health.language_counts.length > 0
      ? repo.health.code_health.language_counts[0][0]
      : null;

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="flex items-center gap-2">
        <span
          className={cn("size-2 rounded-full bg-primary", repo.loadStatus === "loading" && "animate-pulse")}
          aria-hidden="true"
        />
        <span className="truncate text-caption font-medium text-foreground" title={repoLabel(repo.active.repo_url)}>
          {repoLabel(repo.active.repo_url)}
        </span>
      </div>
      <dl className="mt-2 space-y-0.5 text-caption text-muted-foreground">
        <div className="flex justify-between">
          <dt>Files</dt>
          <dd>
            {repo.active.filtered_files} <span className="text-dim">/ {repo.active.total_files} total</span>
          </dd>
        </div>
        <div className="flex justify-between">
          <dt>Language</dt>
          <dd>{primaryLanguage ?? "--"}</dd>
        </div>
      </dl>
    </div>
  );
}
