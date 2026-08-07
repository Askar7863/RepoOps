import { Skeleton } from "@/components/ui/skeleton";
import { useRepo } from "@/state/RepoContext";

function Row({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="flex items-center justify-between text-caption">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-foreground">{value}</span>
    </div>
  );
}

export function RepositorySummaryCard() {
  const repo = useRepo();

  if (!repo.active) {
    return (
      <div>
        <p className="mb-2 text-caption font-medium text-muted-foreground">Repository Summary</p>
        <p className="text-caption text-muted-foreground">No repository loaded.</p>
      </div>
    );
  }

  const codeHealth = repo.health?.code_health;
  const gitInfo = repo.health?.git_info;
  const depInfo = repo.health?.dependency_info;

  return (
    <div>
      <p className="mb-2 text-caption font-medium text-muted-foreground">Repository Summary</p>
      <div className="space-y-1.5 rounded-lg border border-border bg-card p-3">
        <Row label="Files" value={`${repo.active.filtered_files} / ${repo.active.total_files}`} />
        <Row label="Chunks" value={repo.active.total_chunks} />

        {repo.healthStatus === "loading" ? (
          <>
            <Skeleton className="h-4 w-full" />
            <Skeleton className="h-4 w-3/4" />
          </>
        ) : (
          <>
            {codeHealth?.available && codeHealth.language_counts.length > 0 && (
              <Row label="Language" value={codeHealth.language_counts[0][0]} />
            )}
            {gitInfo?.available && (
              <>
                <Row label="Commits" value={gitInfo.total_commits} />
                <Row label="Contributors" value={gitInfo.contributor_count} />
              </>
            )}
            {depInfo?.available && (
              <Row label="Dependencies" value={depInfo.total_runtime_dependencies + depInfo.total_dev_dependencies} />
            )}
            {repo.healthStatus === "error" && <p className="text-caption text-muted-foreground">Health data unavailable.</p>}
          </>
        )}
      </div>
    </div>
  );
}
