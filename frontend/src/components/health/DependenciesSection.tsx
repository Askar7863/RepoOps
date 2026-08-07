import type { DependencyInfo } from "@/lib/api/types";

export function DependenciesSection({ dependencyInfo }: { dependencyInfo: DependencyInfo }) {
  if (!dependencyInfo.available) {
    return <p className="text-caption text-muted-foreground">{dependencyInfo.reason}</p>;
  }

  return (
    <div className="space-y-3">
      <p className="text-caption text-muted-foreground">
        <span className="text-foreground">{dependencyInfo.total_runtime_dependencies}</span> runtime,{" "}
        <span className="text-foreground">{dependencyInfo.total_dev_dependencies}</span> dev
      </p>

      {dependencyInfo.manifests_parsed.length > 0 && (
        <div className="space-y-1">
          {dependencyInfo.manifests_parsed.map((m) => (
            <div key={m.file} className="flex items-center justify-between text-caption">
              <span className="text-foreground">{m.file}</span>
              <span className="text-muted-foreground">
                {m.runtime_count} runtime / {m.dev_count} dev{m.pinned_count !== null ? ` / ${m.pinned_count} pinned` : ""}
              </span>
            </div>
          ))}
        </div>
      )}

      {dependencyInfo.manifests_unparsed.length > 0 && (
        <p className="text-caption text-muted-foreground">
          Found but not parsed: {dependencyInfo.manifests_unparsed.join(", ")}
        </p>
      )}
    </div>
  );
}
