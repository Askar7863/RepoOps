import type { ComplexityInfo } from "@/lib/api/types";

export function ComplexitySection({ complexityInfo }: { complexityInfo: ComplexityInfo }) {
  if (!complexityInfo.available) {
    return <p className="text-caption text-muted-foreground">{complexityInfo.reason}</p>;
  }

  return (
    <div className="space-y-3 text-caption">
      <div className="flex gap-6">
        <span className="text-muted-foreground">
          Functions: <span className="text-foreground">{complexityInfo.function_count}</span>
        </span>
        <span className="text-muted-foreground">
          Avg. complexity: <span className="text-foreground">{complexityInfo.avg_complexity.toFixed(1)}</span>
        </span>
        <span className="text-muted-foreground">
          High-complexity: <span className="text-foreground">{complexityInfo.high_complexity_count}</span>
        </span>
      </div>

      {complexityInfo.high_complexity_sample.length > 0 && (
        <ul className="space-y-0.5 text-muted-foreground">
          {complexityInfo.high_complexity_sample.map(([path, name, score], i) => (
            <li key={i}>
              <span className="text-foreground">{name}</span> -- {path} (score {score})
            </li>
          ))}
        </ul>
      )}

      <p className="text-muted-foreground">Approximate, keyword-based complexity estimate -- not real cyclomatic/AST complexity.</p>
    </div>
  );
}
