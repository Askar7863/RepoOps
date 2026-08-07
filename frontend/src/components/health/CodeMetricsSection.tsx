import type { CodeHealth } from "@/lib/api/types";

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <p className="text-section font-semibold text-foreground">{value}</p>
      <p className="text-caption text-muted-foreground">{label}</p>
    </div>
  );
}

export function CodeMetricsSection({ codeHealth }: { codeHealth: CodeHealth }) {
  if (!codeHealth.available) {
    return <p className="text-caption text-muted-foreground">{codeHealth.reason}</p>;
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <Stat label="Files" value={codeHealth.total_files} />
        <Stat label="Symbols" value={codeHealth.total_symbols} />
        <Stat label="TODO/FIXME/HACK" value={codeHealth.todo_fixme_hack_total} />
        <Stat label="Oversized symbols" value={codeHealth.oversized_symbol_count} />
      </div>

      {codeHealth.python_docstring_coverage_pct !== null && (
        <p className="text-caption text-muted-foreground">
          Python docstring coverage: <span className="text-foreground">{codeHealth.python_docstring_coverage_pct.toFixed(1)}%</span>{" "}
          ({codeHealth.python_symbol_count} symbols)
        </p>
      )}

      {codeHealth.oversized_symbols_sample.length > 0 && (
        <div>
          <p className="mb-1 text-caption font-medium text-muted-foreground">Oversized symbols (sample)</p>
          <ul className="space-y-0.5 text-caption text-muted-foreground">
            {codeHealth.oversized_symbols_sample.map(([path, name], i) => (
              <li key={i}>
                <span className="text-foreground">{name}</span> -- {path}
              </li>
            ))}
          </ul>
        </div>
      )}

      {codeHealth.duplicate_names_sample.length > 0 && (
        <div>
          <p className="mb-1 text-caption font-medium text-muted-foreground">Duplicate symbol names (sample)</p>
          <ul className="space-y-0.5 text-caption text-muted-foreground">
            {codeHealth.duplicate_names_sample.map(([name, kind, count], i) => (
              <li key={i}>
                <span className="text-foreground">{name}</span> ({kind}) -- {count} occurrences
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
