import type { CodeHealth } from "@/lib/api/types";

export function LanguagesSection({ codeHealth }: { codeHealth: CodeHealth }) {
  if (!codeHealth.available) {
    return <p className="text-caption text-muted-foreground">{codeHealth.reason}</p>;
  }

  if (codeHealth.language_counts.length === 0) {
    return <p className="text-caption text-muted-foreground">No language data.</p>;
  }

  const max = Math.max(...codeHealth.language_counts.map(([, count]) => count));

  return (
    <div className="space-y-2">
      {codeHealth.language_counts.map(([lang, count]) => (
        <div key={lang} className="flex items-center gap-3 text-caption">
          <span className="w-24 shrink-0 truncate text-foreground">{lang}</span>
          <div className="h-2 flex-1 overflow-hidden rounded-full bg-secondary">
            <div className="h-full rounded-full bg-primary" style={{ width: `${(count / max) * 100}%` }} />
          </div>
          <span className="w-10 shrink-0 text-right text-muted-foreground">{count}</span>
        </div>
      ))}
    </div>
  );
}
