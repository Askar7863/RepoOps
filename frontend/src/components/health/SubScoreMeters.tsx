import type { SubScores } from "@/lib/api/types";
import { scoreColor } from "@/lib/health-colors";

const LABELS: { key: keyof SubScores; label: string }[] = [
  { key: "documentation", label: "Documentation" },
  { key: "testing", label: "Testing" },
  { key: "security", label: "Security" },
  { key: "complexity", label: "Complexity" },
  { key: "activity", label: "Activity" },
  { key: "bus_factor", label: "Bus Factor" },
];

export function SubScoreMeters({ subScores }: { subScores: SubScores }) {
  return (
    <div className="space-y-2">
      {LABELS.map(({ key, label }) => {
        const value = subScores[key];
        return (
          <div key={key} className="grid grid-cols-[90px_1fr_34px] items-center gap-2 text-caption">
            <span className="text-muted-foreground">{label}</span>
            <div className="h-1.5 overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${value ?? 0}%`, backgroundColor: scoreColor(value) }}
              />
            </div>
            <span className="text-right text-foreground">{value === null ? "N/A" : Math.round(value)}</span>
          </div>
        );
      })}
    </div>
  );
}
