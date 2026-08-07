import { scoreColor } from "@/lib/health-colors";

export function HealthHero({ score }: { score: number | null }) {
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-title font-semibold" style={{ color: scoreColor(score) }}>
        {score === null ? "N/A" : score.toFixed(1)}
      </span>
      {score !== null && <span className="text-body text-muted-foreground">/ 100</span>}
    </div>
  );
}
