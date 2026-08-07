import type { HealthResponse } from "@/lib/api/types";
import { formatSize } from "@/lib/health-colors";
import { HealthHero } from "./HealthHero";
import { SubScoreMeters } from "./SubScoreMeters";
import { ActivitySparkline } from "./ActivitySparkline";

export function OverviewSection({ health }: { health: HealthResponse }) {
  return (
    <div className="space-y-4">
      <HealthHero score={health.health_score} />
      <SubScoreMeters subScores={health.sub_scores} />
      <div className="flex gap-6 text-caption text-muted-foreground">
        <span>Size: {formatSize(health.size_info.total_size_mb)}</span>
        <span>Lines: {health.size_info.total_lines.toLocaleString()}</span>
        <span>Files: {health.size_info.file_count}</span>
      </div>
      {health.git_info.available ? (
        <ActivitySparkline weekly={health.git_info.weekly_activity} />
      ) : (
        <p className="text-caption text-muted-foreground">{health.git_info.reason}</p>
      )}
    </div>
  );
}
