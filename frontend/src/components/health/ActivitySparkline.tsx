export function ActivitySparkline({ weekly }: { weekly: [string, number][] }) {
  if (weekly.length === 0) {
    return <p className="text-caption text-muted-foreground">No weekly activity data.</p>;
  }

  const max = Math.max(...weekly.map(([, count]) => count), 1);

  return (
    <div>
      <p className="mb-1.5 text-caption text-muted-foreground">Last {weekly.length} weeks</p>
      <div className="flex h-12 items-end gap-1">
        {weekly.map(([week, count]) => (
          <div
            key={week}
            title={`${week}: ${count} commit${count === 1 ? "" : "s"}`}
            className="flex-1 rounded-sm bg-primary/70"
            style={{ height: `${Math.max((count / max) * 100, count > 0 ? 6 : 2)}%` }}
          />
        ))}
      </div>
    </div>
  );
}
