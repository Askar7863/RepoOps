import { Separator } from "@/components/ui/separator";
import { RecentTopics } from "@/components/context-panel/RecentTopics";
import { RepositorySummaryCard } from "@/components/context-panel/RepositorySummaryCard";
import { QuickActions } from "@/components/context-panel/QuickActions";

export function ContextPanel() {
  return (
    <div className="space-y-5 p-4">
      <RecentTopics />
      <Separator />
      <RepositorySummaryCard />
      <Separator />
      <QuickActions />
    </div>
  );
}
