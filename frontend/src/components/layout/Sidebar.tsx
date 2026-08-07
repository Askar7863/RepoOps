import { Box, MessageSquare, HeartPulse, TrendingUp, Clock3 } from "lucide-react";
import { useNav } from "@/state/NavContext";
import type { NavView } from "@/types/domain";
import { SidebarNavItem } from "./SidebarNavItem";
import { RepoStatusCard } from "./RepoStatusCard";

const NAV_ITEMS: { view: NavView; label: string; icon: typeof MessageSquare }[] = [
  { view: "chat", label: "Chat", icon: MessageSquare },
  { view: "health", label: "Health Analysis", icon: HeartPulse },
  { view: "evolution", label: "Evolution", icon: TrendingUp },
  { view: "history", label: "History", icon: Clock3 },
];

export function Sidebar() {
  const { view, setView } = useNav();

  return (
    <nav aria-label="Main" className="flex h-full w-64 flex-col justify-between border-r border-border bg-sidebar p-4">
      <div>
        <div className="mb-6 flex items-center gap-2 px-1">
          <Box className="size-6 text-primary" aria-hidden="true" />
          <div>
            <p className="text-body font-semibold leading-tight text-foreground">RepoOps</p>
            <p className="text-caption leading-tight text-muted-foreground">AI Repository Assistant</p>
          </div>
        </div>

        <div className="space-y-1">
          {NAV_ITEMS.map((item) => (
            <SidebarNavItem
              key={item.view}
              icon={item.icon}
              label={item.label}
              active={view === item.view}
              onClick={() => setView(item.view)}
            />
          ))}
        </div>
      </div>

      <RepoStatusCard />
    </nav>
  );
}
