import { MessageSquare } from "lucide-react";
import { useRepo } from "@/state/RepoContext";

export function EmptyState() {
  const repo = useRepo();

  return (
    <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
      <MessageSquare className="size-8 text-muted-foreground" aria-hidden="true" />
      <p className="text-section font-medium text-foreground">
        {repo.active ? "Ask anything about this repository" : "Load a repository to get started"}
      </p>
      <p className="max-w-sm text-caption text-muted-foreground">
        {repo.active
          ? "Ask about its structure, health, history, or request a code change."
          : "Paste a GitHub URL above, then come back here to chat about it."}
      </p>
    </div>
  );
}
