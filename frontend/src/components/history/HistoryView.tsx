import { useState } from "react";
import { ArrowLeft, Clock3 } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { getHistorySessions } from "@/lib/storage";
import type { HistorySession } from "@/types/domain";
import { MessageBubbleUser } from "@/components/chat/MessageBubbleUser";
import { MessageBubbleAssistant } from "@/components/chat/MessageBubbleAssistant";

function repoLabel(repoUrl: string): string {
  const match = repoUrl.match(/github\.com\/([^/]+\/[^/]+?)(?:\.git)?\/?$/i);
  return match ? match[1] : repoUrl;
}

export function HistoryView() {
  const [sessions] = useState<HistorySession[]>(() => getHistorySessions());
  const [selected, setSelected] = useState<HistorySession | null>(null);

  if (selected) {
    return (
      <div className="flex h-full flex-col">
        <div className="flex items-center gap-2 border-b border-border px-6 py-3">
          <Button variant="ghost" size="icon" onClick={() => setSelected(null)} aria-label="Back to history">
            <ArrowLeft className="size-4" aria-hidden="true" />
          </Button>
          <p className="text-body font-medium text-foreground">{repoLabel(selected.repoLabel)}</p>
          <p className="text-caption text-muted-foreground">-- viewing past conversation, read-only</p>
        </div>
        <ScrollArea className="flex-1">
          <div className="mx-auto flex max-w-3xl flex-col gap-4 px-6 py-6">
            {selected.messages.map((m) => (m.role === "user" ? <MessageBubbleUser key={m.id} message={m} /> : <MessageBubbleAssistant key={m.id} message={m} />))}
          </div>
        </ScrollArea>
      </div>
    );
  }

  if (sessions.length === 0) {
    return (
      <div className="flex h-full items-center justify-center text-caption text-muted-foreground">
        No past conversations yet -- they're saved here when you switch to a different repository.
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-3xl space-y-2 px-6 py-6">
        <p className="text-title font-semibold text-foreground">History</p>
        {sessions.map((session) => (
          <button
            key={session.id}
            type="button"
            onClick={() => setSelected(session)}
            className="flex w-full items-center justify-between rounded-lg border border-border bg-card px-4 py-3 text-left transition-all hover:-translate-y-0.5 hover:bg-accent"
          >
            <div>
              <p className="text-body text-foreground">{repoLabel(session.repoLabel)}</p>
              <p className="flex items-center gap-1 text-caption text-muted-foreground">
                <Clock3 className="size-3" aria-hidden="true" />
                {new Date(session.startedAt).toLocaleString()} -- {session.messages.length} messages
              </p>
            </div>
          </button>
        ))}
      </div>
    </ScrollArea>
  );
}
