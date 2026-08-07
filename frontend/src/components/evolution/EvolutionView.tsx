import { useEffect, useMemo, useRef } from "react";
import { TrendingUp, Users, GitCommitHorizontal } from "lucide-react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { useChat } from "@/state/ChatContext";
import { useRepo } from "@/state/RepoContext";
import { useNav } from "@/state/NavContext";
import { CANNED_MESSAGES } from "@/lib/canned-messages";
import { MessageBubbleUser } from "@/components/chat/MessageBubbleUser";
import { MessageBubbleAssistant } from "@/components/chat/MessageBubbleAssistant";
import { TypingIndicator } from "@/components/chat/TypingIndicator";

const PRESETS = [
  { label: "Recent activity", icon: TrendingUp, message: CANNED_MESSAGES.activityEvolution },
  { label: "Ownership & risk", icon: Users, message: CANNED_MESSAGES.ownershipRisk },
  { label: "What changed recently", icon: GitCommitHorizontal, message: "What kind of changes have been happening recently -- bug fixes, features, or refactors?" },
];

// Structurally different from HealthView by design: there is no
// dedicated endpoint for this, so repo_evolution_tool is only reachable
// by sending a message through /api/chat that the router classifies to
// it. This view is "send a canned prompt, render prose" -- never a fake
// structured dashboard (no charts/tables the API doesn't actually return).
export function EvolutionView() {
  const chat = useChat();
  const repo = useRepo();
  const nav = useNav();
  const autoSent = useRef(false);

  // Pairs each repo_evolution_tool reply with the user question that
  // preceded it -- filtering the shared list by tool alone would leave
  // orphaned user questions with their (differently-toolUsed) answers
  // stripped out, which reads as broken rather than focused.
  const evolutionMessages = useMemo(() => {
    const pairs: typeof chat.messages = [];
    chat.messages.forEach((m, i) => {
      if (m.role === "assistant" && m.toolUsed === "repo_evolution_tool") {
        const prev = chat.messages[i - 1];
        if (prev?.role === "user" && pairs[pairs.length - 1]?.id !== prev.id) {
          pairs.push(prev);
        }
        pairs.push(m);
      }
    });
    return pairs;
  }, [chat.messages]);
  const hasEvolutionReply = evolutionMessages.length > 0;

  useEffect(() => {
    if (repo.active && !hasEvolutionReply && !autoSent.current && chat.status === "idle") {
      autoSent.current = true;
      void chat.send(CANNED_MESSAGES.activityEvolution);
    }
  }, [repo.active, hasEvolutionReply, chat, chat.status]);

  if (!repo.active) {
    return <div className="flex h-full items-center justify-center text-caption text-muted-foreground">Load a repository first.</div>;
  }

  return (
    <div className="flex h-full flex-col">
      <ScrollArea className="flex-1">
        <div className="mx-auto flex max-w-3xl flex-col gap-4 px-6 py-6">
          {evolutionMessages.length === 0 && chat.status === "idle" ? (
            <p className="text-caption text-muted-foreground">Asking about repository activity...</p>
          ) : (
            evolutionMessages.map((m) => (m.role === "user" ? <MessageBubbleUser key={m.id} message={m} /> : <MessageBubbleAssistant key={m.id} message={m} />))
          )}
          {chat.status === "sending" && <TypingIndicator />}
        </div>
      </ScrollArea>

      <div className="border-t border-border px-6 py-4">
        <div className="mx-auto flex max-w-3xl flex-wrap gap-2">
          {PRESETS.map((preset) => (
            <Button
              key={preset.label}
              variant="outline"
              size="sm"
              disabled={chat.status === "sending"}
              onClick={() => void chat.send(preset.message)}
            >
              <preset.icon className="size-3.5" aria-hidden="true" />
              {preset.label}
            </Button>
          ))}
          <Button variant="ghost" size="sm" onClick={() => nav.setView("chat")}>
            Ask something else in Chat
          </Button>
        </div>
      </div>
    </div>
  );
}
