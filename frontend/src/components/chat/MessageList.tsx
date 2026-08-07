import { useEffect } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useChat } from "@/state/ChatContext";
import { MessageBubbleUser } from "./MessageBubbleUser";
import { MessageBubbleAssistant } from "./MessageBubbleAssistant";
import { TypingIndicator } from "./TypingIndicator";
import { EmptyState } from "./EmptyState";

export function MessageList() {
  const { messages, status, scrollTargetId, clearScrollTarget } = useChat();

  useEffect(() => {
    if (messages.length === 0) return;
    const last = messages[messages.length - 1];
    document.getElementById(`msg-${last.id}`)?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length]);

  useEffect(() => {
    if (!scrollTargetId) return;
    document.getElementById(`msg-${scrollTargetId}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    clearScrollTarget();
  }, [scrollTargetId, clearScrollTarget]);

  if (messages.length === 0) {
    return (
      <div className="h-full px-6 py-6">
        <EmptyState />
      </div>
    );
  }

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto flex max-w-3xl flex-col gap-4 px-6 py-6" role="log" aria-live="polite">
        {messages.map((m) => (m.role === "user" ? <MessageBubbleUser key={m.id} message={m} /> : <MessageBubbleAssistant key={m.id} message={m} />))}
        {status === "sending" && <TypingIndicator />}
      </div>
    </ScrollArea>
  );
}
