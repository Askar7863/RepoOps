import { useMemo } from "react";
import { useChat } from "@/state/ChatContext";

// Derived from the frontend's OWN visible message list -- not a read of
// the server's internal conversation memory (app/agent/memory.py), which
// has no API exposing it. This is the honest frontend-only analog.
export function useRecentTopics(limit = 8) {
  const { messages } = useChat();

  return useMemo(
    () =>
      messages
        .filter((m) => m.role === "user")
        .slice(-limit)
        .reverse(),
    [messages, limit],
  );
}
