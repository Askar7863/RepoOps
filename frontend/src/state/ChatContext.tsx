import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { postChat } from "@/lib/api/chat";
import { ApiError } from "@/lib/api/client";
import type { ChatMessage } from "@/types/domain";
import { archiveHistorySession } from "@/lib/storage";
import { usePersona } from "./PersonaContext";
import { useRepo } from "./RepoContext";

type SendStatus = "idle" | "sending";

interface ChatContextValue {
  messages: ChatMessage[];
  status: SendStatus;
  send: (text: string) => Promise<void>;
  scrollTargetId: string | null;
  requestScrollTo: (id: string) => void;
  clearScrollTarget: () => void;
  // Lives here (not local to Composer) so QuickActions (see ContextPanel)
  // can prefill a templated message -- e.g. "Compare Branches/Tags" --
  // without a guaranteed-incomplete auto-send.
  draft: string;
  setDraft: (text: string) => void;
}

const ChatContext = createContext<ChatContextValue | undefined>(undefined);

function makeId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const { persona } = usePersona();
  const repo = useRepo();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [status, setStatus] = useState<SendStatus>("idle");
  const [scrollTargetId, setScrollTargetId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  // Mirrors the backend's own memory.reset() timing (app/agent/memory.py):
  // the moment the active repository actually changes, archive whatever
  // was said about the PREVIOUS repo into local history and start fresh.
  const messagesRef = useRef<ChatMessage[]>(messages);
  const prevRepoLabelRef = useRef<string | null>(null);

  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);

  useEffect(() => {
    const currentLabel = repo.active?.repo_url ?? null;
    const previousLabel = prevRepoLabelRef.current;

    if (previousLabel !== null && currentLabel !== previousLabel) {
      if (messagesRef.current.length > 0) {
        archiveHistorySession(previousLabel, messagesRef.current);
      }
      setMessages([]);
    }

    prevRepoLabelRef.current = currentLabel;
  }, [repo.active?.repo_url]);

  const send = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || status === "sending") return;

      const userMessage: ChatMessage = {
        id: makeId(),
        role: "user",
        text: trimmed,
        createdAt: Date.now(),
      };
      setMessages((prev) => [...prev, userMessage]);
      setStatus("sending");

      try {
        const res = await postChat({ message: trimmed, persona });
        const assistantMessage: ChatMessage = {
          id: makeId(),
          role: "assistant",
          text: res.reply,
          sources: res.sources,
          verified: res.verified,
          verification_reason: res.verification_reason,
          toolUsed: res.tool_used,
          createdAt: Date.now(),
        };
        setMessages((prev) => [...prev, assistantMessage]);
      } catch (err) {
        const message = err instanceof ApiError ? err.message : "Sorry, I couldn't reach the server. Please try again.";
        setMessages((prev) => [
          ...prev,
          { id: makeId(), role: "assistant", text: message, createdAt: Date.now() },
        ]);
      } finally {
        setStatus("idle");
      }
    },
    [persona, status],
  );

  const requestScrollTo = useCallback((id: string) => setScrollTargetId(id), []);
  const clearScrollTarget = useCallback(() => setScrollTargetId(null), []);

  return (
    <ChatContext.Provider
      value={{ messages, status, send, scrollTargetId, requestScrollTo, clearScrollTarget, draft, setDraft }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChat(): ChatContextValue {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used within a ChatProvider");
  return ctx;
}
