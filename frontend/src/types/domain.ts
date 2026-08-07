import type { ChatSource } from "@/lib/api/types";

export type NavView = "chat" | "health" | "evolution" | "history";

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  sources?: ChatSource[];
  verified?: boolean | null;
  verification_reason?: string | null;
  toolUsed?: string;
  createdAt: number;
}

export interface HistorySession {
  id: string;
  repoLabel: string;
  startedAt: number;
  messages: ChatMessage[];
}
