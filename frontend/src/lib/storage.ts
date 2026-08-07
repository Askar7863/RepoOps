import type { ChatMessage, HistorySession } from "@/types/domain";

const PERSONA_KEY = "repoops-persona"; // same key the old app/index.html used
const HISTORY_KEY = "repoops-history";
const MAX_HISTORY_SESSIONS = 20;

export function readLocalStorage<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function writeLocalStorage<T>(key: string, value: T): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // localStorage unavailable (private browsing, quota, etc.) -- the app
    // still works, it just won't remember this across a reload.
  }
}

export { PERSONA_KEY };

export function getHistorySessions(): HistorySession[] {
  return readLocalStorage<HistorySession[]>(HISTORY_KEY, []);
}

export function archiveHistorySession(repoLabel: string, messages: ChatMessage[]): void {
  if (messages.length === 0) return;

  const sessions = getHistorySessions();
  const session: HistorySession = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    repoLabel,
    startedAt: messages[0]?.createdAt ?? Date.now(),
    messages,
  };

  const updated = [session, ...sessions].slice(0, MAX_HISTORY_SESSIONS);
  writeLocalStorage(HISTORY_KEY, updated);
}
