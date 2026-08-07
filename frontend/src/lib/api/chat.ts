import { apiPost } from "./client";
import type { ChatRequest, ChatResponse } from "./types";

export function postChat(req: ChatRequest): Promise<ChatResponse> {
  return apiPost<ChatRequest, ChatResponse>("/api/chat", req);
}
