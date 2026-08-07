import { apiPost } from "./client";
import type { HealthRequest, HealthResponse, RepositoryLoadRequest, RepositoryLoadResponse } from "./types";

export function loadRepository(req: RepositoryLoadRequest): Promise<RepositoryLoadResponse> {
  return apiPost<RepositoryLoadRequest, RepositoryLoadResponse>("/api/repositories/load", req);
}

export function fetchRepositoryHealth(req: HealthRequest): Promise<HealthResponse> {
  return apiPost<HealthRequest, HealthResponse>("/api/repositories/health", req);
}
