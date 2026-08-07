import { createContext, useCallback, useContext, useState, type ReactNode } from "react";
import { loadRepository as apiLoadRepository, fetchRepositoryHealth } from "@/lib/api/repositories";
import { ApiError } from "@/lib/api/client";
import type { HealthResponse, RepositoryLoadResponse } from "@/lib/api/types";
import { usePersona } from "./PersonaContext";

type AsyncStatus = "idle" | "loading" | "success" | "error";

interface RepoContextValue {
  active: RepositoryLoadResponse | null;
  loadStatus: AsyncStatus;
  loadError: string | null;
  health: HealthResponse | null;
  healthStatus: AsyncStatus;
  healthError: string | null;
  loadRepository: (repoUrl: string) => Promise<void>;
  refreshHealth: () => Promise<void>;
  resetLoadStatus: () => void;
}

const RepoContext = createContext<RepoContextValue | undefined>(undefined);

export function RepoProvider({ children }: { children: ReactNode }) {
  const { persona } = usePersona();

  const [active, setActive] = useState<RepositoryLoadResponse | null>(null);
  const [loadStatus, setLoadStatus] = useState<AsyncStatus>("idle");
  const [loadError, setLoadError] = useState<string | null>(null);

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthStatus, setHealthStatus] = useState<AsyncStatus>("idle");
  const [healthError, setHealthError] = useState<string | null>(null);

  const refreshHealth = useCallback(async () => {
    setHealthStatus("loading");
    setHealthError(null);
    try {
      const res = await fetchRepositoryHealth({ persona });
      setHealth(res);
      setHealthStatus("success");
    } catch (err) {
      // A 400 here just means no repo is loaded -- the panel/view consuming
      // this degrades gracefully rather than treating it as a hard failure.
      setHealth(null);
      setHealthStatus("error");
      setHealthError(err instanceof ApiError ? err.message : "Could not run the health check.");
    }
  }, [persona]);

  const loadRepository = useCallback(
    async (repoUrl: string) => {
      setLoadStatus("loading");
      setLoadError(null);
      setHealth(null);
      setHealthStatus("idle");
      setHealthError(null);

      try {
        const res = await apiLoadRepository({ repo_url: repoUrl });
        setActive(res);
        setLoadStatus("success");
        // Fire-and-forget from the caller's perspective, but tracked in
        // this same context -- populates RepositorySummaryCard without a
        // manual click, per the plan (section 3.5).
        void refreshHealth();
      } catch (err) {
        setLoadStatus("error");
        setLoadError(err instanceof ApiError ? err.message : "Could not process repository.");
      }
    },
    [refreshHealth],
  );

  const resetLoadStatus = useCallback(() => {
    setLoadStatus((prev) => (prev === "loading" ? prev : "idle"));
  }, []);

  return (
    <RepoContext.Provider
      value={{
        active,
        loadStatus,
        loadError,
        health,
        healthStatus,
        healthError,
        loadRepository,
        refreshHealth,
        resetLoadStatus,
      }}
    >
      {children}
    </RepoContext.Provider>
  );
}

export function useRepo(): RepoContextValue {
  const ctx = useContext(RepoContext);
  if (!ctx) throw new Error("useRepo must be used within a RepoProvider");
  return ctx;
}
