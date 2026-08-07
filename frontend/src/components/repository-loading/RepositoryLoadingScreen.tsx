import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useRepo } from "@/state/RepoContext";
import { PipelineStageList } from "./PipelineStageList";
import { LoadResultReveal } from "./LoadResultReveal";

const SETTLE_HOLD_MS = 600; // brief pause showing all stages checked before revealing stats
const REVEAL_HOLD_MS = 1800; // how long the stats grid stays up before fading into chat

export function RepositoryLoadingScreen() {
  const repo = useRepo();
  const [phase, setPhase] = useState<"stages" | "reveal">("stages");

  const visible = repo.loadStatus !== "idle";

  useEffect(() => {
    if (repo.loadStatus === "loading") {
      setPhase("stages");
      return;
    }

    if (repo.loadStatus === "success") {
      const toReveal = setTimeout(() => setPhase("reveal"), SETTLE_HOLD_MS);
      const dismiss = setTimeout(() => repo.resetLoadStatus(), SETTLE_HOLD_MS + REVEAL_HOLD_MS);
      return () => {
        clearTimeout(toReveal);
        clearTimeout(dismiss);
      };
    }
  }, [repo.loadStatus, repo]);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
          className={cn(
            "fixed inset-0 z-50 flex items-center justify-center bg-background",
            // Once the load has actually succeeded, this is purely a
            // celebratory confirmation with nothing left to interact with --
            // it must not trap clicks meant for the real UI underneath
            // (repo.active/the Composer are already "ready" at this point,
            // even while this overlay is still fading out).
            repo.loadStatus === "success" && "pointer-events-none",
          )}
        >
          <div className="w-full max-w-md px-6">
            {repo.loadStatus === "error" ? (
              <div className="text-center">
                <AlertTriangle className="mx-auto mb-3 size-8 text-destructive" aria-hidden="true" />
                <p className="mb-2 text-section font-semibold text-foreground">Could not load repository</p>
                <p className="mb-4 text-body text-muted-foreground">{repo.loadError}</p>
                <Button onClick={() => repo.resetLoadStatus()}>Try again</Button>
              </div>
            ) : phase === "reveal" && repo.active ? (
              <LoadResultReveal result={repo.active} />
            ) : (
              <PipelineStageList settled={repo.loadStatus === "success"} />
            )}
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
