import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

// The real, known pipeline stages (app/main.py's load_repository steps 1-5,
// steps 3+4 merged into one label since "filter" has no separate visible
// checkpoint of its own). This is a SIMULATED, timed sequence, not literally
// synced to server progress -- there is no progress channel on
// /api/repositories/load (a single synchronous call). See the plan
// (section 3.6) for why that's an honest tradeoff, not an oversight.
const STAGES = ["Cloning repository", "Scanning & filtering files", "Chunking code", "Generating embeddings & indexing"];
const STAGE_INTERVAL_MS = 1800;

export function PipelineStageList({ settled }: { settled: boolean }) {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (settled) return;

    const id = setInterval(() => {
      setCurrentIndex((i) => Math.min(i + 1, STAGES.length - 1));
    }, STAGE_INTERVAL_MS);

    return () => clearInterval(id);
  }, [settled]);

  return (
    <div className="space-y-3">
      {STAGES.map((label, i) => {
        const done = settled || i < currentIndex;
        const active = !settled && i === currentIndex;

        return (
          <div key={label} className="flex items-center gap-3">
            <div
              className={cn(
                "flex size-6 shrink-0 items-center justify-center rounded-full border text-caption",
                done ? "border-primary bg-primary text-primary-foreground" : "border-border text-muted-foreground",
              )}
            >
              {done ? (
                <Check className="size-3.5" aria-hidden="true" />
              ) : active ? (
                <Loader2 className="size-3.5 animate-spin" aria-hidden="true" />
              ) : (
                i + 1
              )}
            </div>
            <span className={cn("text-body", done || active ? "text-foreground" : "text-muted-foreground")}>{label}</span>
          </div>
        );
      })}

      <div className="mt-2 h-1 overflow-hidden rounded-full bg-secondary">
        <motion.div
          className="h-full rounded-full bg-primary"
          initial={{ width: "4%" }}
          animate={{ width: settled ? "100%" : `${Math.min(((currentIndex + 1) / STAGES.length) * 100, 92)}%` }}
          transition={{ duration: 0.6, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}
