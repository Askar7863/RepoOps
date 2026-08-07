import { motion } from "framer-motion";
import { CheckCircle2 } from "lucide-react";
import type { RepositoryLoadResponse } from "@/lib/api/types";

const STAT_STAGGER = 0.08;

function Stat({ label, value, delay }: { label: string; value: number | string; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
      className="rounded-lg border border-border bg-card px-3 py-2 text-center"
    >
      <p className="text-section font-semibold text-foreground">{value}</p>
      <p className="text-caption text-muted-foreground">{label}</p>
    </motion.div>
  );
}

export function LoadResultReveal({ result }: { result: RepositoryLoadResponse }) {
  return (
    <div className="text-center">
      <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="mb-3 flex justify-center">
        <CheckCircle2 className="size-8 text-primary" aria-hidden="true" />
      </motion.div>
      <p className="mb-4 text-section font-semibold text-foreground">Repository ready</p>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="Files" value={result.filtered_files} delay={0 * STAT_STAGGER} />
        <Stat label="Chunks" value={result.total_chunks} delay={1 * STAT_STAGGER} />
        <Stat label="Vectors" value={result.indexing.vectors_stored} delay={2 * STAT_STAGGER} />
        <Stat label="Sparse indexed" value={result.indexing.sparse_indexed} delay={3 * STAT_STAGGER} />
      </div>
    </div>
  );
}
