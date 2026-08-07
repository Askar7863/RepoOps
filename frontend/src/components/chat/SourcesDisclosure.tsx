import { useState } from "react";
import { ChevronDown, FileText } from "lucide-react";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { ChatSource } from "@/lib/api/types";
import { cn } from "@/lib/utils";

export function SourcesDisclosure({ sources }: { sources: ChatSource[] }) {
  const [open, setOpen] = useState(false);

  if (sources.length === 0) return null;

  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mt-2">
      <CollapsibleTrigger className="flex items-center gap-1 text-caption text-muted-foreground hover:text-foreground">
        <ChevronDown className={cn("size-3 transition-transform", open && "rotate-180")} aria-hidden="true" />
        {sources.length} source{sources.length === 1 ? "" : "s"}
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-1.5 space-y-1">
        {sources.map((s, i) => (
          <div key={`${s.file}-${s.chunk_name}-${i}`} className="flex items-start gap-1.5 text-caption text-muted-foreground">
            <FileText className="mt-0.5 size-3 shrink-0" aria-hidden="true" />
            <span className="truncate">
              <span className="text-foreground">{s.file}</span> -- {s.chunk_name} ({s.chunk_type}
              {s.lines ? `, lines ${s.lines}` : ""})
            </span>
          </div>
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}
