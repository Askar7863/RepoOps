import type { Persona } from "@/lib/api/types";
import { usePersona } from "@/state/PersonaContext";
import { cn } from "@/lib/utils";

const PERSONAS: { value: Persona; label: string; title: string }[] = [
  { value: "student", label: "Student", title: "Simpler, conceptual explanations for someone learning the codebase" },
  { value: "developer", label: "Developer", title: "Technical, direct answers for someone actively working in the code" },
  { value: "architect", label: "Architect", title: "Structural, trade-off and risk framing for assessing the system as a whole" },
];

export function PersonaSwitcher() {
  const { persona, setPersona } = usePersona();

  return (
    <div role="radiogroup" aria-label="Response style" className="flex items-center gap-1 rounded-lg border border-border bg-card p-1">
      {PERSONAS.map((p) => {
        const active = persona === p.value;
        return (
          <button
            key={p.value}
            type="button"
            role="radio"
            aria-checked={active}
            title={p.title}
            onClick={() => setPersona(p.value)}
            className={cn(
              "rounded-md px-3 py-1 text-caption font-medium transition-colors",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
              active ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
            )}
          >
            {p.label}
          </button>
        );
      })}
    </div>
  );
}
