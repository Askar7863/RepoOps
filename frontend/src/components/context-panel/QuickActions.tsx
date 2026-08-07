import type { LucideIcon } from "lucide-react";
import { BookOpen, HeartPulse, TrendingUp, Users, GitCompareArrows, ShieldAlert } from "lucide-react";
import { useChat } from "@/state/ChatContext";
import { useNav } from "@/state/NavContext";
import { useRepo } from "@/state/RepoContext";
import { CANNED_MESSAGES } from "@/lib/canned-messages";
import { cn } from "@/lib/utils";

interface QuickAction {
  label: string;
  icon: LucideIcon;
  title?: string;
  run: (ctx: { send: (t: string) => void; setDraft: (t: string) => void; setView: (v: "chat" | "health") => void }) => void;
}

// Every action here maps to a real backend capability -- no diagrams, no
// dependency graph, no bug-finder, no README generation (see the plan's
// scope-cut table for why those are excluded, not forgotten).
const ACTIONS: QuickAction[] = [
  {
    label: "Explain This Repo",
    icon: BookOpen,
    run: ({ send, setView }) => {
      setView("chat");
      send(CANNED_MESSAGES.explainRepo);
    },
  },
  {
    label: "Health Check",
    icon: HeartPulse,
    // Goes straight to the dedicated endpoint via the Health nav view,
    // not through chat -- routing this through /api/chat would hit
    // repo_health_tool but return only prose, discarding the structured
    // numbers the dedicated endpoint gives for free.
    run: ({ setView }) => setView("health"),
  },
  {
    label: "Activity",
    icon: TrendingUp,
    run: ({ send, setView }) => {
      setView("chat");
      send(CANNED_MESSAGES.activityEvolution);
    },
  },
  {
    label: "Ownership & Bus-Factor Risk",
    icon: Users,
    run: ({ send, setView }) => {
      setView("chat");
      send(CANNED_MESSAGES.ownershipRisk);
    },
  },
  {
    label: "Compare Branches/Tags",
    icon: GitCompareArrows,
    // No ref-picker exists in scope -- prefill, don't auto-send, since
    // the message is guaranteed incomplete until the user fills in refs.
    run: ({ setDraft, setView }) => {
      setView("chat");
      setDraft(CANNED_MESSAGES.compareTemplate);
    },
  },
  {
    label: "Security Notes",
    icon: ShieldAlert,
    title: "Heuristic secret/.env scan -- not a CVE or dependency vulnerability audit",
    run: ({ send, setView }) => {
      setView("chat");
      send(CANNED_MESSAGES.securityNotes);
    },
  },
];

export function QuickActions() {
  const chat = useChat();
  const nav = useNav();
  const repo = useRepo();
  const disabled = !repo.active;

  return (
    <div>
      <p className="mb-2 text-caption font-medium text-muted-foreground">Quick Actions</p>
      <div className="space-y-1">
        {ACTIONS.map((action) => (
          <button
            key={action.label}
            type="button"
            disabled={disabled}
            title={action.title}
            onClick={() =>
              action.run({
                send: (t) => void chat.send(t),
                setDraft: chat.setDraft,
                setView: nav.setView,
              })
            }
            className={cn(
              "flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-caption transition-colors",
              disabled ? "cursor-not-allowed text-muted-foreground/50" : "text-muted-foreground hover:bg-accent hover:text-foreground",
            )}
          >
            <action.icon className="size-3.5 shrink-0" aria-hidden="true" />
            {action.label}
          </button>
        ))}
      </div>
    </div>
  );
}
