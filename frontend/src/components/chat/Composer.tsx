import { useState, type KeyboardEvent } from "react";
import { ArrowUp } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { useChat } from "@/state/ChatContext";
import { useRepo } from "@/state/RepoContext";
import { useAutoResizeTextarea } from "@/hooks/useAutoResizeTextarea";

export function Composer() {
  const { draft, setDraft, send, status } = useChat();
  const repo = useRepo();
  const textareaRef = useAutoResizeTextarea(draft);
  const [isComposing, setIsComposing] = useState(false);

  const disabled = status === "sending" || !repo.active;

  const handleSend = () => {
    const text = draft;
    if (!text.trim() || disabled) return;
    setDraft("");
    void send(text);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !isComposing) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="border-t border-border bg-background px-6 py-4">
      <div className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-border bg-input px-3 py-2">
        <Textarea
          ref={textareaRef}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          placeholder={repo.active ? "Ask anything about your repository..." : "Load a repository first..."}
          disabled={disabled}
          rows={1}
          className="max-h-40 min-h-9 resize-none border-none bg-transparent px-1 py-1.5 shadow-none focus-visible:ring-0"
        />
        <Button
          type="button"
          size="icon"
          onClick={handleSend}
          disabled={disabled || !draft.trim()}
          aria-label="Send message"
          className="mb-0.5 shrink-0 rounded-full"
        >
          <ArrowUp className="size-4" aria-hidden="true" />
        </Button>
      </div>
    </div>
  );
}
