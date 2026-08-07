import { useRecentTopics } from "@/hooks/useRecentTopics";
import { useChat } from "@/state/ChatContext";
import { useNav } from "@/state/NavContext";

export function RecentTopics() {
  const topics = useRecentTopics();
  const { requestScrollTo } = useChat();
  const { setView } = useNav();

  if (topics.length === 0) return null;

  return (
    <div>
      <p className="mb-2 text-caption font-medium text-muted-foreground">Recent topics</p>
      <div className="space-y-0.5">
        {topics.map((m) => (
          <button
            key={m.id}
            type="button"
            onClick={() => {
              setView("chat");
              requestScrollTo(m.id);
            }}
            className="block w-full truncate rounded-md px-2 py-1.5 text-left text-caption text-muted-foreground transition-colors hover:bg-accent hover:text-foreground"
            title={m.text}
          >
            {m.text}
          </button>
        ))}
      </div>
    </div>
  );
}
