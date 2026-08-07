import { MessageList } from "./MessageList";
import { Composer } from "./Composer";

export function ChatView() {
  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 overflow-hidden">
        <MessageList />
      </div>
      <Composer />
    </div>
  );
}
