import { motion } from "framer-motion";
import type { ChatMessage } from "@/types/domain";

export function MessageBubbleUser({ message }: { message: ChatMessage }) {
  return (
    <motion.div
      id={`msg-${message.id}`}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className="flex justify-end"
    >
      <div className="max-w-[75%] rounded-2xl border border-primary/20 bg-secondary px-4 py-2.5 text-body text-foreground">
        {message.text}
      </div>
    </motion.div>
  );
}
