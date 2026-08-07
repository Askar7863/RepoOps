import { motion } from "framer-motion";
import { Box } from "lucide-react";
import type { ChatMessage } from "@/types/domain";
import { Markdown } from "./Markdown";
import { SourcesDisclosure } from "./SourcesDisclosure";
import { VerificationBadge } from "./VerificationBadge";

export function MessageBubbleAssistant({ message }: { message: ChatMessage }) {
  return (
    <motion.div
      id={`msg-${message.id}`}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      className="flex gap-3"
    >
      <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full border border-border bg-card">
        <Box className="size-3.5 text-primary" aria-hidden="true" />
      </div>
      <div className="max-w-[85%] rounded-2xl border border-border bg-card px-4 py-3">
        <Markdown text={message.text} />
        {(message.sources?.length ?? 0) > 0 && <SourcesDisclosure sources={message.sources!} />}
        {message.verified !== undefined && (
          <div className="mt-2">
            <VerificationBadge verified={message.verified} reason={message.verification_reason} />
          </div>
        )}
      </div>
    </motion.div>
  );
}
