import { motion } from "framer-motion";

const DOT_TRANSITION = (delay: number) => ({
  duration: 0.9,
  repeat: Infinity,
  ease: "easeInOut" as const,
  delay,
});

export function TypingIndicator() {
  return (
    <div className="flex items-center gap-1 px-1 py-2" aria-label="Assistant is typing" role="status">
      {[0, 0.15, 0.3].map((delay, i) => (
        <motion.span
          key={i}
          className="size-1.5 rounded-full bg-muted-foreground"
          animate={{ y: [0, -4, 0], opacity: [0.4, 1, 0.4] }}
          transition={DOT_TRANSITION(delay)}
        />
      ))}
    </div>
  );
}
