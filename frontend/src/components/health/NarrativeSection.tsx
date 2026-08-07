import { Markdown } from "@/components/chat/Markdown";

export function NarrativeSection({ narrative }: { narrative: string }) {
  return <Markdown text={narrative} />;
}
