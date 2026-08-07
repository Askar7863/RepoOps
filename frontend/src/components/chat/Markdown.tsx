import ReactMarkdown, { type Components } from "react-markdown";
import { MARKDOWN_REMARK_PLUGINS, MARKDOWN_REHYPE_PLUGINS } from "@/lib/markdown";
import { cn } from "@/lib/utils";

// One shared renderer for every place markdown shows up: chat replies,
// Health's AI Summary section, and Evolution -- replaces the old app's
// hand-rolled markdownLite() (headers/bullets/bold only) with real
// markdown: tables, fenced code with syntax highlighting, lists, links.
const COMPONENTS: Components = {
  p: ({ children }) => <p className="mb-3 last:mb-0">{children}</p>,
  h1: ({ children }) => <h1 className="mb-2 mt-4 text-section font-semibold first:mt-0">{children}</h1>,
  h2: ({ children }) => <h2 className="mb-2 mt-4 text-section font-semibold first:mt-0">{children}</h2>,
  h3: ({ children }) => <h3 className="mb-1.5 mt-3 text-body font-semibold first:mt-0">{children}</h3>,
  h4: ({ children }) => <h4 className="mb-1.5 mt-3 text-body font-semibold first:mt-0">{children}</h4>,
  ul: ({ children }) => <ul className="mb-3 list-disc space-y-1 pl-5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-3 list-decimal space-y-1 pl-5 last:mb-0">{children}</ol>,
  li: ({ children }) => <li>{children}</li>,
  a: ({ href, children }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-primary underline underline-offset-2 hover:no-underline">
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="mb-3 border-l-2 border-border pl-3 text-muted-foreground last:mb-0">{children}</blockquote>
  ),
  hr: () => <hr className="my-4 border-border" />,
  strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
  table: ({ children }) => (
    <div className="mb-3 overflow-x-auto last:mb-0">
      <table className="w-full border-collapse text-left text-caption">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="border-b border-border">{children}</thead>,
  th: ({ children }) => <th className="px-2 py-1 font-medium text-muted-foreground">{children}</th>,
  td: ({ children }) => <td className="px-2 py-1 align-top">{children}</td>,
  tr: ({ children }) => <tr className="border-b border-border/50">{children}</tr>,
  pre: ({ children }) => <pre className="mb-3 overflow-x-auto rounded-md bg-secondary p-3 text-[13px] last:mb-0">{children}</pre>,
  code: ({ className, children, ...props }) => {
    const isBlock = /\b(hljs|language-)/.test(className ?? "");
    if (isBlock) {
      return (
        <code className={cn(className, "font-mono text-[13px]")} {...props}>
          {children}
        </code>
      );
    }
    return (
      <code className="rounded bg-secondary px-1 py-0.5 font-mono text-[13px] text-foreground" {...props}>
        {children}
      </code>
    );
  },
};

export function Markdown({ text }: { text: string }) {
  return (
    <div className="text-body leading-relaxed">
      <ReactMarkdown remarkPlugins={MARKDOWN_REMARK_PLUGINS} rehypePlugins={MARKDOWN_REHYPE_PLUGINS} components={COMPONENTS}>
        {text}
      </ReactMarkdown>
    </div>
  );
}
