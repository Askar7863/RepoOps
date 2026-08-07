import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";

// Shared, referentially-stable plugin arrays for every place that renders
// markdown (chat messages, Health's AI Summary section, Evolution) -- one
// canonical config, not one per call site.
export const MARKDOWN_REMARK_PLUGINS = [remarkGfm];
export const MARKDOWN_REHYPE_PLUGINS = [rehypeHighlight];
