import { useState, type FormEvent } from "react";
import { FolderGit2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { useRepo } from "@/state/RepoContext";

function looksLikeGithubUrl(value: string): boolean {
  // Cosmetic-only check -- the real validation is server-side
  // (POST /api/repositories/load rejects anything else with a 400
  // whose message is surfaced verbatim); this just avoids firing an
  // obviously-empty or non-GitHub request.
  return /^https?:\/\/(www\.)?github\.com\//i.test(value.trim());
}

export function RepoLoaderInput() {
  const repo = useRepo();
  const [url, setUrl] = useState("");
  const isLoading = repo.loadStatus === "loading";

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (!looksLikeGithubUrl(url) || isLoading) return;
    repo.loadRepository(url.trim());
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-1 items-center gap-2">
      <div className="relative flex-1">
        <FolderGit2 className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
        <Input
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          placeholder="https://github.com/owner/repository"
          aria-label="GitHub repository URL"
          disabled={isLoading}
          className="pl-9"
        />
      </div>
      <Button type="submit" disabled={isLoading || !looksLikeGithubUrl(url)}>
        {isLoading ? "Loading..." : "Load Repository"}
      </Button>
    </form>
  );
}
