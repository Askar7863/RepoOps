import { Menu, PanelRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { RepoLoaderInput } from "./RepoLoaderInput";
import { PersonaSwitcher } from "./PersonaSwitcher";

interface HeaderProps {
  onOpenSidebar?: () => void;
  onOpenContextPanel?: () => void;
}

export function Header({ onOpenSidebar, onOpenContextPanel }: HeaderProps) {
  return (
    <header className="flex flex-wrap items-center gap-3 border-b border-border bg-background/80 px-4 py-3 backdrop-blur">
      <div className="flex min-w-[220px] flex-1 items-center gap-3">
        {onOpenSidebar && (
          <Button variant="ghost" size="icon" onClick={onOpenSidebar} aria-label="Open navigation">
            <Menu className="size-5" aria-hidden="true" />
          </Button>
        )}
        <RepoLoaderInput />
      </div>

      <div className="flex shrink-0 items-center gap-3">
        <PersonaSwitcher />
        {onOpenContextPanel && (
          <Button variant="ghost" size="icon" onClick={onOpenContextPanel} aria-label="Open context panel">
            <PanelRight className="size-5" aria-hidden="true" />
          </Button>
        )}
      </div>
    </header>
  );
}
