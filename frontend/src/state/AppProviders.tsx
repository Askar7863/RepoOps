import type { ReactNode } from "react";
import { TooltipProvider } from "@/components/ui/tooltip";
import { PersonaProvider } from "./PersonaContext";
import { RepoProvider } from "./RepoContext";
import { ChatProvider } from "./ChatContext";
import { NavProvider } from "./NavContext";

// Order matters: RepoProvider consumes usePersona(), ChatProvider consumes
// both usePersona() and useRepo() -- each must be an ANCESTOR of the ones
// that read it. NavProvider has no dependencies, innermost is arbitrary.
export function AppProviders({ children }: { children: ReactNode }) {
  return (
    <TooltipProvider delayDuration={200}>
      <PersonaProvider>
        <RepoProvider>
          <ChatProvider>
            <NavProvider>{children}</NavProvider>
          </ChatProvider>
        </RepoProvider>
      </PersonaProvider>
    </TooltipProvider>
  );
}
