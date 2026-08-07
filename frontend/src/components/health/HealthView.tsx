import { useEffect } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { useRepo } from "@/state/RepoContext";
import { CollapsibleSection } from "./CollapsibleSection";
import { OverviewSection } from "./OverviewSection";
import { CodeMetricsSection } from "./CodeMetricsSection";
import { LanguagesSection } from "./LanguagesSection";
import { DependenciesSection } from "./DependenciesSection";
import { SecuritySection } from "./SecuritySection";
import { ComplexitySection } from "./ComplexitySection";
import { NarrativeSection } from "./NarrativeSection";

export function HealthView() {
  const repo = useRepo();

  // Reuses RepoContext.health if the auto-trigger (right after a
  // successful load) already populated it -- only fetches fresh if this
  // view is opened before that's happened, or the repo has no health data
  // at all yet.
  useEffect(() => {
    if (repo.active && repo.healthStatus === "idle") {
      void repo.refreshHealth();
    }
  }, [repo.active, repo.healthStatus, repo]);

  if (!repo.active) {
    return <div className="flex h-full items-center justify-center text-caption text-muted-foreground">Load a repository first.</div>;
  }

  if (repo.healthStatus === "loading" || repo.healthStatus === "idle") {
    return <div className="flex h-full items-center justify-center text-caption text-muted-foreground">Analyzing repository...</div>;
  }

  if (repo.healthStatus === "error" || !repo.health) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 text-caption text-muted-foreground">
        <p>{repo.healthError ?? "Could not run the health check."}</p>
        <Button size="sm" onClick={() => repo.refreshHealth()}>
          Try again
        </Button>
      </div>
    );
  }

  const health = repo.health;

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto max-w-3xl space-y-3 px-6 py-6">
        <div className="flex items-center justify-between">
          <p className="text-title font-semibold text-foreground">Repository Health</p>
          <Button variant="ghost" size="icon" onClick={() => repo.refreshHealth()} aria-label="Refresh health check">
            <RefreshCw className="size-4" aria-hidden="true" />
          </Button>
        </div>

        <CollapsibleSection title="Overview">
          <OverviewSection health={health} />
        </CollapsibleSection>
        <CollapsibleSection title="Code Metrics">
          <CodeMetricsSection codeHealth={health.code_health} />
        </CollapsibleSection>
        <CollapsibleSection title="Languages">
          <LanguagesSection codeHealth={health.code_health} />
        </CollapsibleSection>
        <CollapsibleSection title="Dependencies">
          <DependenciesSection dependencyInfo={health.dependency_info} />
        </CollapsibleSection>
        <CollapsibleSection title="Security" subtitle="Heuristic pattern scan, not a CVE audit">
          <SecuritySection securityInfo={health.security_info} />
        </CollapsibleSection>
        <CollapsibleSection title="Performance / Complexity">
          <ComplexitySection complexityInfo={health.complexity_info} />
        </CollapsibleSection>
        <CollapsibleSection title="AI Summary">
          <NarrativeSection narrative={health.narrative} />
        </CollapsibleSection>
      </div>
    </ScrollArea>
  );
}
