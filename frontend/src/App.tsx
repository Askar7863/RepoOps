import { AppProviders } from "@/state/AppProviders";
import { AppShell } from "@/components/layout/AppShell";
import { ContextPanel } from "@/components/layout/ContextPanel";
import { RepositoryLoadingScreen } from "@/components/repository-loading/RepositoryLoadingScreen";
import { useNav } from "@/state/NavContext";
import { ChatView } from "@/components/chat/ChatView";
import { HealthView } from "@/components/health/HealthView";
import { EvolutionView } from "@/components/evolution/EvolutionView";
import { HistoryView } from "@/components/history/HistoryView";

function MainContent() {
  const { view } = useNav();

  switch (view) {
    case "chat":
      return <ChatView />;
    case "health":
      return <HealthView />;
    case "evolution":
      return <EvolutionView />;
    case "history":
      return <HistoryView />;
  }
}

function App() {
  return (
    <AppProviders>
      <AppShell contextPanel={<ContextPanel />}>
        <MainContent />
      </AppShell>
      <RepositoryLoadingScreen />
    </AppProviders>
  );
}

export default App;
