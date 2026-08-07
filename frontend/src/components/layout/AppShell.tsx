import { useState, type ReactNode } from "react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { Sidebar } from "./Sidebar";
import { Header } from "./Header";

interface AppShellProps {
  children: ReactNode;
  contextPanel?: ReactNode;
}

// Sidebar and the context panel are fixed-px side columns with the main
// content as the flexible remainder -- literal percentage columns (as
// worded in the design brief) degrade badly under ~1280px viewports.
export function AppShell({ children, contextPanel }: AppShellProps) {
  const isDesktop = useMediaQuery("(min-width: 1024px)");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      {isDesktop ? (
        <Sidebar />
      ) : (
        <Sheet open={sidebarOpen} onOpenChange={setSidebarOpen}>
          <SheetContent side="left" className="w-64 border-none p-0">
            <SheetTitle className="sr-only">Navigation</SheetTitle>
            <Sidebar />
          </SheetContent>
        </Sheet>
      )}

      <div className="flex flex-1 flex-col overflow-hidden">
        <Header
          onOpenSidebar={isDesktop ? undefined : () => setSidebarOpen(true)}
          onOpenContextPanel={isDesktop || !contextPanel ? undefined : () => setPanelOpen(true)}
        />

        <div className="flex flex-1 overflow-hidden">
          <main className="flex-1 overflow-hidden">{children}</main>

          {contextPanel &&
            (isDesktop ? (
              <aside className="w-80 shrink-0 overflow-y-auto border-l border-border">{contextPanel}</aside>
            ) : (
              <Sheet open={panelOpen} onOpenChange={setPanelOpen}>
                <SheetContent side="right" className="w-80 border-none p-0">
                  <SheetTitle className="sr-only">Context</SheetTitle>
                  {contextPanel}
                </SheetContent>
              </Sheet>
            ))}
        </div>
      </div>
    </div>
  );
}
