import { createContext, useContext, useState, type ReactNode } from "react";
import type { NavView } from "@/types/domain";

interface NavContextValue {
  view: NavView;
  setView: (view: NavView) => void;
}

const NavContext = createContext<NavContextValue | undefined>(undefined);

export function NavProvider({ children }: { children: ReactNode }) {
  const [view, setView] = useState<NavView>("chat");

  return <NavContext.Provider value={{ view, setView }}>{children}</NavContext.Provider>;
}

export function useNav(): NavContextValue {
  const ctx = useContext(NavContext);
  if (!ctx) throw new Error("useNav must be used within a NavProvider");
  return ctx;
}
