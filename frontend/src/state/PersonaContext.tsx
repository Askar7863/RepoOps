import { createContext, useContext, type ReactNode } from "react";
import type { Persona } from "@/lib/api/types";
import { useLocalStorage } from "@/hooks/useLocalStorage";
import { PERSONA_KEY } from "@/lib/storage";

const VALID_PERSONAS: Persona[] = ["student", "developer", "architect"];
const DEFAULT_PERSONA: Persona = "developer";

interface PersonaContextValue {
  persona: Persona;
  setPersona: (persona: Persona) => void;
}

const PersonaContext = createContext<PersonaContextValue | undefined>(undefined);

export function PersonaProvider({ children }: { children: ReactNode }) {
  const [stored, setStored] = useLocalStorage<Persona>(PERSONA_KEY, DEFAULT_PERSONA);
  const persona = VALID_PERSONAS.includes(stored) ? stored : DEFAULT_PERSONA;

  return <PersonaContext.Provider value={{ persona, setPersona: setStored }}>{children}</PersonaContext.Provider>;
}

export function usePersona(): PersonaContextValue {
  const ctx = useContext(PersonaContext);
  if (!ctx) throw new Error("usePersona must be used within a PersonaProvider");
  return ctx;
}
