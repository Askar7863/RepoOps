import { useCallback, useState } from "react";
import { readLocalStorage, writeLocalStorage } from "@/lib/storage";

export function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T) => void] {
  const [value, setValue] = useState<T>(() => readLocalStorage(key, initialValue));

  const set = useCallback(
    (next: T) => {
      setValue(next);
      writeLocalStorage(key, next);
    },
    [key],
  );

  return [value, set];
}
