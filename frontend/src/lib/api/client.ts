import type { ApiErrorBody } from "./types";

// "" in both dev (Vite proxies /api/* to the backend, see vite.config.ts)
// and prod (served same-origin through app/main.py's StaticFiles mount) --
// no branching needed between the two.
const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function apiPost<TReq, TRes>(path: string, body: TReq): Promise<TRes> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = ((await res.json()) as ApiErrorBody).detail ?? detail;
    } catch {
      // non-JSON error body -- fall back to statusText
    }
    throw new ApiError(detail, res.status);
  }

  return res.json() as Promise<TRes>;
}
