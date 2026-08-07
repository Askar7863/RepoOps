import { AlertTriangle, Check } from "lucide-react";
import type { SecurityInfo } from "@/lib/api/types";

export function SecuritySection({ securityInfo }: { securityInfo: SecurityInfo }) {
  if (!securityInfo.available) {
    return <p className="text-caption text-muted-foreground">{securityInfo.reason}</p>;
  }

  return (
    <div className="space-y-2 text-caption">
      <p className="text-muted-foreground">
        {securityInfo.suspected_secret_hits > 0 ? (
          <span className="inline-flex items-center gap-1 text-[#e0b872]">
            <AlertTriangle className="size-3.5" aria-hidden="true" />
            {securityInfo.suspected_secret_hits} suspected secret pattern{securityInfo.suspected_secret_hits === 1 ? "" : "s"} across{" "}
            {securityInfo.suspected_secret_files} file{securityInfo.suspected_secret_files === 1 ? "" : "s"}
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 text-[#8fc49a]">
            <Check className="size-3.5" aria-hidden="true" />
            No suspected secret patterns found
          </span>
        )}
      </p>

      {securityInfo.committed_env_file && (
        <p className="inline-flex items-center gap-1 text-[#e0b872]">
          <AlertTriangle className="size-3.5" aria-hidden="true" />
          Committed .env file: {securityInfo.committed_env_file}
        </p>
      )}

      <p className="text-muted-foreground">{securityInfo.has_gitignore ? "Has a .gitignore" : "No .gitignore found"}</p>
    </div>
  );
}
