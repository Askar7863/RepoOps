import { CheckCircle2, AlertTriangle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

interface VerificationBadgeProps {
  verified: boolean | null | undefined;
  reason: string | null | undefined;
}

// Real data the backend already returns (verify_answer's output) --
// rendered honestly: nothing shown for the null case rather than
// inventing a third visual state the API never actually signals.
export function VerificationBadge({ verified, reason }: VerificationBadgeProps) {
  if (verified === null || verified === undefined) return null;

  if (verified) {
    return (
      <Badge variant="outline" className="border-transparent bg-[#6b9e78]/15 text-[#8fc49a]">
        <CheckCircle2 className="size-3" aria-hidden="true" />
        Verified
      </Badge>
    );
  }

  const badge = (
    <Badge variant="outline" className="border-transparent bg-[#c9963f]/15 text-[#e0b872]">
      <AlertTriangle className="size-3" aria-hidden="true" />
      Unverified
    </Badge>
  );

  if (!reason) return badge;

  return (
    <Tooltip>
      <TooltipTrigger asChild>{badge}</TooltipTrigger>
      <TooltipContent className="max-w-xs">{reason}</TooltipContent>
    </Tooltip>
  );
}
