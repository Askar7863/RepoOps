// Phrased to match the router's own tuned examples (app/agent/router.py's
// TOOL_SCHEMAS descriptions) so each Quick Action reliably lands on the
// tool it's meant to demonstrate, rather than guessing at wording.
export const CANNED_MESSAGES = {
  explainRepo: "What is this repo about?",
  activityEvolution: "How active has this repo been recently?",
  ownershipRisk: "Who owns this repo? Is there a bus-factor risk?",
  securityNotes: "Give me a security check of this repo.",
  compareTemplate: "What changed between  and ?",
} as const;
