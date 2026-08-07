// Every field here is pinned to the verified backend contracts in the
// implementation plan (section 0) -- nothing invented. If the backend
// ever returns a field not listed here, that's a signal this file is
// out of date, not that the extra field should be silently ignored.

export type Persona = "student" | "developer" | "architect";

// ---------------------------------------------------------------------
// POST /api/chat
// ---------------------------------------------------------------------

export interface ChatRequest {
  message: string;
  persona?: Persona;
}

export interface ChatSource {
  file: string;
  chunk_name: string;
  chunk_type: string;
  lines?: string; // present for code_writer_tool + some repo_evolution_tool foci, absent otherwise
}

export interface ChatResponse {
  reply: string;
  tool_used: string; // "general_chat" | "code_writer_tool" | "repo_summarizer_tool" | "repo_health_tool" | "repo_evolution_tool"
  sources: ChatSource[];
  verified: boolean | null;
  verification_reason: string | null;
}

// ---------------------------------------------------------------------
// POST /api/repositories/health
// ---------------------------------------------------------------------

export interface HealthRequest {
  persona?: Persona;
}

export interface Unavailable {
  available: false;
  reason: string;
}

export interface CodeHealthAvailable {
  available: true;
  total_files: number;
  total_symbols: number;
  language_counts: [string, number][];
  todo_fixme_hack_total: number;
  todo_fixme_hack_files: number;
  oversized_symbol_count: number;
  oversized_symbols_sample: [string, string][];
  duplicate_names_sample: [string, string, number][];
  python_docstring_coverage_pct: number | null;
  python_symbol_count: number;
}
export type CodeHealth = CodeHealthAvailable | Unavailable;

export interface GitInfoAvailable {
  available: true;
  total_commits: number;
  first_commit_date: string;
  last_commit_date: string;
  commits_last_90_days: number;
  contributor_count: number;
  top_contributors: [string, number][];
  bus_factor_top_author: string;
  bus_factor_pct: number;
  hotspot_files: [string, number][];
  weekly_activity: [string, number][]; // 12 entries, [weekLabel, commitCount]
}
export type GitInfo = GitInfoAvailable | Unavailable;

export interface SizeInfo {
  total_size_bytes: number;
  total_size_mb: number;
  total_lines: number;
  file_count: number;
}

export interface DependencyManifest {
  file: string;
  runtime_count: number;
  dev_count: number;
  pinned_count: number | null;
}

export interface DependencyInfoAvailable {
  available: true;
  manifests_parsed: DependencyManifest[];
  manifests_unparsed: string[];
  total_runtime_dependencies: number;
  total_dev_dependencies: number;
}
export type DependencyInfo = DependencyInfoAvailable | Unavailable;

export interface SecurityInfoAvailable {
  available: true;
  suspected_secret_hits: number;
  suspected_secret_files: number;
  committed_env_file: string | null;
  has_gitignore: boolean;
}
export type SecurityInfo = SecurityInfoAvailable | Unavailable;

export interface ComplexityInfoAvailable {
  available: true;
  function_count: number;
  avg_complexity: number;
  high_complexity_count: number;
  high_complexity_sample: [string, string, number][];
}
export type ComplexityInfo = ComplexityInfoAvailable | Unavailable;

export interface SubScores {
  documentation: number | null;
  testing: number | null;
  security: number | null;
  complexity: number | null;
  activity: number | null;
  bus_factor: number | null;
}

export interface DocumentationScore {
  score: number;
  has_readme: boolean;
  docstring_coverage_pct: number | null;
  doc_file_ratio_pct: number;
}

export interface TestingScore {
  score: number;
  test_file_count: number;
  source_file_count: number;
  has_ci_config: boolean; // the only real CI-detection signal that exists anywhere
}

export interface HealthResponse {
  repository: string;
  health_score: number | null;
  sub_scores: SubScores;
  code_health: CodeHealth;
  git_info: GitInfo;
  size_info: SizeInfo;
  dependency_info: DependencyInfo;
  security_info: SecurityInfo;
  complexity_info: ComplexityInfo;
  documentation_score: DocumentationScore;
  testing_score: TestingScore;
  narrative: string;
}

// ---------------------------------------------------------------------
// POST /api/repositories/load
// ---------------------------------------------------------------------

export interface RepositoryLoadRequest {
  repo_url: string;
}

export interface SampleChunk {
  chunk_id: string;
  chunk_name: string;
  chunk_type: string;
  path: string;
  start_line: number;
  end_line: number;
}

export interface IndexingResult {
  collection: string;
  vectors_stored: number;
  sparse_indexed: number;
  [key: string]: unknown;
}

export interface RepositoryLoadResponse {
  success: true;
  repo_url: string;
  local_path: string;
  total_files: number;
  filtered_files: number;
  total_chunks: number;
  message: string;
  indexing: IndexingResult;
  sample_files: string[]; // up to 10
  sample_chunks: SampleChunk[]; // up to 5
}

// ---------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------

export interface ApiErrorBody {
  detail: string;
}
