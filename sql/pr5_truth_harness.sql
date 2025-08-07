-- Every numeric/assertive claim must be traceable to a tool call
CREATE TABLE IF NOT EXISTS claims_log (
  id               TEXT PRIMARY KEY DEFAULT (uuid()),
  ts               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  route            TEXT,                 -- /council/vote, /agentic/...
  symbol           TEXT,
  message_id       TEXT,                 -- id for a single LLM response
  claim_text       TEXT,                 -- "median +0.28% over 10 bars"
  value_numeric    DOUBLE,               -- parsed number (bps/%, scaled to decimal)
  unit             TEXT,                 -- "pct", "bps", "bars", etc.
  tool_name        TEXT,                 -- council_vote, whatif, duckdb_sql
  tool_inputs_hash TEXT,                 -- canonical hash of inputs
  tool_outputs_hash TEXT,                -- canonical hash of outputs used
  verdict          TEXT,                 -- "ok" | "corrected" | "rejected"
  notes            TEXT
);

-- Enforce that a final rendered message can't sneak untraced numbers (referee writes a row)
CREATE TABLE IF NOT EXISTS final_responses (
  message_id       TEXT PRIMARY KEY,
  ts               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  route            TEXT,
  symbol           TEXT,
  body             TEXT,
  claims_count     INTEGER,
  verified_count   INTEGER,
  unverified_count INTEGER
);