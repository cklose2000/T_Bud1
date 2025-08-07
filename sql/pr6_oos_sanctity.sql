-- PR6: OOS Sanctity & Leakage Kill-Switch
-- Enforce strict temporal boundaries to prevent future peeking

-- OOS configuration - defines what is considered "current" vs "future"
CREATE TABLE IF NOT EXISTS oos_config (
  id               TEXT PRIMARY KEY DEFAULT (uuid()),
  ts               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  current_cutoff   TIMESTAMP,       -- Everything after this is "future" and off-limits
  buffer_hours     INTEGER DEFAULT 24,  -- Safety buffer to prevent near-future peeking
  enforced         BOOLEAN DEFAULT true,  -- Global kill-switch
  notes            TEXT
);

-- Data access audit log - tracks all queries with temporal boundaries
CREATE TABLE IF NOT EXISTS data_access_log (
  id               TEXT PRIMARY KEY DEFAULT (uuid()),
  ts               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  query_hash       TEXT,            -- SHA256 of the actual SQL query
  user_context     TEXT,            -- route, tool, etc that made the query
  symbol           TEXT,
  timeframe        TEXT,
  min_timestamp    TIMESTAMP,       -- Earliest data accessed
  max_timestamp    TIMESTAMP,       -- Latest data accessed (must be < current_cutoff)
  row_count        INTEGER,         -- Rows returned
  violation_type   TEXT,            -- "future_peek", "buffer_violation", null if clean
  stack_trace      TEXT,            -- For debugging violations
  query_text       TEXT             -- Actual SQL for forensics (limited to 2000 chars)
);

-- Purged k-fold splits for backtesting
CREATE TABLE IF NOT EXISTS backtest_splits (
  id               TEXT PRIMARY KEY DEFAULT (uuid()),
  ts               TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  fold_id          INTEGER,
  train_start      TIMESTAMP,
  train_end        TIMESTAMP,
  purge_start      TIMESTAMP,       -- Gap to prevent leakage
  purge_end        TIMESTAMP,
  test_start       TIMESTAMP,
  test_end         TIMESTAMP,
  symbol           TEXT,
  timeframe        TEXT,
  metadata         JSON             -- Additional config
);

-- Pattern event tracking with strict OOS enforcement
CREATE TABLE IF NOT EXISTS pattern_events_oos (
  id               TEXT PRIMARY KEY DEFAULT (uuid()),
  event_ts         TIMESTAMP,       -- When the pattern completed
  discovered_ts    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- When we first detected it
  symbol           TEXT,
  timeframe        TEXT,
  pattern_name     TEXT,
  event_data       JSON,            -- Pattern-specific data
  outcome_data     JSON,            -- Forward returns (computed OOS only)
  is_oos           BOOLEAN DEFAULT false,  -- true if outcome computed out-of-sample
  cutoff_at_event  TIMESTAMP,       -- What was "current_cutoff" when pattern fired
  
  -- Prevent accidentally using in-sample outcomes
  CONSTRAINT check_oos_outcomes CHECK (
    is_oos = false OR (outcome_data IS NOT NULL AND cutoff_at_event > event_ts)
  )
);

-- Initialize default OOS configuration
INSERT OR IGNORE INTO oos_config (id, current_cutoff, buffer_hours, enforced, notes)
VALUES (
  'default',
  CURRENT_TIMESTAMP - INTERVAL '24 hours',  -- Conservative default
  24,
  true,
  'Default OOS configuration - 24h buffer to prevent future peeking'
);