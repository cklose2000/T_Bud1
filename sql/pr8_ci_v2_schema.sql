-- PR8: Consistency Index v2 Schema with Uncertainty & FDR
-- Stores bootstrap confidence intervals and FDR-adjusted p-values

CREATE TABLE IF NOT EXISTS consistency_matrix_v2 (
    id TEXT PRIMARY KEY,
    
    -- Pattern identifiers
    base_tf TEXT NOT NULL,
    base_pattern TEXT NOT NULL,
    ctx_tf TEXT NOT NULL,
    ctx_pattern TEXT NOT NULL,
    symbol TEXT NOT NULL,
    
    -- Sample sizes
    n_oos_present INTEGER NOT NULL,  -- Count when context is present
    n_oos_absent INTEGER NOT NULL,   -- Count when context is absent
    
    -- Expectancy lift with uncertainty
    exp_lift_mean DOUBLE NOT NULL,
    exp_lift_ci_lo DOUBLE NOT NULL,
    exp_lift_ci_hi DOUBLE NOT NULL,
    p_value_exp DOUBLE,  -- P-value from Welch's t-test
    
    -- Hit rate lift with uncertainty  
    hit_lift_mean DOUBLE NOT NULL,
    hit_lift_ci_lo DOUBLE NOT NULL,
    hit_lift_ci_hi DOUBLE NOT NULL,
    p_value_hit DOUBLE,  -- P-value from two-proportion z-test
    
    -- FDR control
    fdr_q DOUBLE,  -- FDR-adjusted p-value (q-value)
    fdr_pass BOOLEAN DEFAULT false,  -- Whether passes FDR threshold
    fdr_month TEXT,  -- YYYY-MM for grouping
    
    -- Power and stability
    power_score DOUBLE NOT NULL,  -- min(1, n_eff/120)
    stability DOUBLE NOT NULL,  -- Stability metric [0,1]
    
    -- Final consistency index
    ci_raw DOUBLE NOT NULL,  -- Raw CI before clamping
    ci_final DOUBLE NOT NULL,  -- Final CI after power/stability adjustment
    trade_gate TEXT NOT NULL CHECK (trade_gate IN ('allowed', 'watch_only', 'blocked')),
    gate_reason TEXT,  -- Why gate decision was made
    
    -- Metadata
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    computation_time_ms INTEGER  -- How long bootstrap took
);

-- Create indices separately
CREATE INDEX IF NOT EXISTS idx_pattern_lookup ON consistency_matrix_v2 (base_tf, base_pattern, ctx_tf, ctx_pattern, symbol, computed_at);
CREATE INDEX IF NOT EXISTS idx_fdr_month ON consistency_matrix_v2 (symbol, fdr_month);
CREATE INDEX IF NOT EXISTS idx_trade_gate ON consistency_matrix_v2 (trade_gate, computed_at);

-- View for latest CI values per pattern combination
CREATE OR REPLACE VIEW consistency_matrix_latest AS 
SELECT cm.*
FROM consistency_matrix_v2 cm
INNER JOIN (
    SELECT base_tf, base_pattern, ctx_tf, ctx_pattern, symbol, 
           MAX(computed_at) as max_computed_at
    FROM consistency_matrix_v2
    GROUP BY base_tf, base_pattern, ctx_tf, ctx_pattern, symbol
) latest 
ON cm.base_tf = latest.base_tf 
AND cm.base_pattern = latest.base_pattern
AND cm.ctx_tf = latest.ctx_tf
AND cm.ctx_pattern = latest.ctx_pattern
AND cm.symbol = latest.symbol
AND cm.computed_at = latest.max_computed_at;

-- Summary statistics view
CREATE OR REPLACE VIEW ci_v2_summary AS
SELECT 
    symbol,
    COUNT(*) as total_pairs,
    COUNT(CASE WHEN trade_gate = 'allowed' THEN 1 END) as allowed_count,
    COUNT(CASE WHEN trade_gate = 'watch_only' THEN 1 END) as watch_only_count,
    COUNT(CASE WHEN trade_gate = 'blocked' THEN 1 END) as blocked_count,
    AVG(ci_final) as avg_ci_final,
    AVG(power_score) as avg_power,
    AVG(stability) as avg_stability,
    COUNT(CASE WHEN fdr_pass THEN 1 END) as fdr_discoveries,
    MIN(computed_at) as oldest_computation,
    MAX(computed_at) as latest_computation
FROM consistency_matrix_latest
GROUP BY symbol;

-- Migration helper to track CI version in use
CREATE TABLE IF NOT EXISTS ci_version_config (
    id TEXT PRIMARY KEY DEFAULT ('default'),
    version TEXT NOT NULL CHECK (version IN ('v1', 'v2')),
    enabled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rollout_percentage INTEGER DEFAULT 0 CHECK (rollout_percentage BETWEEN 0 AND 100),
    notes TEXT
);

-- Initialize with v1 (existing system)
INSERT OR IGNORE INTO ci_version_config (id, version, rollout_percentage, notes)
VALUES ('default', 'v1', 0, 'Initial state - using v1 CI without uncertainty');

-- Audit log for CI v2 decisions
CREATE TABLE IF NOT EXISTS ci_v2_decision_log (
    id TEXT PRIMARY KEY,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    request_id TEXT,  -- For correlating with requests
    symbol TEXT,
    base_pattern TEXT,
    ctx_pattern TEXT,
    
    -- Decision inputs
    n_eff INTEGER,
    exp_lift_mean DOUBLE,
    exp_lift_ci TEXT,  -- JSON [lo, hi]
    hit_lift_mean DOUBLE,
    hit_lift_ci TEXT,  -- JSON [lo, hi]
    stability DOUBLE,
    power_score DOUBLE,
    fdr_q DOUBLE,
    fdr_pass BOOLEAN,
    
    -- Decision outputs
    ci_raw DOUBLE,
    ci_final DOUBLE,
    trade_gate TEXT,
    gate_reason TEXT,
    
    -- Version tracking
    ci_version TEXT
);