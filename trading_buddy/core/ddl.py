from duckdb import DuckDBPyConnection


def create_events_table(conn: DuckDBPyConnection):
    """Create events table for storing detected patterns."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            symbol TEXT,
            timeframe TEXT,
            event_ts TIMESTAMP,
            pattern TEXT,
            attrs JSON,
            lead_bars INTEGER,
            fwd_ret DOUBLE,
            max_dd DOUBLE,
            label TEXT,
            oos_split TEXT DEFAULT 'train',  -- 'train', 'test', or 'live'
            PRIMARY KEY (symbol, timeframe, event_ts, pattern)
        )
    """)


def create_contexts_table(conn: DuckDBPyConnection):
    """Create contexts table for storing market conditions."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS contexts (
            symbol TEXT,
            timeframe TEXT,
            event_ts TIMESTAMP,
            context TEXT,
            attrs JSON,
            PRIMARY KEY (symbol, timeframe, event_ts, context)
        )
    """)


def create_consistency_matrix_table(conn: DuckDBPyConnection):
    """Create consistency matrix table for cross-timeframe validation metrics."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS consistency_matrix (
            base_tf TEXT,
            base_pattern TEXT,
            ctx_tf TEXT,
            ctx_pattern TEXT,
            n INTEGER,
            exp_lift DOUBLE,
            hit_rate DOUBLE,
            stability DOUBLE,
            ci DOUBLE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (base_tf, base_pattern, ctx_tf, ctx_pattern)
        )
    """)


def create_indices(conn: DuckDBPyConnection):
    """Create indices for better query performance."""
    # Events indices
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_symbol_ts 
        ON events(symbol, event_ts)
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_pattern 
        ON events(pattern, symbol, timeframe)
    """)
    
    # Contexts indices
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_contexts_symbol_ts 
        ON contexts(symbol, event_ts)
    """)
    
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_contexts_context 
        ON contexts(context, symbol, timeframe)
    """)


def create_reports_daily_table(conn: DuckDBPyConnection):
    """Create reports_daily table for storing compressed daily snapshots."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports_daily (
            symbol TEXT,
            as_of DATE,
            summary_md TEXT,
            top_edges_json JSON,
            drift_json JSON,
            council_ci_json JSON,
            examples_json JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, as_of)
        )
    """)


def create_whatif_cache_table(conn: DuckDBPyConnection):
    """Create whatif_cache table for storing precomputed what-if scenarios."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS whatif_cache (
            hypothesis_hash TEXT,
            overrides_hash TEXT,
            response_json JSON,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            hit_count INTEGER DEFAULT 0,
            PRIMARY KEY (hypothesis_hash, overrides_hash)
        )
    """)
    
    # Index for cleanup of expired entries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_whatif_expires 
        ON whatif_cache(expires_at)
    """)


def create_precursor_tables(conn: DuckDBPyConnection):
    """Create tables for precursor detection and alerts."""
    
    # Alert subscriptions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS alert_subscriptions (
            user_id TEXT,
            symbol TEXT,
            pattern TEXT,
            timeframe TEXT,
            min_probability DOUBLE DEFAULT 0.7,
            quiet_hours_start INTEGER DEFAULT NULL,  -- Hour of day (0-23)
            quiet_hours_end INTEGER DEFAULT NULL,    -- Hour of day (0-23) 
            webhook_url TEXT,
            active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, symbol, pattern, timeframe)
        )
    """)
    
    # Precursor alerts queue/history
    conn.execute("""
        CREATE TABLE IF NOT EXISTS precursor_alerts (
            symbol TEXT,
            pattern TEXT,
            timeframe TEXT,
            probability DOUBLE,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sent_at TIMESTAMP DEFAULT NULL,
            webhook_response_code INTEGER DEFAULT NULL,
            user_id TEXT DEFAULT 'system',
            metadata JSON DEFAULT NULL,
            PRIMARY KEY (symbol, pattern, timeframe, detected_at)
        )
    """)
    
    # Index for cleanup and deduplication
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_precursor_sent 
        ON precursor_alerts(symbol, pattern, sent_at)
    """)
    
    # Index for user queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_alerts_user 
        ON alert_subscriptions(user_id, active)
    """)


def initialize_database(conn: DuckDBPyConnection):
    """Initialize all database tables and indices."""
    create_events_table(conn)
    create_contexts_table(conn)
    create_consistency_matrix_table(conn)
    create_reports_daily_table(conn)
    create_whatif_cache_table(conn)
    create_precursor_tables(conn)
    create_indices(conn)