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


def initialize_database(conn: DuckDBPyConnection):
    """Initialize all database tables and indices."""
    create_events_table(conn)
    create_contexts_table(conn)
    create_consistency_matrix_table(conn)
    create_indices(conn)