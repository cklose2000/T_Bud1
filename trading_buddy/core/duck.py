import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
from duckdb import DuckDBPyConnection

from trading_buddy.core.config import settings


class DuckDBManager:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.duckdb_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[DuckDBPyConnection] = None
    
    @property
    def conn(self) -> DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn
    
    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def discover_tables(conn: DuckDBPyConnection) -> List[str]:
    """Discover all tables and views in the database."""
    result = conn.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
        ORDER BY table_name
    """).fetchall()
    return [row[0] for row in result]


def get_column_info(conn: DuckDBPyConnection, table_name: str) -> List[Dict[str, Any]]:
    """Get column information for a table."""
    result = conn.execute(f"""
        SELECT 
            column_name,
            data_type,
            is_nullable,
            column_default
        FROM information_schema.columns
        WHERE table_name = '{table_name}'
        AND table_schema = 'main'
        ORDER BY ordinal_position
    """).fetchall()
    
    return [
        {
            "name": row[0],
            "type": row[1],
            "nullable": row[2] == "YES",
            "default": row[3],
        }
        for row in result
    ]


def guess_bars_table(conn: DuckDBPyConnection) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Find table(s) with OHLCV columns. Returns table name and column mapping.
    Accepts synonyms for column names.
    """
    tables = discover_tables(conn)
    
    # Define synonyms for each required column
    column_synonyms = {
        "timestamp": ["time", "timestamp", "ts", "datetime", "date"],
        "symbol": ["symbol", "ticker", "code", "instrument"],
        "open": ["open", "o", "open_price"],
        "high": ["high", "h", "high_price"],
        "low": ["low", "l", "low_price"],
        "close": ["close", "c", "close_price", "price"],
        "volume": ["volume", "vol", "v", "qty", "quantity"],
        "timeframe": ["timeframe", "tf", "interval", "period"],
    }
    
    for table in tables:
        columns = get_column_info(conn, table)
        column_names = [col["name"].lower() for col in columns]
        
        # Try to map each required column
        mapping = {}
        for required, synonyms in column_synonyms.items():
            for syn in synonyms:
                if syn in column_names:
                    mapping[required] = syn
                    break
        
        # Check if we have at least the core OHLCV columns
        core_columns = ["timestamp", "open", "high", "low", "close", "volume"]
        if all(col in mapping for col in core_columns):
            return table, mapping
    
    return None


def create_meta_table(conn: DuckDBPyConnection):
    """Create meta table for storing configuration."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)


def save_column_mapping(conn: DuckDBPyConnection, mapping: Dict[str, str]):
    """Save column mapping to meta table."""
    create_meta_table(conn)
    json_mapping = json.dumps(mapping)
    conn.execute("""
        INSERT OR REPLACE INTO meta (key, value)
        VALUES ('column_mapping', ?)
    """, [json_mapping])


def load_column_mapping(conn: DuckDBPyConnection) -> Optional[Dict[str, str]]:
    """Load column mapping from meta table."""
    try:
        result = conn.execute("""
            SELECT value FROM meta WHERE key = 'column_mapping'
        """).fetchone()
        if result:
            return json.loads(result[0])
    except:
        pass
    return None


def create_normalized_views(
    conn: DuckDBPyConnection, 
    base_table: str, 
    column_mapping: Dict[str, str]
):
    """
    Create normalized views for different timeframes.
    Creates a general 'bars' view and specific timeframe views.
    """
    # Map actual column names to standard names
    ts_col = column_mapping.get("timestamp", "timestamp")
    symbol_col = column_mapping.get("symbol", "symbol")
    tf_col = column_mapping.get("timeframe", "timeframe")
    
    # Create main bars view
    view_sql = f"""
        CREATE OR REPLACE VIEW bars AS
        SELECT 
            {symbol_col} as symbol,
            {tf_col} as timeframe,
            {ts_col} as ts,
            {column_mapping.get('open', 'open')} as open,
            {column_mapping.get('high', 'high')} as high,
            {column_mapping.get('low', 'low')} as low,
            {column_mapping.get('close', 'close')} as close,
            {column_mapping.get('volume', 'volume')} as volume
        FROM {base_table}
    """
    conn.execute(view_sql)
    
    # Create timeframe-specific views if timeframe column exists
    if tf_col in column_mapping.values():
        timeframes = ["1m", "5m", "15m", "60m", "1d"]
        for tf in timeframes:
            view_name = f"bars_{tf}"
            tf_view_sql = f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM bars
                WHERE timeframe = '{tf}'
            """
            try:
                conn.execute(tf_view_sql)
            except:
                # Skip if timeframe doesn't exist in data
                pass


def initialize_schema_discovery(conn: DuckDBPyConnection) -> bool:
    """
    Main function to auto-discover and set up normalized views.
    Returns True if successful, False otherwise.
    """
    # First check if we already have a saved mapping
    mapping = load_column_mapping(conn)
    if mapping:
        # Verify the mapping still works
        try:
            conn.execute("SELECT * FROM bars LIMIT 1").fetchone()
            return True
        except:
            # Mapping is outdated, rediscover
            pass
    
    # Discover bars table
    result = guess_bars_table(conn)
    if not result:
        return False
    
    base_table, column_mapping = result
    
    # Save the mapping
    save_column_mapping(conn, column_mapping)
    
    # Create normalized views
    create_normalized_views(conn, base_table, column_mapping)
    
    return True