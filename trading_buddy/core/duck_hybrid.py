"""
Enhanced DuckDB manager that supports hybrid database setup.
"""
from pathlib import Path
from typing import Optional

import duckdb
from duckdb import DuckDBPyConnection

from trading_buddy.core.config import settings
from trading_buddy.core.duck import DuckDBManager


class HybridDuckDBManager(DuckDBManager):
    """DuckDB manager that attaches IB database for market data access."""
    
    def __init__(self, db_path: Optional[Path] = None):
        super().__init__(db_path)
        self.ib_db_path = Path("/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb")
    
    @property
    def conn(self) -> DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
            
            # Attach IB database if it exists
            if self.ib_db_path.exists():
                try:
                    self._conn.execute(f"""
                        ATTACH DATABASE '{self.ib_db_path}' AS ib_data (READ_ONLY)
                    """)
                except:
                    # Database might already be attached
                    pass
                    
        return self._conn