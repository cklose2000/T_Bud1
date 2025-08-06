from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Database
    duckdb_path: Path = Field(default=Path("./data/trading.duckdb"))
    
    # IB Gateway
    ib_gateway_host: str = Field(default="localhost")
    ib_gateway_port: int = Field(default=4002)
    ib_client_id: int = Field(default=10)
    ib_db_path: str = Field(default="/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb")
    
    # Real-time Data
    realtime_symbols: str = Field(default="SPY,QQQ,NVDA")
    realtime_timeframes: str = Field(default="5min,15min,60min")
    pattern_detection_enabled: bool = Field(default=True)
    sync_interval_seconds: int = Field(default=60)
    
    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8080)
    api_reload: bool = Field(default=True)
    
    # Logging
    log_level: str = Field(default="INFO")
    
    # Trading
    default_lookback_days: int = Field(default=365)
    default_horizon_bars: int = Field(default=10)
    default_risk_budget: float = Field(default=0.01)
    
    # Alerts
    alert_check_interval_seconds: int = Field(default=60)
    precursor_probability_threshold: float = Field(default=0.7)


settings = Settings()