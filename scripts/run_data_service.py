#!/usr/bin/env python3
"""
Run Trading Buddy Data Sync Service
Continuously syncs data from IB Gateway and detects patterns
"""
import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.data.data_sync_service import DataSyncService
from trading_buddy.core.config import settings


def setup_logging(log_file: str = "data_sync.log"):
    """Setup logging configuration"""
    log_path = Path("logs") / log_file
    log_path.parent.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    # Reduce noise from ib_insync
    logging.getLogger('ib_insync').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Trading Buddy Data Sync Service")
    
    parser.add_argument(
        "--ib-host",
        type=str,
        default=os.getenv("IB_GATEWAY_HOST", "localhost"),
        help="IB Gateway host"
    )
    
    parser.add_argument(
        "--ib-port",
        type=int,
        default=int(os.getenv("IB_GATEWAY_PORT", "7497")),
        help="IB Gateway port (7497 for paper, 7496 for live)"
    )
    
    parser.add_argument(
        "--client-id",
        type=int,
        default=int(os.getenv("IB_CLIENT_ID", "10")),
        help="IB client ID"
    )
    
    parser.add_argument(
        "--ib-db",
        type=str,
        default="/Users/chandler/claude5/claude-flow/ib-backtest-system/data/unified_market_data.duckdb",
        help="Path to IB market data database"
    )
    
    parser.add_argument(
        "--tb-db",
        type=str,
        default="./data/trading_buddy.duckdb",
        help="Path to Trading Buddy database"
    )
    
    parser.add_argument(
        "--no-pattern-detection",
        action="store_true",
        help="Disable automatic pattern detection"
    )
    
    parser.add_argument(
        "--sync-interval",
        type=int,
        default=60,
        help="Sync interval in seconds"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("Trading Buddy Data Sync Service")
    logger.info("=" * 60)
    logger.info(f"IB Gateway: {args.ib_host}:{args.ib_port}")
    logger.info(f"Client ID: {args.client_id}")
    logger.info(f"IB Database: {args.ib_db}")
    logger.info(f"TB Database: {args.tb_db}")
    logger.info(f"Pattern Detection: {'Disabled' if args.no_pattern_detection else 'Enabled'}")
    logger.info(f"Sync Interval: {args.sync_interval}s")
    logger.info("=" * 60)
    
    # Check if IB Gateway is running
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((args.ib_host, args.ib_port))
    sock.close()
    
    if result != 0:
        logger.error(f"Cannot connect to IB Gateway at {args.ib_host}:{args.ib_port}")
        logger.error("Please ensure IB Gateway or TWS is running and configured to accept API connections")
        logger.error("Settings -> API -> Settings -> Enable ActiveX and Socket Clients")
        return 1
    
    # Create and start service
    service = DataSyncService(
        ib_db_path=args.ib_db,
        tb_db_path=args.tb_db,
        ib_host=args.ib_host,
        ib_port=args.ib_port,
        client_id=args.client_id
    )
    
    service.pattern_detection_enabled = not args.no_pattern_detection
    service.sync_interval = args.sync_interval
    
    try:
        logger.info("Starting data sync service...")
        service.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
        service.stop()
    except Exception as e:
        logger.error(f"Service error: {e}", exc_info=True)
        service.stop()
        return 1
    
    logger.info("Data sync service stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())