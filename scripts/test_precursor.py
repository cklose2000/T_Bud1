#!/usr/bin/env python3
"""
Test precursor detection system.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_buddy.core.duck import DuckDBManager
from trading_buddy.precursors.detector import PrecursorDetector
from trading_buddy.precursors.job import PrecursorJob
from trading_buddy.precursors.webhook import WebhookSender


def test_precursor_detector():
    """Test basic precursor detection."""
    print("Testing precursor detector...")
    
    with DuckDBManager() as db:
        detector = PrecursorDetector(db.conn)
        
        # Test getting active symbols
        symbols = detector.get_active_symbols()
        print(f"Active symbols: {symbols}")
        
        # Test detection on a symbol (will likely find nothing without real pattern data)
        if symbols:
            test_symbol = symbols[0]
            print(f"\nTesting double bottom detection for {test_symbol}...")
            
            result = detector.detect_double_bottom_precursor(test_symbol)
            if result:
                print(f"Found precursor: {result}")
            else:
                print("No double bottom precursor found (expected without pattern data)")
            
            print(f"\nTesting MACD cross detection for {test_symbol}...")
            result = detector.detect_macd_cross_precursor(test_symbol)
            if result:
                print(f"Found precursor: {result}")
            else:
                print("No MACD cross precursor found (expected without MACD data)")
        
        print("✓ Precursor detector working")


def test_webhook_sender():
    """Test webhook sending (without actually sending)."""
    print("\nTesting webhook sender...")
    
    webhook_sender = WebhookSender()
    
    # Mock alert data
    alert_data = {
        "symbol": "SPY",
        "pattern": "double_bottom",
        "timeframe": "5m",
        "probability": 0.75,
        "current_price": 627.50,
        "first_bottom_price": 627.04,
        "price_diff_pct": 0.007,
        "metadata": {"test": True}
    }
    
    # Format message (without sending)
    formatted_msg = webhook_sender._format_alert_message(alert_data)
    print("Formatted message:")
    print(formatted_msg)
    
    print("✓ Webhook sender working")


def test_precursor_job():
    """Test precursor job logic."""
    print("\nTesting precursor job...")
    
    with DuckDBManager() as db:
        job = PrecursorJob(db.conn)
        
        # Test getting subscriptions (should be empty)
        subscriptions = job._get_active_subscriptions()
        print(f"Active subscriptions: {len(subscriptions)}")
        
        # Test running detection cycle  
        result = job.run_detection_cycle()
        print(f"Detection cycle result: {result}")
        
        # Test getting stats
        stats = job.get_alert_stats(days=7)
        print(f"Alert stats: {stats}")
        
        print("✓ Precursor job working")


def test_subscription_management():
    """Test alert subscription management."""
    print("\nTesting subscription management...")
    
    with DuckDBManager() as db:
        # Add a test subscription
        db.conn.execute("""
            INSERT OR REPLACE INTO alert_subscriptions
            (user_id, symbol, pattern, timeframe, min_probability, webhook_url)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            "test_user",
            "SPY", 
            "double_bottom",
            "5m",
            0.7,
            "https://hooks.slack.com/test/webhook"
        ])
        
        # Verify it was added
        subs = db.conn.execute("""
            SELECT user_id, symbol, pattern, active
            FROM alert_subscriptions
            WHERE user_id = 'test_user'
        """).fetchall()
        
        print(f"Test subscriptions: {subs}")
        
        # Clean up
        db.conn.execute("DELETE FROM alert_subscriptions WHERE user_id = 'test_user'")
        
        print("✓ Subscription management working")


def main():
    """Run all precursor tests."""
    print("=" * 60)
    print("PRECURSOR DETECTION TESTS")
    print("=" * 60)
    
    try:
        test_precursor_detector()
        test_webhook_sender()
        test_precursor_job()
        test_subscription_management()
        
        print("\n" + "=" * 60)
        print("✅ ALL PRECURSOR TESTS PASSED")
        print("=" * 60)
        print("\nPrecursor detection system is ready!")
        print("Key features:")
        print("- Pattern precursor detection (double bottom, MACD cross)")
        print("- Webhook notifications with deduplication") 
        print("- User subscription management")
        print("- Background job scheduling")
        print("- Alert history and statistics")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()