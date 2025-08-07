"""
Webhook delivery system for precursor alerts.
"""
import json
import logging
from datetime import datetime
from typing import Dict, Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger(__name__)


class WebhookSender:
    """Sends webhook notifications for precursor alerts."""
    
    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        if HAS_HTTPX:
            self.client = httpx.Client(timeout=timeout)
        else:
            self.client = None
    
    def send_precursor_alert(
        self, 
        webhook_url: str, 
        alert_data: Dict,
        user_id: str = "system"
    ) -> Dict[str, any]:
        """
        Send precursor alert via webhook.
        
        Args:
            webhook_url: URL to send webhook to
            alert_data: Precursor alert data
            user_id: User ID for the alert
            
        Returns:
            Dict with status and response info
        """
        try:
            if not HAS_HTTPX or not self.client:
                logger.warning("httpx not available, webhook not sent")
                return {
                    "success": False,
                    "error": "httpx_not_available",
                    "status_code": None,
                    "sent_at": datetime.now(),
                    "webhook_url": webhook_url
                }
            
            # Prepare webhook payload
            payload = {
                "type": "precursor_alert",
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "alert": {
                    "symbol": alert_data.get("symbol"),
                    "pattern": alert_data.get("pattern"),
                    "timeframe": alert_data.get("timeframe"),
                    "probability": alert_data.get("probability"),
                    "current_price": alert_data.get("current_price"),
                    "metadata": alert_data.get("metadata", {})
                },
                "message": self._format_alert_message(alert_data),
                "source": "trading_buddy"
            }
            
            # Send webhook
            response = self.client.post(
                webhook_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "TradingBuddy/1.0"
                }
            )
            
            # Log success
            logger.info(
                f"Webhook sent successfully to {webhook_url}: "
                f"{response.status_code} for {alert_data.get('symbol')} {alert_data.get('pattern')}"
            )
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response_text": response.text[:200] if response.text else "",
                "sent_at": datetime.now(),
                "webhook_url": webhook_url
            }
            
        except httpx.TimeoutException:
            logger.warning(f"Webhook timeout to {webhook_url} for {alert_data.get('symbol')}")
            return {
                "success": False,
                "error": "timeout",
                "status_code": None,
                "sent_at": datetime.now(),
                "webhook_url": webhook_url
            }
            
        except httpx.HTTPStatusError as e:
            logger.warning(f"Webhook HTTP error {e.response.status_code} to {webhook_url}")
            return {
                "success": False,
                "error": f"http_{e.response.status_code}",
                "status_code": e.response.status_code,
                "response_text": e.response.text[:200] if e.response.text else "",
                "sent_at": datetime.now(),
                "webhook_url": webhook_url
            }
            
        except Exception as e:
            logger.error(f"Webhook send failed to {webhook_url}: {e}")
            return {
                "success": False,
                "error": str(e)[:100],
                "status_code": None,
                "sent_at": datetime.now(),
                "webhook_url": webhook_url
            }
    
    def _format_alert_message(self, alert_data: Dict) -> str:
        """Format a human-readable alert message."""
        symbol = alert_data.get("symbol", "???")
        pattern = alert_data.get("pattern", "unknown_pattern")
        timeframe = alert_data.get("timeframe", "5m")
        probability = alert_data.get("probability", 0)
        current_price = alert_data.get("current_price")
        
        # Pattern-specific formatting
        if pattern == "double_bottom":
            first_bottom = alert_data.get("first_bottom_price")
            price_diff = alert_data.get("price_diff_pct", 0) * 100
            
            message = (
                f"🎯 {symbol} Double Bottom Precursor ({timeframe})\n"
                f"Probability: {probability:.1%}\n"
                f"Current: ${current_price:.2f} vs First Bottom: ${first_bottom:.2f} "
                f"({price_diff:+.1f}% diff)\n"
                f"Ready to complete pattern on next few bars"
            )
            
        elif pattern == "macd_bull_cross":
            momentum = alert_data.get("momentum_strength", 0) * 100
            
            message = (
                f"📈 {symbol} MACD Bull Cross Precursor ({timeframe})\n"
                f"Probability: {probability:.1%}\n"
                f"Current: ${current_price:.2f}\n"
                f"Momentum building: {momentum:+.2f}%"
            )
            
        else:
            message = (
                f"🚨 {symbol} {pattern} Precursor ({timeframe})\n"
                f"Probability: {probability:.1%}\n"
                f"Current: ${current_price:.2f}"
            )
        
        return message
    
    def test_webhook(self, webhook_url: str) -> Dict[str, any]:
        """Test webhook endpoint with a simple ping."""
        try:
            payload = {
                "type": "test",
                "timestamp": datetime.now().isoformat(),
                "message": "Trading Buddy webhook test",
                "source": "trading_buddy"
            }
            
            response = self.client.post(
                webhook_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "TradingBuddy/1.0"
                }
            )
            
            return {
                "success": True,
                "status_code": response.status_code,
                "response_text": response.text[:200] if response.text else "",
                "latency_ms": response.elapsed.total_seconds() * 1000
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)[:100],
                "status_code": None,
                "latency_ms": None
            }
    
    def __del__(self):
        """Close HTTP client on cleanup."""
        try:
            self.client.close()
        except:
            pass


def format_slack_message(alert_data: Dict) -> Dict:
    """Format alert for Slack webhook."""
    symbol = alert_data.get("symbol", "???")
    pattern = alert_data.get("pattern", "unknown")
    probability = alert_data.get("probability", 0)
    current_price = alert_data.get("current_price")
    
    # Choose emoji and color based on probability
    if probability >= 0.8:
        color = "good"  # Green
        emoji = "🎯"
    elif probability >= 0.6:
        color = "warning"  # Yellow
        emoji = "⚠️"
    else:
        color = "danger"  # Red
        emoji = "🔍"
    
    # Build Slack attachment
    attachment = {
        "color": color,
        "title": f"{emoji} {symbol} {pattern.replace('_', ' ').title()} Precursor",
        "fields": [
            {
                "title": "Probability",
                "value": f"{probability:.1%}",
                "short": True
            },
            {
                "title": "Current Price",
                "value": f"${current_price:.2f}" if current_price else "N/A",
                "short": True
            }
        ],
        "footer": "Trading Buddy",
        "ts": int(datetime.now().timestamp())
    }
    
    # Add pattern-specific fields
    if pattern == "double_bottom":
        first_bottom = alert_data.get("first_bottom_price")
        if first_bottom:
            attachment["fields"].append({
                "title": "First Bottom",
                "value": f"${first_bottom:.2f}",
                "short": True
            })
    
    return {
        "text": f"{symbol} pattern precursor detected",
        "attachments": [attachment]
    }


def format_discord_message(alert_data: Dict) -> Dict:
    """Format alert for Discord webhook."""
    symbol = alert_data.get("symbol", "???")
    pattern = alert_data.get("pattern", "unknown")
    probability = alert_data.get("probability", 0)
    current_price = alert_data.get("current_price")
    
    # Choose embed color based on probability
    if probability >= 0.8:
        color = 0x00ff00  # Green
    elif probability >= 0.6:
        color = 0xffff00  # Yellow
    else:
        color = 0xff0000  # Red
    
    embed = {
        "title": f"{symbol} {pattern.replace('_', ' ').title()} Precursor",
        "description": f"Pattern likely to complete with {probability:.1%} probability",
        "color": color,
        "fields": [
            {
                "name": "Current Price",
                "value": f"${current_price:.2f}" if current_price else "N/A",
                "inline": True
            }
        ],
        "footer": {
            "text": "Trading Buddy"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "embeds": [embed]
    }