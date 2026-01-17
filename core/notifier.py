"""
Telegram Notifier Module
Handles async Telegram notifications for the Grid Trading Bot
"""

import logging
import os
from typing import Optional, Dict, Any
from telegram import Bot
from telegram.error import TelegramError

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Async Telegram Notification Handler
    
    Sends formatted messages to Telegram without blocking the main bot loop.
    Implements graceful error handling to prevent crashes if Telegram is down.
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        """
        Initialize Telegram notifier
        
        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Telegram chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = Bot(token=bot_token)
        
        logger.info(f"[TELEGRAM] Notifier initialized for chat: {chat_id}")
    
    async def send_message(self, text: str, parse_mode: str = 'Markdown') -> bool:
        """
        Send a message to Telegram
        
        Args:
            text: Message text (supports Markdown)
            parse_mode: Parsing mode ('Markdown' or 'HTML')
        
        Returns:
            bool: True if sent successfully, False otherwise
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
            logger.debug(f"[TELEGRAM] Message sent successfully")
            return True
            
        except TelegramError as e:
            logger.warning(f"[TELEGRAM] Failed to send message: {str(e)}")
            return False
            
        except Exception as e:
            logger.error(f"[TELEGRAM] Unexpected error sending message: {str(e)}")
            return False
    
    async def send_startup_message(self, grid_params: Dict[str, Any], symbol: str = "BTC/USDT") -> bool:
        """
        Send bot startup notification
        
        Args:
            grid_params: Grid parameters dict
            symbol: Trading symbol
        
        Returns:
            bool: True if sent successfully
        """
        lower_bound = grid_params.get('lower_bound', 0)
        upper_bound = grid_params.get('upper_bound', 0)
        grid_spacing = grid_params.get('grid_spacing', 0)
        
        message = (
            f"🤖 *Grid Trading Bot Started*\n\n"
            f"Symbol: `{symbol}`\n"
            f"Lower Bound: `${lower_bound:,.2f}`\n"
            f"Upper Bound: `${upper_bound:,.2f}`\n"
            f"Grid Spacing: `${grid_spacing:,.2f}`\n\n"
            f"Status: ✅ *Monitoring Active*"
        )
        
        return await self.send_message(message)
    
    async def send_trade_notification(self, filled_order: Dict[str, Any], profit_real: float) -> bool:
        """
        Send trade execution notification
        
        Args:
            filled_order: Filled order info dict
            profit_real: Real profit from this trade (0 for BUY, positive for SELL)
        
        Returns:
            bool: True if sent successfully
        """
        level = filled_order.get('level', 0)
        side = filled_order.get('side', 'UNKNOWN')
        price = filled_order.get('price', 0)
        amount = filled_order.get('amount', 0)
        
        # Determine emoji and message based on side
        if side == "SELL":
            emoji = "💰"
            profit_msg = f"Real Profit: `+${profit_real:,.2f}`\n"
        else:
            emoji = "🔵"
            profit_msg = "Position Opened (unrealized)\n"
        
        message = (
            f"{emoji} *Trade Completed*\n\n"
            f"Level {level}: `{side}` @ `${price:,.2f}` → FILLED\n"
            f"Amount: `{amount:.6f}`\n"
            f"{profit_msg}\n"
            f"✅ Order flipped automatically"
        )
        
        return await self.send_message(message)
    
    async def send_emergency_alert(self, current_price: float, threshold: float, orders_cancelled: int) -> bool:
        """
        Send emergency stop alert
        
        Args:
            current_price: Current market price
            threshold: Emergency threshold price
            orders_cancelled: Number of orders cancelled
        
        Returns:
            bool: True if sent successfully
        """
        drop_percent = ((threshold - current_price) / threshold * 100)
        
        message = (
            f"🚨 *EMERGENCY STOP TRIGGERED* 🚨\n\n"
            f"Current Price: `${current_price:,.2f}`\n"
            f"Emergency Threshold: `${threshold:,.2f}`\n"
            f"Drop: `{drop_percent:.2f}%` below threshold\n\n"
            f"*Action Taken:*\n"
            f"- Cancelled `{orders_cancelled}` orders\n"
            f"- Bot stopped\n\n"
            f"⚠️ *Manual intervention required!*\n"
            f"Review market conditions before restarting."
        )
        
        return await self.send_message(message)
    
    def __repr__(self) -> str:
        """String representation of TelegramNotifier"""
        return f"<TelegramNotifier(chat_id={self.chat_id})>"
