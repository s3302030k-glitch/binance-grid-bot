# 🤖 Grid Trading Bot - Binance Edition

A professional cryptocurrency grid trading bot built with Python and CCXT, specifically designed for Binance.

## 📋 Project Status

- ✅ **Phase 1**: Binance Connection & Price Fetching
- ✅ **Phase 2**: Grid Strategy Implementation
- ✅ **Phase 3**: Order Management & Execution
- ✅ **Phase 4**: Monitoring, Order Flipping & Emergency Stop (Current)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env
   ```

2. Edit `.env` and add your Binance API credentials:
   ```env
   BINANCE_API_KEY=your_actual_api_key
   BINANCE_API_SECRET=your_actual_api_secret
   USE_TESTNET=True
   ```

### 3. Get Binance Testnet API Keys

1. Visit [Binance Testnet](https://testnet.binance.vision/)
2. Login with GitHub or email
3. Generate API Key & Secret
4. Copy them to your `.env` file

### 4. Run the Bot

```bash
python main.py
```

## 📁 Project Structure

```
gridtrading/
├── core/
│   ├── __init__.py
│   ├── exchange.py      # Binance exchange manager (async)
│   └── strategy.py      # Grid trading strategy (coming soon)
├── logs/                # Log files
├── .env                 # Environment variables (keep secret!)
├── .env.example         # Environment template
├── .gitignore
├── main.py              # Entry point
├── PROJECT_RULES.md     # Development guidelines
├── README.md            # This file
└── requirements.txt     # Python dependencies
```

## 🎯 Features

### Core Features
- ✅ Asynchronous exchange operations with `ccxt.async_support`
- ✅ Binance Testnet/Mainnet switching
- ✅ ATR-based dynamic grid calculation
- ✅ Automatic order placement and tracking
- ✅ Real-time order monitoring (30-second cycles)
- ✅ Automatic order flipping when filled
- ✅ Atomic state persistence with backup
- ✅ Professional logging system
- ✅ Rate limiting protection

### 🛡️ Emergency Stop Protection
- ✅ **2% Safety Buffer**: Automatic emergency stop if price falls 2% below lower bound
- ✅ **Automatic Order Cancellation**: All orders cancelled when emergency triggered
- ✅ **Self-Check on Startup**: Bot refuses to start if emergency stop is active
- ✅ **Manual Intervention Required**: Forces you to review market conditions before resuming

## 🚨 Emergency Stop System

**توضیحات فارسی:** سیستم توقف اضطراری از ضررهای سنگین جلوگیری می‌کند.

### How It Works

1. **Continuous Monitoring**: Every 30 seconds, the bot checks current price
2. **Safety Threshold**: Calculates `emergency_price = lower_bound × 0.98` (2% below)
3. **Automatic Trigger**: If `current_price < emergency_price`:
   - 🛑 Cancels ALL open orders immediately
   - 💾 Sets `is_emergency_stopped: true` in `data/state.json`
   - 📝 Logs critical error with details
   - 🚪 Exits the bot

### After Emergency Stop

The bot will **refuse to start** until you:

1. ✅ Review current market conditions
2. ✅ Open `data/state.json` and set `"is_emergency_stopped": false`
3. ✅ Consider adjusting grid parameters if needed
4. ✅ Restart the bot with `python main.py`

**Example Log Output:**
```
🚨 EMERGENCY STOP TRIGGERED! 🚨
Current Price: $40,400.00
Emergency Threshold: $40,500.00
Lower Bound: $41,327.00
Price has fallen below safety zone!
Cancelling 10 open orders...
⚠️  BOT STOPPED - MANUAL INTERVENTION REQUIRED
```

## 📖 Development Guidelines

See [PROJECT_RULES.md](PROJECT_RULES.md) for detailed development guidelines including:
- Asynchronous programming requirements
- Code organization standards
- Error handling protocols
- Order management rules

## 🔐 Security

- ⚠️ **Never commit your `.env` file!**
- ⚠️ **Use Testnet for development and testing**
- ⚠️ **Keep your API keys secure**
- ⚠️ **Enable IP whitelist on Binance API settings**

## 🐛 Troubleshooting

### Authentication Error
- Check your API key and secret in `.env`
- Ensure API keys are from the correct network (Testnet vs Mainnet)
- Verify API key permissions

### Network Error
- Check your internet connection
- Verify Binance API status: https://www.binance.com/en/support/announcement
- Check if you're rate-limited (wait a few minutes)

### Module Not Found
```bash
pip install -r requirements.txt
```

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- [CCXT Library](https://github.com/ccxt/ccxt) - Cryptocurrency exchange trading API
- [Binance API](https://binance-docs.github.io/apidocs/) - Binance API Documentation

---

**⚠️ Disclaimer**: This bot is for educational purposes only. Trading cryptocurrencies carries risk. Always test thoroughly on Testnet before using real funds.

**🛡️ Risk Management**: The emergency stop system provides a safety net, but it cannot prevent all losses. Market gaps, network issues, or exchange outages may result in losses exceeding the 2% buffer. Never trade with funds you cannot afford to lose.
