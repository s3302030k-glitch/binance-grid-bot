# Senior Assistant Instructions

## 🎯 Core Development Principles

### 1. Asynchronous Programming
- **Always use asynchronous code** with `ccxt.async_support`
- All exchange operations must use `async/await` pattern
- Never use synchronous ccxt library

### 2. Code Organization
- **Separation of Concerns**:
  - Exchange handling → `core/exchange.py`
  - Trading Strategy logic → `core/strategy.py`
- Keep modules focused and decoupled

### 3. Logging & Debugging
- **Use Python's logging module** instead of `print()` statements
- Configure appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Include timestamps and context in logs

### 4. Error Handling
- **Implement strict error handling** for:
  - `NetworkError` - Connection and network issues
  - `ExchangeError` - Exchange-specific errors
- Use try/except blocks with specific exception types
- Log all errors with context
- Implement retry logic where appropriate

### 5. Order Management
- **Every order MUST have a `clientOrderID`** for tracking
- Use unique identifiers for order tracking
- Maintain order state and history

### 6. Development Workflow
- **Before generating code**: 
  - Think step-by-step
  - Explain the logic in Persian (به فارسی)
  - Discuss approach and potential issues
- **Then implement** the solution

---

## 📝 Code Review Checklist

Before submitting any code, verify:
- [ ] Uses `async/await` with `ccxt.async_support`
- [ ] Logic properly separated between `exchange.py` and `strategy.py`
- [ ] Uses `logging` module (no `print()` statements)
- [ ] Has error handling for `NetworkError` and `ExchangeError`
- [ ] All orders include `clientOrderID`
- [ ] Code is explained in Persian first

---

## 🏗️ Project Structure

```
gridtrading/
├── core/
│   ├── exchange.py      # Exchange connection & API calls
│   └── strategy.py      # Trading strategy logic
├── config/              # Configuration files
├── logs/                # Log files
└── PROJECT_RULES.md     # This file
```

---

## 🤝 Workflow Agreement

### Terminal Commands
- **Assistant**: Provides terminal commands in text format
- **User**: Executes commands manually and shares the output
- **Benefit**: Better control and visibility over system operations

---

*Last updated: 2026-01-16*
