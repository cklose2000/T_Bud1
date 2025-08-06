# Trading Buddy

An NLP-fluent, agentic trading buddy that provides council-based signal validation using DuckDB.

## Features

- **Natural Language Processing**: Parse trader text into structured hypotheses
- **Multi-Timeframe Analysis**: Validate patterns across different timeframes
- **Council-Based Voting**: Aggregate signals with Consistency Index (CI) weighting
- **Historical Backtesting**: Compute expectancy and performance metrics
- **Proactive Alerts**: Precursor detection and counterfactual analysis
- **Zero Configuration**: Auto-discovers database schema

## Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd T_Buds

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment configuration
cp .env.example .env

# Initialize database
make init-db

# Run backfill
make backfill

# Start the API
make run
```

## Architecture

```
trading_buddy/
├── core/          # Database and schema discovery
├── detectors/     # Pattern detection algorithms
├── council/       # Voting and CI calculation
├── nlp/          # Natural language parsing
├── api/          # FastAPI endpoints
├── schemas/      # Pydantic models
└── utils/        # Utility functions
```

## Usage

### Parse Natural Language Query

```bash
curl -X POST http://localhost:8080/nlp/parse \
  -H "Content-Type: application/json" \
  -d '{"text": "Watching the MACD on SPY — about to cross on the 5 after a double tap following a big drop. thoughts?"}'
```

### Get Council Vote

```bash
curl -X POST http://localhost:8080/council/vote \
  -H "Content-Type: application/json" \
  -d @hypothesis.json
```

## Development

```bash
# Run tests
make test

# Run linting
make lint

# Build CI matrix
make ci

# View logs
tail -f logs/trading_buddy.log
```

## License

MIT