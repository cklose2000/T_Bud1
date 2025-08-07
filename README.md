# Trading Buddy

An NLP-fluent, agentic trading buddy that provides council-based signal validation using DuckDB.

## Features

- **Natural Language Processing**: Parse trader text into structured hypotheses
- **Multi-Timeframe Analysis**: Validate patterns across different timeframes
- **Council-Based Voting**: Aggregate signals with Consistency Index (CI) weighting
- **Historical Backtesting**: Compute expectancy and performance metrics
- **Proactive Alerts**: Precursor detection and counterfactual analysis
- **Zero Configuration**: Auto-discovers database schema

## Recent Updates

### PR8: Statistical Rigor & Uncertainty Quantification (2024-01)
- **Bootstrap Confidence Intervals**: 95% CIs for lift metrics with automatic method selection (standard/median/block)
- **FDR Control**: Benjamini-Hochberg procedure to limit false discoveries to 10%
- **Robustness Features**:
  - Automatic skew detection switches to median bootstrap for heavy-tailed returns
  - Block bootstrap for time-dependent data preserves serial correlation
  - Effect-size based power calculations replace naive linear scaling
  - Minimum CI width enforcement (1.9×SE) prevents overconfidence on thin samples
- **Operational Hardening**:
  - Chunked processing with job ledger for restart resilience
  - Performance monitoring with 85% runtime threshold
  - Code version hashing for automatic cache invalidation
- **Trade Gates**: Strict quality controls (n_eff≥60, stability≥0.4, CI excludes zero, FDR pass)
- **Monitoring**: Enhanced `/metrics/summary` with uncertainty metrics and gate distributions

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