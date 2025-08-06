.PHONY: help init-db backfill ci test lint format run clean

help:
	@echo "Available commands:"
	@echo "  make init-db    - Initialize database and discover schema"
	@echo "  make backfill   - Backfill historical events and contexts"
	@echo "  make ci         - Build consistency index matrix"
	@echo "  make test       - Run test suite"
	@echo "  make lint       - Run linting checks"
	@echo "  make format     - Format code"
	@echo "  make run        - Start FastAPI server"
	@echo "  make clean      - Clean generated files"

init-db:
	python scripts/init_db.py

backfill:
	python scripts/backfill_events.py --symbols SPY,QQQ,NVDA --days 365

ci:
	python scripts/build_ci.py

test:
	pytest tests/ -v --cov=trading_buddy --cov-report=term-missing

lint:
	ruff check trading_buddy/
	mypy trading_buddy/

format:
	black trading_buddy/ scripts/ tests/
	isort trading_buddy/ scripts/ tests/

run:
	uvicorn trading_buddy.api.main:app --host 0.0.0.0 --port 8080 --reload

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info