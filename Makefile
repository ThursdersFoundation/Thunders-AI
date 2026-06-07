.PHONY: install dev test lint format clean build publish docs

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=thunders_ai

lint:
	ruff check thunders_ai/ tests/
	mypy thunders_ai/

format:
	black thunders_ai/ tests/
	isort thunders_ai/ tests/
	ruff check --fix thunders_ai/ tests/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:
	python -m build

publish:
	twine upload dist/*

docs:
	cd docs && sphinx-build -b html source _build/html

docker-build:
	docker build -t thunders-ai:latest -f deployment/docker/Dockerfile .

docker-run:
	docker-compose -f deployment/docker/docker-compose.yml up -d

benchmark:
	bash scripts/benchmark.sh

train:
	bash scripts/train_model.sh

deploy:
	bash scripts/deploy.sh
