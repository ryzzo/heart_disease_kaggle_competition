.PHONY: format link test train tune register

format:
	black src tests
	ruff check src tests --fix

lint:
	ruff check src tests

test:
	pytest -q

train:
	python -m src.train

tune:
	python -m src.tune

register:
	python -m src.registry