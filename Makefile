app-dir = bot


.PHONY: build
build:
	uv run -m bot


.PHONY: format
format:
	echo "Running ruff check with --fix..."
	uv run ruff check --config pyproject.toml --fix --unsafe-fixes $(app-dir)

	echo "Running ruff..."
	uv run ruff format --config pyproject.toml $(app-dir)

	echo "Running isort..."
	uv run isort --settings-file pyproject.toml $(app-dir)


.PHONY: tuner-gui
tuner-gui:
	uv run --extra tuner bot/utils/on_review_tuner.py test4.jpg
