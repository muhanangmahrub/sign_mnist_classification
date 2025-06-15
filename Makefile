.PHONY: run install clean runner
.DEFAULT_GOAL:= runner

run: install
	cd src; poetry run python runner.py

install: pyproject.toml
	poetry install

clean:
	rm -rf `find . type d -name __pycache__`

runner: run clean