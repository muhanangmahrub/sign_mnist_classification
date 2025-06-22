.PHONY: run_builder run_inference install clean runner_builder runner_inference
.DEFAULT_GOAL:= runner

run_builder: install
	cd src; poetry run python runner_builder.py

run_inference: install
	cd src; poetry run python runner_inference.py

install: pyproject.toml
	poetry install

clean:
	rm -rf `find . type d -name __pycache__`

runner_builder: run_builder clean

runner_inference: run_inference clean