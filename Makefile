docs:
	sphinx-apidoc -f -o docs/source/ .
	cd docs && make html
	@echo "Check out file://docs/html/index.html for the generated docs."

lint-fix:
	@echo "Running automated lint fixes"
	python -m isort .
	python -m black tests/* wicker/*

lint:
	@echo "Running wicker lints"
	# Ignoring slice errors E203: https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html#slices
	python -m flake8 --ignore E203,W503
	python -m isort . --check --diff
	python -m black tests/* wicker/* --check

type-check:
	@echo "Running wicker type checking with mypy"
	python -m mypy

test:
	@echo "Running wicker tests"
	python -m unittest discover

.PHONY: black docs lint type-check test
