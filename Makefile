.PHONY: setup data train sql all test lint check notebook notebooks dashboard requirements

UV ?= $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
export UV_CACHE_DIR := $(CURDIR)/.uv-cache
export UV_PYTHON_INSTALL_DIR := $(CURDIR)/.uv-python
export JUPYTER_CONFIG_DIR := $(CURDIR)/.jupyter
export JUPYTER_DATA_DIR := $(CURDIR)/.jupyter-data
export JUPYTER_RUNTIME_DIR := $(CURDIR)/.jupyter-runtime
export IPYTHONDIR := $(CURDIR)/.ipython
export MPLCONFIGDIR := $(CURDIR)/.matplotlib-cache
export MPLBACKEND := Agg

setup:
	$(UV) sync --locked --all-groups

data:
	$(UV) run credit-risk download

train:
	$(UV) run credit-risk train

sql:
	$(UV) run credit-risk sql

all:
	$(UV) run credit-risk all

test:
	$(UV) run pytest

check: lint test

lint:
	$(UV) run ruff check .

notebook:
	$(UV) run jupyter lab --no-browser --ip=127.0.0.1

notebooks:
	$(UV) run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
	$(UV) run jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_evaluation.ipynb

dashboard:
	$(UV) run streamlit run dashboard/app.py

requirements:
	$(UV) export --locked --no-dev --no-emit-project --format requirements-txt -o requirements.txt > /dev/null
	$(UV) export --locked --all-groups --no-emit-project --format requirements-txt -o requirements-dev.txt > /dev/null
