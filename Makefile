# Use bash for shell commands
SHELL := /bin/bash

conda_env:
	conda env create -f conda_env.yaml


remove_env:
	conda remove -n sam2 --all

install:
	pip install -e .[dev]

clean: remove_env
