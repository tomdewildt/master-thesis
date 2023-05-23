.PHONY: init notebook lint
.DEFAULT_GOAL := help

NAMESPACE := tomdewildt
NAME := master-thesis

export PYTHONPATH=${PWD}/src

help: ## Show this help
	@echo "${NAMESPACE}/${NAME}"
	@echo
	@fgrep -h "##" $(MAKEFILE_LIST) | \
	fgrep -v fgrep | sed -e 's/## */##/' | column -t -s##

##

init: ## Initialize the environment
	for f in requirements/*.txt; do \
		pip install -r "$$f"; \
	done

##

notebook: ## Run the notebook server
	jupyter lab

##

lint: ## Run lint
	pylint src test
