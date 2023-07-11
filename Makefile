.PHONY: init notebook lint deploy/plan deploy/apply deploy/destroy
.DEFAULT_GOAL := help

NAMESPACE := tomdewildt
NAME := master-thesis

export PYTHONPATH=${PWD}/src

ifneq (,$(wildcard ./.env))
	include .env
	export
endif

help: ## Show this help
	@echo "${NAMESPACE}/${NAME}"
	@echo
	@fgrep -h "##" $(MAKEFILE_LIST) | \
	fgrep -v fgrep | sed -e 's/## */##/' | column -t -s##

##

init: ## Initialize the environment
	terraform -chdir=./infrastructure init
	for f in requirements/*.txt; do \
		pip install -r "$$f" --extra-index-url https://download.pytorch.org/whl/cu113; \
	done

##

notebook: ## Run the notebook server
	jupyter lab

##

lint: ## Run lint
	pylint src test

##

deploy/plan: ## Plan deployment
	terraform -chdir=./infrastructure plan

deploy/apply: ## Apply deployment
	terraform -chdir=./infrastructure apply

deploy/destroy: ## Destroy deployment
	terraform -chdir=./infrastructure destroy
