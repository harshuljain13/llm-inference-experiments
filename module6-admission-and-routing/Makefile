VENV := .venv
PYTHON ?= $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
export PYTHONPATH := $(CURDIR)

.PHONY: setup sync ssh smoke test bench replicas app

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo
	@echo "Mac:     bash setup/sync_to_lambda.sh && bash setup/ssh.sh"
	@echo "Lambda:  bash setup/lambda_setup.sh && bash setup/launch_replicas.sh && make smoke"

sync:
	bash setup/sync_to_lambda.sh

ssh:
	bash setup/ssh.sh

replicas:
	bash setup/launch_replicas.sh

smoke:
	bash setup/smoke_test.sh

test:
	$(PYTHON) -m pytest tests/

app:
	$(PYTHON) app.py

bench:
	$(PYTHON) -m bench.report --replicas http://127.0.0.1:8001,http://127.0.0.1:8002 --queries 12 --workers 4
