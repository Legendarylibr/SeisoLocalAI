.PHONY: ci ci-fast ci-list precheck check-changed deps lint types test test-parallel test-hardware security frontend imports fix

ci:
	python3 scripts/run_ci_local.py

ci-fast:
	python3 scripts/run_ci_local.py --fast

ci-list:
	python3 scripts/run_ci_local.py --list

precheck: ci-fast

check-changed:
	python3 scripts/run_ci_local.py --changed --skip-install

deps:
	python3 scripts/run_ci_local.py --job deps --skip-install

lint:
	python3 scripts/run_ci_local.py --job lint --skip-install

types:
	python3 scripts/run_ci_local.py --job types --skip-install

test:
	python3 scripts/run_ci_local.py --job test --skip-install

test-parallel:
	python3 scripts/run_ci_local.py --job test --pytest-workers 2 --skip-install

test-hardware:
	python3 scripts/run_ci_local.py --job test --hardware-tests --skip-install

security:
	python3 scripts/run_ci_local.py --job security --skip-install

frontend:
	python3 scripts/run_ci_local.py --job frontend --skip-install

imports:
	python3 scripts/run_ci_local.py --job imports --skip-install

fix:
	python3 scripts/run_ci_local.py --job lint --fix --skip-install

live-check:
	@echo "Requires Forge running at http://127.0.0.1:8765 (seiso forge)"
	python3 scripts/live_frontend_backend_check.py
