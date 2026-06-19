.PHONY: ci ci-fast ci-list precheck deps lint types test security frontend imports fix

ci:
	python3 scripts/run_ci_local.py

ci-fast:
	python3 scripts/run_ci_local.py --fast

ci-list:
	python3 scripts/run_ci_local.py --list

precheck: ci-fast

deps:
	python3 scripts/run_ci_local.py --job deps --skip-install

lint:
	python3 scripts/run_ci_local.py --job lint --skip-install

types:
	python3 scripts/run_ci_local.py --job types --skip-install

test:
	python3 scripts/run_ci_local.py --job test --skip-install

security:
	python3 scripts/run_ci_local.py --job security --skip-install

frontend:
	python3 scripts/run_ci_local.py --job frontend --skip-install

imports:
	python3 scripts/run_ci_local.py --job imports --skip-install

fix:
	python3 scripts/run_ci_local.py --job lint --fix --skip-install
