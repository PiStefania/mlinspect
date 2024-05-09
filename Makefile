export DOCKER_BUILDKIT=1

help:
	@echo "Please use 'make <target>' where <target> is one of the following:"
	@echo "  run                    to run the docker image."
	@echo "  logs                   to output (follow) docker logs."
	@echo "  teardown               to teardown the docker container."
	@echo "  recreate               to teardown and run the docker container again."
	@echo "  test               	to run the tests. Use the 'target' arg if you want to limit the tests that will run."
	@echo "  test-one               to run one test with 'test_name' argument"
	@echo "  install-pre-commit     to install pre-commit hooks on commit that run on docker."
	@echo "  pre-commit             to run the pre-commit checks."
	@echo "  lock-noupdate          to run poetry lock with no update."

run:
	docker compose up --build

logs:
	docker compose logs -f

teardown:
	docker compose down -v

recreate: teardown run

test:
	docker compose run ${exec_args} --rm mlinspect pytest $(target) -x

test-one:
	docker compose run ${exec_args} --rm mlinspect pytest -k ${test_name} -x --disable-warnings;

pre-commit:
	docker compose run ${exec_args} --rm mlinspect pre-commit run ${args}

install-pre-commit:
	echo '#!/usr/bin/env bash\nset -eo pipefail\nmake -f "$(PWD)/Makefile" pre-commit exec_args="-T"' > "$(PWD)/.git/hooks/pre-commit"
	chmod ug+x .git/hooks/pre-commit

lock-noupdate:
	docker compose run ${exec_args} --rm mlinspect sh -c "poetry lock --no-update"


.PHONY: \
	help \
	run \
	logs \
	teardown \
	recreate \
	test \
	test-one \
	pre-commit \
	install-pre-commit \
	lock-noupdate
