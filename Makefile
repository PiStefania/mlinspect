export DOCKER_BUILDKIT=1

help:
	@echo "Please use 'make <target>' where <target> is one of the following:"
	@echo "  run                    to run the docker image."
	@echo "  logs                   to output (follow) docker logs."
	@echo "  teardown               to teardown the docker container."
	@echo "  recreate               to teardown and run the docker container again."
	@echo "  test               	to run the tests. Use the 'target' arg if you want to limit the tests that will run."

run:
	docker compose up --build

logs:
	docker compose logs -f

teardown:
	docker compose down -v

recreate: teardown run

test:
	docker compose run ${exec_args} --rm mlinspect pytest $(target) -x


.PHONY: \
	help \
	run \
	logs \
	teardown \
	recreate \
	test