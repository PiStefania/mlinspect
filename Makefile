export DOCKER_BUILDKIT=1

help:
	@echo "Please use 'make <target>' where <target> is one of the following:"
	@echo "  build                  to build the docker container."
	@echo "  run                    to run the docker container."
	@echo "  logs                   to output (follow) docker logs."
	@echo "  teardown               to teardown the docker container."
	@echo "  recreate               to teardown and run the docker container again."
	@echo "  test               	to run the tests. Use the 'target' arg if you want to limit the tests that will run."

build:
	docker build -t mlinspect .

run: build
	docker run mlinspect

logs:
	docker logs mlinspect

teardown:
	docker stop mlinspect
	docker rm mlinspect

recreate: teardown run

test:
	docker exec -it mlinspect pytest $(target)


.PHONY: \
	help \
	build \
	run \
	logs \
	teardown \
	recreate \
	test \