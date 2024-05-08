ARG PYTHON_VERSION=3.10.13

FROM python:${PYTHON_VERSION}-slim-bullseye AS base

RUN apt-get clean  \
  && apt-get update \
  && apt-get install -y git=1:2.30.2-1+deb11u2 --no-install-recommends \
  #  psycopg2 dependencies
  && apt-get install -y libpq5=13.13-0+deb11u1 --no-install-recommends \
  && apt-get install -y libpq-dev=13.13-0+deb11u1 --no-install-recommends \
  # dependencies for building Python packages
  && apt-get install -y build-essential=12.9 --no-install-recommends \
  # for visualization purposes
  && apt-get install -y graphviz=2.42.2-5  --no-install-recommends \
  && apt-get install -y graphviz-dev  --no-install-recommends \
  # cleaning up unused files
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && rm -rf /var/lib/apt/lists/*

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV JUPYTER_PLATFORM_DIRS 1
ENV SETUPTOOLS_USE_DISTUTILS stdlib
ENV POETRY_VIRTUALENVS_CREATE false
ENV PATH="/opt/venv/bin:$PATH"

# create virtual environment
RUN python -m venv /opt/venv \
  && /opt/venv/bin/pip install wheel

FROM base AS builder

# set work directory
WORKDIR /project

# copy necessary files for deps
COPY poetry-requirements.txt poetry-requirements.txt
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

# install poetry
RUN /opt/venv/bin/pip install -r poetry-requirements.txt --no-cache-dir

# install local (dev) dependencies
RUN poetry export -f requirements.txt --output requirements.txt --with dev --without-hashes
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir

COPY mlinspect/ mlinspect/
COPY README.md README.md
COPY .pre-commit-config.yaml .pre-commit-config.yaml

RUN poetry build \
    && /opt/venv/bin/pip install dist/*.whl \
    && git init . \
    && git config --global --add safe.directory "/opt/venv" \
    && git config --global --add safe.directory "/project" \
    && pre-commit install --install-hooks

FROM builder AS final

COPY  --from=builder /opt/venv /opt/venv

RUN jupyter --paths \
    && ipython kernel install --user --name="venv"

COPY example_pipelines/ example_pipelines/
COPY demo/ demo/
COPY test/ test/
COPY codecov.yml codecov.yml

ENV PYTHONPATH /project

EXPOSE 8888
