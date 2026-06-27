# Python dev environment for the original implementation.
# Build context is the python package dir (free_form_.../python), so the
# requirements file is copied by its bare name. See devcontainer.json.
FROM python:3.10.12-slim

COPY requirements.txt /requirements.txt

RUN python -m pip install --upgrade pip
RUN pip install -r /requirements.txt
# Dev tool (not an app runtime dep): drives the black/flake8 hooks declared in
# free_form_.../python/.pre-commit-config.yaml; without it the post-create
# `pre-commit install` step has nothing to run.
RUN pip install pre-commit

# git/zsh for the shell; curl + ca-certificates are required by the oh-my-zsh
# installer below (python:slim ships neither).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    zsh \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*
RUN sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended
