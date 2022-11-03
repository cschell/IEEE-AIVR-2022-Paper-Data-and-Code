FROM docker.io/python:3.8 AS builder

RUN apt-get update && apt install -y --no-install-recommends default-jre

RUN pip install --user pipenv

# Tell pipenv to create venv in the current directory
ENV PIPENV_VENV_IN_PROJECT=1

# Pipefile contains requests
ADD Pipfile.lock Pipfile /usr/src/

WORKDIR /usr/src

RUN /root/.local/bin/pipenv sync --clear

# due to a bug in PyTorch we must not use the latest setuptools version
# this will be fixed in future versions https://github.com/pytorch/pytorch/pull/69904
RUN /usr/src/.venv/bin/pip install --upgrade setuptools==59.5.0

# Test if everything was installed as working correctly
RUN /usr/src/.venv/bin/python -c "import pytorch_lightning"

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 AS runtime

RUN mkdir -v /usr/src/.venv

COPY --from=builder /usr/src/.venv/ /usr/src/.venv/

ENV PATH /usr/src/.venv/bin/:$PATH

RUN mkdir /workdir

WORKDIR /workdir

# set home to workdir, so that routines that write to it don't crash
ENV HOME /workdir/

# make sure packages don't ask questions during installation
ARG DEBIAN_FRONTEND=noninteractive

# avoids UTF8 issues
ENV LANG C.UTF-8

# silence warnings about git not being installed
ENV GIT_PYTHON_REFRESH quiet

RUN apt-get update && apt install -y --no-install-recommends python3.8 python3.8-distutils


RUN rm /usr/src/.venv/bin/python; ln -s `which python3.8` /usr/src/.venv/bin/python

COPY ./ /workdir/
RUN chmod a+w /workdir/ -R
