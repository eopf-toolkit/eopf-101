FROM quay.io/jupyter/base-notebook:python-3.12.10

RUN pip install pdm

USER root

COPY ../pyproject.toml /home/${NB_USER}/deps/pyproject.toml
RUN chown -R ${NB_UID}:${NB_GID} /home/${NB_USER}/deps

USER ${NB_UID}

WORKDIR /home/${NB_USER}/deps 

RUN pdm config python.use_venv false && \
    pdm lock && \
    pdm export --without-hashes --format requirements | pip install -r /dev/stdin

WORKDIR /home/${NB_USER}
RUN rm -rf deps
