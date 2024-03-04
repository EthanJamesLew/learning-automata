# Our modification of Jupyter Docker for jupytext support
FROM jupyter/datascience-notebook

RUN pip install jupytext

USER root
RUN apt update && apt install -y graphviz libgraphviz-dev

USER ${NB_UID}
RUN pip install 'automata-lib[visual]'

