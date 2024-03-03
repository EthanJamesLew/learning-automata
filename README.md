# learning-automata
my exploration of learning DFAs and WFAs

## Installation

Before installing via pip, make sure that you have the required 
graphviz libraries needed by automata-lib
```shell
apt install libgraphiz-dev
```

## Docker 

As the pygraphviz proved difficult to install on M1 Mac, use docker
```shell
docker build . -t learning-automata
docker run -it --rm -p 10000:8888 -v "${PWD}":/home/jovyan/work learning-automata
```
