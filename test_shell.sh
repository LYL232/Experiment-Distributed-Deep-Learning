#!/bin/bash -e
script_abs=$(readlink -f "$0")
script_dir=$(dirname "$script_abs")

mpirun -n 4 python "$script_dir"/src/py/ddl/examples/data_parallelism.py --epoch=2 && \
mpirun -n 6 python "$script_dir"/src/py/ddl/examples/pipeline_sequential_2stages.py --epoch=2 && \
mpirun -n 6 python "$script_dir"/src/py/ddl/examples/pipeline_sequential_3stages.py --epoch=2 && \
mpirun -n 8 python "$script_dir"/src/py/ddl/examples/pipeline_network_like_stages.py --epoch=2
