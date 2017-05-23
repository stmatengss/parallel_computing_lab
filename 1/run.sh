#!/bin/bash

mpirun -np $1 ./Reduction_mpi $2 $2 -d -t
