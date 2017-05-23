#!/bin/bash

n=$1

./Random_matrix.py $n $n

./Reduction $n $n -d -t

./Reduction_mpi $n $n -d -t

./Validation_mpi $n $n

