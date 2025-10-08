#!/bin/bash 

rm -r ./data/*.dat 

mpirun -np 4 ./ex_rel.x $1
