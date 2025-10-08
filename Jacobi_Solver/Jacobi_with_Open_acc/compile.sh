#!/bin/bash 

module load openmpi/4.1.6--nvhpc--24.3

mpic++ -Minfo -acc -I./include/ main.cpp -o ex_rel.x
