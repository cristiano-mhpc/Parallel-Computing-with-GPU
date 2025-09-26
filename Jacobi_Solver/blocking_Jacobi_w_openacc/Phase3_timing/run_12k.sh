#!/bin/bash

process=({1..10})
sizes=(10 12000)

for j in ${sizes[@]}
do

  > timings_12k.txt #clear the file

  for i in ${process[@]}  
  do
    echo -n $i " " 
    mpirun -np ${i} ./ex_rel.x ${j}

  done > timings_12k.txt

done

 
