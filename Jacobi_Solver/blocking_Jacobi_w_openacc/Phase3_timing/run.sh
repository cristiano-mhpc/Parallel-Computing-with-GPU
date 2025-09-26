#!/bin/bash

process=({1..10})
sizes=(10 1200)

for j in ${sizes[@]}
do

  > timings_12h.txt #clear the file

  for i in ${process[@]}  
  do
    echo -n $i " " 
    mpirun -np ${i} ./ex_rel.x ${j}

  done > timings_12h.txt

done

 
