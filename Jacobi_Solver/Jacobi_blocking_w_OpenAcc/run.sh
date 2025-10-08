#!/bin/bash

> two_threads.txt #clear the file
 
sizes=(300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500)
for size in ${sizes[@]}
do 
 ./ex_rel.x ${size}
done
