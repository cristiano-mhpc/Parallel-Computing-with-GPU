reset 
set key
set grid

GRAPHFILE="times.png"

set terminal pngcairo size 640, 480 enhanced font 'Verdana, 10'
set output GRAPHFILE

#show pointsize

set xlabel "Number of threads"
set ylabel "Run times in (us)"
set autoscale  

plot "timings.txt" using 1:2 title '10000000 trapezoid' with lines



