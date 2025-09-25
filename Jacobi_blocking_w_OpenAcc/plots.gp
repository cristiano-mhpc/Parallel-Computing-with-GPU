set title "Data from File 1 and File 2"
set xlabel "X"
set ylabel "Y"

plot "two_threads.dat" using 1:2 with linespoints title "Not Parallel"
