#!/usr/bin/gnuplot

dat_file="./12k.txt"
dat_plot="fig2.png"

reset

set terminal pngcairo size 640, 480 enhanced font 'Verdana, 10'
set output dat_plot

# okabe and ito color blind palette
set style line 1 lt 1 lc rgb '#000000' lw 2
set style line 2 lt 1 lc rgb '#009E73' lw 2
set style line 3 lt 1 lc rgb '#0072B2' lw 2
set style line 4 lt 1 lc rgb '#56B4E9' lw 2
set style line 5 lt 1 lc rgb '#F0E442' lw 2
set style line 6 lt 1 lc rgb '#E69F00' lw 2
set style line 7 lt 1 lc rgb '#D55E00' lw 2
set style line 8 lt 1 lc rgb '#CC79A7' lw 2

# axes
#set style line 11 lc rgb '#808080' lt 1
#set border 3 front ls 11
#set tics nomirror out scale 0.75

#set xrange[-6.0:6.0]
#set yrange[-2.0:2.0]
#set logscale y

# grid
set style line 12 lc rgb'#808080' lt 0 lw 1
set grid back ls 12
set grid xtics ytics mxtics

# controlling the legend (key): 
set nokey
set key below vertical maxrows 1 samplen 1 spacing 1.5 width -0.5

# title and labels
set title "Executions times Jacobi Blocking With OpenACC  (N=12K, iter=1000)"
set xlabel "Nodes(GPUs/Node = Ntask/Node = 4)"
set ylabel "time in [us]"

# Histogram style
set style data histogram
set style histogram rowstacked
set boxwidth 0.5 relative
set style fill solid 1.0 border -1

plot dat_file \
        u ($3/1):xtic(1)  t'comp' ,\
     "" u ($2/1):xtic(1)  t'comm'
