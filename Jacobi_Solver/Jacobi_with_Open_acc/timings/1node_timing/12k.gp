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
unset key
set key top right spacing 1.2 samplen 1.5 nobox

# title and labels
set title "Performance for the Jacobi solver With OpenACC Blocking calls  (N=12K, iter=1000)"
set xlabel "Number of Nodes (GPUs/Node = Ntask/Node = 4)"
set ylabel "Wall clock time(us)"

# Histogram style
set style data histogram
set style histogram rowstacked
set boxwidth 0.5 relative
set style fill solid 1.0 border -1

# === Caption (hardware info) ===
set label 1 \
"Leonardo Booster: 1 nodes consists of a single-socket 32-core Intel Xeon Platinum 8358 CPU (2.60 GHz), 512 GB DDR4\nRAM,4× NVIDIA A100 GPUs (64 GB HBM2e, NVLink 3.0, 200 GB/s), network: 2× dual-port HDR100 per node (400 Gbps/node)." \
    at graph 0.4, graph -0.20 center font ',7.5'

# Add bottom margin for caption
set bmargin 6

plot dat_file \
        u ($3/1):xtic(1)  t'comp' ,\
     "" u ($2/1):xtic(1)  t'comm'
