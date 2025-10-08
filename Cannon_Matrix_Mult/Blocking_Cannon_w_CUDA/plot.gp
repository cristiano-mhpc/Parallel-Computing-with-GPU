# plot_script.gp
set terminal pngcairo size 900,650 font ",10"
set output 'flops.png'

# === Titles and labels ===
set title "Compute Intensity of Blocking Cannon (4 GPUs × 4 Processes)" font ",12"
set xlabel "Matrix size" font ",11"
set ylabel "Performance [TeraFLOP/s]" font ",11"

# === Caption (hardware info) ===
set label 1 " Run on Leonardo Booster partition: 3456 nodes, each with a single-socket 32-core Intel Xeon Platinum 8358 CPU (2.60 GHz), 512 GB DDR4 RAM,\n4× NVIDIA A100 GPUs (64 GB HBM2e, NVLink 3.0, 200 GB/s), network: 2× dual-port HDR100 per node (400 Gbps/node)." \
    at graph 0.5, graph -0.18 center font ",9"

# Extend bottom margin to make space for caption
set bmargin 8

# === Plot the data ===
plot 'times.dat' using 1:2 with linespoints lw 2 pt 7 title 'Measured Data'

