# plot_script.gp
set terminal pngcairo size 800,600
set output 'flops.png'

# Set log scale for both axes
#set logscale x
#set logscale y

# Set titles and labels
set title "Compute intensity of Blocking Canon with 4 GPUs and 4 Process"
set ylabel "The TeraFlops/s "
set xlabel "Matrix size"

# Plot the data
plot 'times.dat' using 1:2 with linespoints title 'Data'

