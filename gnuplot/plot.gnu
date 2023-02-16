set terminal postscript eps color size 60,50 solid enhanced font 'TimesNewRoman' 190 linewidth 11

set key inside right top
set style line 1 linecolor  rgb "#0072bd" linetype 1 linewidth 2 pointtype 1 pointsize 5
set style line 2 linecolor  rgb '#d95319' linetype 1 linewidth 2 pointtype 7 pointsize 5
set style line 3 linecolor  rgb '#edb120' linetype 1 linewidth 2 pointtype 3 pointsize 5
set style line 4 linecolor  rgb '#7e2f8e' linetype 1 linewidth 2 pointtype 9 pointsize 5
set style line 5 linecolor  rgb '#77ac30' linetype 1 linewidth 2 pointtype 14 pointsize 5
set style line 6 linecolor  rgb '#4dbeee' linetype 1 linewidth 2 pointtype 12 pointsize 5

set xtics nomirror
set tics scale 8

set output "results.eps"
set multiplot layout 3,1 columnsfirst
plot "loss.txt" using 0:1 title "Loss" ls 1 with linespoints
plot [][0:1] "acctrain.txt" using 0:1 title "Trainning accuracy" ls 2 with linespoints
plot [][0:1] "acctest.txt" using 0:1 title "Testing accuracy" ls 3 with linespoints


unset multiplot
