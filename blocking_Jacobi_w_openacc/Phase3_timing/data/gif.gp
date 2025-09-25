set terminal gif animate delay 5
set output 'jacobi.gif'


set size square
unset colorbox
set palette rgb 33,13,10
set yr [] reverse #otherwise it flips the image due to default axis orientation


#WRITE A LOOP HERE TO GO OVER THE FILES AND PLOT THEM ALL. HINT: USE MATRIX WITH IMAGE (GOOGLE DETAILS)

list = system('ls *.dat')
do for [file in list]{
	plot file matrix with image
} 




#plot for [file in system("ls *.dat")] file matrix with image


reset 
set output
set terminal qt #just to return back to normal output terminal; MIGHT NOT BE QT ON YOUR SYSTEM 
 
 
