.SUFFIXES : .o .cpp

.cpp.o:
	$(MPICXX) -acc -cudalib:curand -c $<

prog: main.o write_vector_start.o fill_random_vector.o
	$(MPICXX) -acc -cudalib:curand main.o write_vector_start.o fill_random_vector.o 
