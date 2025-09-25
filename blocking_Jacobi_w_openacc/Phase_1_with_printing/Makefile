# This is a comment line
#CC=mpic++
CC=nvc++

# This is a folder to look for header files
INCDIR=./include
DBUG=./build/debug
RELEASE=./build/release

# CFLAGS will be the options passed to the compiler.
#CFLAGS= -c -I$(INCDIR) -O3 
CFLAGS=  -I$(INCDIR) -O3 -acc
#LDFLAGS= -ltbb -pthrea
LDFLAGS= -acc

ROBJECTS= $(RELEASE)/main.o #$(RELEASE)/matMul.o
DBUGOBJECTS= $(DBUG)/main.o #$(DBUG)/matMul.o

DBUGFLAGS=$(CFLAGS) -Wall -g -fsanitize=undefined -fsanitize=undefined -fsanitize=leak 
RFLAGS=$(CFLAGS)

release:$(ROBJECTS)
	$(CC) $(ROBJECTS) -o ex_rel.x $(LDFLAGS) 

debug:$(DBUGOBJECTS)
	$(CC) $(DBUGOBJECTS) -o ex_dgb.x $(LDFLAGS) -fsanitize=undefined -fsanitize=undefined -fsanitize=leak

$(RELEASE)/main.o: main.cpp
	$(CC) $(RFLAGS) main.cpp -o $@

$(DBUG)/main.o: main.cpp
	$(CC) $(DBUGFLAGS) main.cpp -o $@


clean:
	rm -rf $(DBUG)/*.o $(RELEASE)/*.o *x
