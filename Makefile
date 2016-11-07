cc=g++

#cflags = -O0 -g3 -w -msse3 -funroll-loops -std=c++11\
	-I/home/fox/project/FoxUtil \
	-I/home/fox/Downloads/eigen-master \
	-I/home/fox/Downloads/LibN3L-2.0-master \
	-Ibasic -Imodel
	
cflags = -O3 -w -msse3 -funroll-loops  -std=c++11\
	-I/home/fox/project/FoxUtil \
	-I/home/fox/Downloads/eigen-master \
	-I/home/fox/Downloads/LibN3L-2.0-master \
	-static-libgcc -static-libstdc++ \
	-Ibasic -Imodel

libs = -lm -Wl,-rpath,./ \
 
all: bb3 cem gm

bb3: bb3.cpp NNbb3.h 
	$(cc) -o bb3 bb3.cpp $(cflags) $(libs)
	
cem: cem.cpp NNcem.h
	$(cc) -o cem cem.cpp $(cflags) $(libs)

gm: gm.cpp NNgm.h
	$(cc) -o gm gm.cpp $(cflags) $(libs)




clean:
	rm -rf *.o
	rm -rf bb3
	rm -rf cem
	rm -rf gm

