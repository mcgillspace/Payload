tetra: tetra.o linalg.o
	gcc tetra.o linalg.o -lm

tetra.o: tetra.c
	gcc -c tetra.c

linalg.o: linalg.c
	gcc -c linalg.c

clean:
	rm *.o a.out