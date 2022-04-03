CC = mpicc
PROGRAM_NAME = main
OBJECTS = main.o Lab4_IO.o

$(PROGRAM_NAME): main.o Lab4_IO.o
	$(CC) -g -o $(PROGRAM_NAME) $(OBJECTS) -lm

main.o: main.c Lab4_IO.h timer.h
	$(CC) -c main.c -o main.o

Lab4_IO.o: Lab4_IO.c
	gcc -c Lab4_IO.c -o Lab4_IO.o

clean:
	rm *.o $(PROGRAM_NAME)
