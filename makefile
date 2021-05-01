driver: main.c
	gcc main.c -lOpenCL -o driver

run: driver
	./driver
