simple_opencl: main.c
	gcc main.c -lOpenCL -o simple_opencl

run: simple_opencl
	./simple_opencl
