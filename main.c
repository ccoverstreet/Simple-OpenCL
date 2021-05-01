#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>

#define N 1000 // Size of arrays used

char *loadCLSource(const char *filename) {
	// User must free memory used
	FILE *fp = NULL;
	fp = fopen(filename, "r");
	if (fp == NULL) {
		printf("Unable to load program source file\n");
		exit(1);
	}

	fseek(fp, 0, SEEK_END);
	long int length = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	char *buf = malloc(sizeof(char) * (length + 1));
	buf[length] = 0;

	fread(buf, 1, length, fp);

	return buf;
}

int main() {
	//cl_platform_id platform;
	//clGetPlatformIDs(1, &platform, NULL);

	cl_uint num_dev_returned;
	cl_device_id devices[2];
	cl_int err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &devices[0],  &num_dev_returned);
	if (err) {
		printf("Unable to get GPU\n");
		return 0;
	}

	cl_context context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, NULL);
	
	cl_command_queue queue = clCreateCommandQueueWithProperties(context, devices[0], NULL, NULL);

	// Create buffers for output
	cl_mem in_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * N, NULL, &err);
	cl_mem out_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * N, NULL, &err);
	cl_mem red_out_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint), NULL, &err);

	// Create program and kernel
	// First load simple.cl file
	char *source = loadCLSource("./simple.cl");
	printf("Kernel Source:\n");
	printf("========================================\n");
	printf("%s\n\n", source);
	printf("========================================\n");
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &err);
	free(source);
	if (err) {
		printf("Create Program err: %d\n", err);
		exit(1);
	}

	err = clBuildProgram(program, 1, devices, " -cl-std=CL2.0", NULL, NULL);
	if (err) {
		printf("Build Program err: %d\n", err);
		exit(1);
	}

	// Create some kernels
	cl_kernel kernel_1 = clCreateKernel(program, "copy_test", &err);
	if (err) {
		printf("Create Kernel err: %d\n", err);
		exit(1);
	}
	clSetKernelArg(kernel_1, 0, sizeof(cl_mem), (void*)&in_buffer);
	clSetKernelArg(kernel_1, 1, sizeof(cl_mem), (void*)&out_buffer);

	cl_kernel kernel_2 = clCreateKernel(program, "simple_reduce", &err);
	if (err) {
		printf("Create Kernel err: %d\n", err);
		exit(1);
	}
	clSetKernelArg(kernel_2, 0, sizeof(cl_mem), (void*)&in_buffer);
	clSetKernelArg(kernel_2, 1, sizeof(cl_mem), (void*)&red_out_buffer);

	// Work sizes
	size_t global_size = N;
	size_t local_size = N / 4;

	// Host data storage
	cl_float data[N] = {2.3};
	cl_float out[N] = {0};
	cl_uint red_out = 0;

	// Initialize input data
	for (int i = 0; i < N; i++) {
		data[i] = i;	
	}

	clEnqueueWriteBuffer(queue, in_buffer, CL_TRUE, 0, sizeof(cl_float) * N, data, 0, NULL, NULL);

	clEnqueueNDRangeKernel(queue, kernel_1, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
	clEnqueueNDRangeKernel(queue, kernel_2, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

	clFinish(queue); // Forces synchronization

	clEnqueueReadBuffer(queue, out_buffer, CL_TRUE, 0, sizeof(cl_float) * N, out, 0, NULL, NULL);

	clEnqueueReadBuffer(queue,
			red_out_buffer,
			CL_TRUE,
			0,
			sizeof(cl_uint),
			&red_out,
			0, NULL, NULL);
	
	for (int i = 0; i < N; i++) {
		printf("%f\n", out[i]);
	}

	printf("Reduction Output: %f\n", red_out / (float)1000);

	return 0;
}
