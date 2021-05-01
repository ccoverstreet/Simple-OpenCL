#include <CL/cl.h>

#include <stdio.h>
#include <unistd.h>

#define N 32

void checkErr(cl_int err, const char *name) {
	if (err != CL_SUCCESS) {
		printf("%d\n", err);
	}
}

const char *source =
"__kernel void memset_test(__global float *in, __global float *dst)\n"
"{\n"
"	if (get_global_id(0) <= 0 || get_global_id(0) >= get_global_size(0) - 1) { dst[get_global_id(0)] = in[get_global_id(0)]; return;}"
"	dst[get_global_id(0)] = (in[get_global_id(0) + 1] + in[get_global_id(0) - 1]) / 2;\n"
"}\n";

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

	cl_mem in_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_float) * N, NULL, &err);
	cl_mem out_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * N, NULL, &err);

	// Create program and kernel
	cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
	printf("Create Program err: %d\n", err);

	err = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	printf("Build Program err: %d\n", err);

	cl_kernel kernel = clCreateKernel(program, "memset_test", &err);
	printf("Create Kernel err: %d\n", err);
	clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&in_buffer);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&out_buffer);

	size_t global_size = N;
	size_t local_size = N;

	cl_float data[N] = {2.3};
	cl_float out[N] = {0};
	for (int i = 0; i < N; i++) {
		data[i] = i * 0.2;	
	}

	clEnqueueWriteBuffer(queue, in_buffer, CL_TRUE, 0, sizeof(cl_float) * N, data, 0, NULL, NULL);

	clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);

	clEnqueueReadBuffer(queue, out_buffer, CL_TRUE, 0, sizeof(cl_float) * N, out, 0, NULL, NULL);
	//clEnqueueReadBuffer(queue, in_buffer, CL_TRUE, sizeof(cl_float), sizeof(cl_float) * 9, out, 0, NULL, NULL);
	
	for (int i = 0; i < N; i++) {
		printf("%f\n", out[i]);
	}

	return 0;
}
