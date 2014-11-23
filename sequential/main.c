#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)

void build_kernel(const char* kernel_file, const char* kernel_name, cl_context* context, cl_device_id device_id,
                        cl_program* program, cl_kernel* kernel) { 
  cl_int ret;
  // Load the add kernel source code into the array add_source_str
  FILE *fp;
  char *source_str;
  size_t source_size;

  fp = fopen(kernel_file, "r");
  if (!fp) {
    fprintf(stderr, "Failed to load kernel.\n");
    exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );

  // Create a program from the kernel source
  *program = clCreateProgramWithSource(*context, 1, 
      (const char **)&source_str, (const size_t *)&source_size, &ret);

  // Build the program
  ret = clBuildProgram(*program, 1, &device_id, NULL, NULL, NULL);
  printf("return code building kernel: %s : %d\n", kernel_name, ret);
  // Create the OpenCL kernel
  *kernel = clCreateKernel(*program, kernel_name, &ret);
}


int main(void) {
  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;   
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_ALL, 1, 
      &device_id, &ret_num_devices);

  // Create an OpenCL context
  cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
  // cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);

  // build the kernel to add two vectors
  cl_program add_program; cl_kernel add_kernel;
  build_kernel("vector_add.cl", "vector_add", &context, device_id, &add_program, &add_kernel);
  // build the kernel to multiply a vector by a scalar
  cl_program mult_program; cl_kernel mult_kernel;
  build_kernel("vector_multiply_by_scalar.cl", "vector_multiply_by_scalar", &context, device_id, &mult_program, &mult_kernel);

  // Create the two input vectors
  int i;
  const int LIST_SIZE = 8;
  int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
  int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
  for(i = 0; i < LIST_SIZE; i++) {
    A[i] = i;
    B[i] = LIST_SIZE - i;
  }

  // Create memory buffers on the device for each vector 
  cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
      LIST_SIZE * sizeof(int), NULL, &ret);
  cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
      LIST_SIZE * sizeof(int), NULL, &ret);
  cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
      LIST_SIZE * sizeof(int), NULL, &ret);

  // Copy the lists A and B to their respective memory buffers
  ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
      LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, 
      LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

  // Set the arguments of the add kernel
  ret = clSetKernelArg(add_kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  ret = clSetKernelArg(add_kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
  ret = clSetKernelArg(add_kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

  // Set the arguments of the mult kernel
  ret = clSetKernelArg(mult_kernel, 0, sizeof(cl_mem), (void *)&c_mem_obj);

  // Execute the OpenCL add_kernel on the list
  size_t global_item_size = LIST_SIZE; // Process the entire lists
  size_t local_item_size = 4; // Process in groups of 4
  // one event to do the scalar mult after the addition
  cl_event add_finished;
  // one event to read the data after the scalar multiplication
  cl_event mult_finished;
  // enqueue add operation
  ret = clEnqueueNDRangeKernel(command_queue, add_kernel, 1, NULL, 
      &global_item_size, &local_item_size, 0, NULL, &add_finished);
  // enqueue mult operation
  ret = clEnqueueNDRangeKernel(command_queue, mult_kernel, 1, NULL, 
      &global_item_size, &local_item_size, 1, &add_finished, &mult_finished);

  // Read the memory buffer C on the device to the local variable C
  int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
  ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, 
      LIST_SIZE * sizeof(int), C, 1, &mult_finished, NULL);

  // Display the result to the screen
  for(i = 0; i < LIST_SIZE; i++)
    printf("(%d + %d) * 2 = %d\n", A[i], B[i], C[i]);

  // Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(add_kernel);
  ret = clReleaseProgram(add_program);
  ret = clReleaseKernel(mult_kernel);
  ret = clReleaseProgram(mult_program);
  ret = clReleaseMemObject(a_mem_obj);
  ret = clReleaseMemObject(b_mem_obj);
  ret = clReleaseMemObject(c_mem_obj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
  free(A);
  free(B);
  free(C);
  return 0;
}

