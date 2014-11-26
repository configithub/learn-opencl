#include <stdio.h>
#include <stdlib.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
#define LIST_SIZE 64

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

  // build the kernel to deploy a vector on gpu memory, from a scalar value
  cl_program deploy_program; cl_kernel deploy_kernel;
  build_kernel("deploy.cl", "deploy", &context, device_id, &deploy_program, &deploy_kernel);

  int scalar = 4;
  int size = LIST_SIZE;

  // Create memory buffers on the device for the vector to deploy
  cl_mem vec_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE, 
      LIST_SIZE * sizeof(int), NULL, &ret);
  
  // Set the arguments of the mult kernel
  ret = clSetKernelArg(deploy_kernel, 0, sizeof(int), &scalar);
  ret = clSetKernelArg(deploy_kernel, 1, sizeof(int), &size);
  ret = clSetKernelArg(deploy_kernel, 2, sizeof(cl_mem), (void *)&vec_mem_obj);

  // Execute the OpenCL add_kernel on the list
  size_t global_item_size = 1; // Process the entire lists
  size_t local_item_size = 1; // Process in groups of 4
  // one event to do the scalar mult after the addition
  cl_event deploy_finished;
  // enqueue deploy operation
  ret = clEnqueueNDRangeKernel(command_queue, deploy_kernel, 1, NULL, 
      &global_item_size, &local_item_size, 0, NULL, &deploy_finished);

  // Read the memory buffer C on the device to the local variable C
  int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
  ret = clEnqueueReadBuffer(command_queue, vec_mem_obj, CL_TRUE, 0, 
      LIST_SIZE * sizeof(int), C, 1, &deploy_finished, NULL);

  int *Csingle = (int*)malloc(sizeof(int));
  ret = clEnqueueReadBuffer(command_queue, vec_mem_obj, CL_TRUE, 0, 
      sizeof(int), Csingle, 1, &deploy_finished, NULL);
  
  // Display the result to the screen
  for(int i = 0; i < LIST_SIZE; i++)
    printf("deployed on gpu %d \n", C[i]);

  printf("Csingle: %d\n", *Csingle);

  // Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(deploy_kernel);
  ret = clReleaseProgram(deploy_program);
  ret = clReleaseMemObject(vec_mem_obj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
  free(C);
  return 0;
}

