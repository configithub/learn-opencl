#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MAX_SOURCE_SIZE (0x100000)
//#define LIST_SIZE 4096
#define LIST_SIZE 4194304
#define LOCAL_ITEM_SIZE 1024
#define MAX_VALUE 128
#define NB_TEST 10

// result arrays
int gpu[NB_TEST];

int gpu_add[NB_TEST];
int gpu_add_alloc[NB_TEST];
int gpu_add_write[NB_TEST];
int gpu_add_calc[NB_TEST];
int gpu_add_read[NB_TEST];
int gpu_add_release[NB_TEST];

int gpu_mult[NB_TEST];
int gpu_mult_alloc[NB_TEST];
int gpu_mult_write[NB_TEST];
int gpu_mult_calc[NB_TEST];
int gpu_mult_read[NB_TEST];
int gpu_mult_release[NB_TEST];

int gpu_div[NB_TEST];
int gpu_div_alloc[NB_TEST];
int gpu_div_write[NB_TEST];
int gpu_div_calc[NB_TEST];
int gpu_div_read[NB_TEST];
int gpu_div_release[NB_TEST];

int cpu[NB_TEST];

int cpu_add[NB_TEST];
int cpu_mult[NB_TEST];
int cpu_div[NB_TEST];

static struct timeval tm_global;
static struct timeval tm_local;
static struct timeval tm_atom;

static inline void start(struct timeval* tm) {
    gettimeofday(tm, NULL);
}

static inline void stop(struct timeval* tm, int* res, int j) {
    struct timeval tm2;
    gettimeofday(&tm2, NULL);
    unsigned long long t = 1000000 * (tm2.tv_sec - tm->tv_sec) 
                              + (tm2.tv_usec - tm->tv_usec) ;
    res[j] = t;
}

static inline void stop_print(struct timeval* tm) {
    struct timeval tm2;
    gettimeofday(&tm2, NULL);
    unsigned long long t = 1000000 * (tm2.tv_sec - tm->tv_sec)
                               + (tm2.tv_usec - tm->tv_usec) ;
    printf("%llu us\n", t);
}

double avg(int* res) {
  int sum =0;
  for(int j =0; j < NB_TEST; j++) {
    sum += res[j];
  }
  return (double)sum / NB_TEST;
}


void build_kernel(const char* kernel_file, const char* kernel_name,
                       cl_context* context, cl_device_id device_id,
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

int vector_add_on_cpu(int* A, int* B) {
  int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
  for (int i = 0; i < LIST_SIZE; i++) {
    C[i] = A[i] + B[i];
  }
  // printf("avg add cpu: %f\n", avg(C));
  free(C);
  return 0;
}
int vector_mult_on_cpu(int* A, int* B) {
  int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
  for (int i = 0; i < LIST_SIZE; i++) {
    C[i] = A[i] * B[i];
  }
  // printf("avg mult cpu: %f\n", avg(C));
  free(C);
  return 0;
}
int vector_div_on_cpu(int* A, int* B) {
  int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
  for (int i = 0; i < LIST_SIZE; i++) {
    C[i] = A[i] / B[i];
  }
  // printf("avg div cpu: %f\n", avg(C));
  free(C);
  return 0;
}


int vector_operation_on_gpu(cl_context* context, cl_command_queue* command_queue,
                             cl_kernel* kernel, int* A, int* B, int* res,
                    int* alloc, int* write, int* calc, int* read, int* release, int j) {
  /*
  for (int i = 0; i < LIST_SIZE; i++) {
    printf("A[%d] = %d\n", i, A[i]);
  }
  for (int i = 0; i < LIST_SIZE; i++) {
    printf("B[%d] = %d\n", i, B[i]);
  }*/
  start(&tm_local);
  start(&tm_atom);
  cl_int ret;
  // Create memory buffers on the device for each vector 
  cl_mem a_mem_obj = clCreateBuffer(*context, CL_MEM_READ_ONLY, 
      LIST_SIZE * sizeof(int), NULL, &ret);
  cl_mem b_mem_obj = clCreateBuffer(*context, CL_MEM_READ_ONLY,
      LIST_SIZE * sizeof(int), NULL, &ret);

  cl_mem c_mem_obj = clCreateBuffer(*context, CL_MEM_READ_WRITE, 
      LIST_SIZE * sizeof(int), NULL, &ret);
  stop(&tm_atom, alloc, j);

  start(&tm_atom);
  // Copy the lists A and B to their respective memory buffers
  ret = clEnqueueWriteBuffer(*command_queue, a_mem_obj, CL_TRUE, 0,
      LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(*command_queue, b_mem_obj, CL_TRUE, 0, 
      LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

  // Set the arguments of the kernel
  ret = clSetKernelArg(*kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  ret = clSetKernelArg(*kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
  ret = clSetKernelArg(*kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
  stop(&tm_atom, write, j);

  // Execute the OpenCL add_kernel on the list
  size_t global_item_size = LIST_SIZE; // Process the entire lists
  size_t local_item_size = LOCAL_ITEM_SIZE; // Process in groups of 4

  start(&tm_atom);
  ret = clEnqueueNDRangeKernel(*command_queue, *kernel, 1, NULL, 
      &global_item_size, &local_item_size, 0, NULL, NULL);
  stop(&tm_atom, calc, j);

  // Read the memory buffer C on the device to the local variable C
  int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
  start(&tm_atom);
  ret = clEnqueueReadBuffer(*command_queue, c_mem_obj, CL_TRUE, 0, 
      LIST_SIZE * sizeof(int), C, 0, NULL, NULL);
  stop(&tm_atom, read, j);
  
  /*
  for (int i = 0; i < LIST_SIZE; i++) {
    printf("C[%d] = %d\n", i, C[i]);
  }*/
  
  // cleanup
  start(&tm_atom);
  ret = clReleaseMemObject(a_mem_obj);
  ret = clReleaseMemObject(b_mem_obj);
  ret = clReleaseMemObject(c_mem_obj);
  stop(&tm_atom, release, j);
  //printf("avg res gpu: %f\n", avg(C));
  free(C);
  stop(&tm_local, res, j);
  return 0;

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

  // build the kernel to add two vectors
  cl_program add_program; cl_kernel add_kernel;
  build_kernel("vector_add.cl", "vector_add", &context, 
                        device_id, &add_program, &add_kernel);
  // build the kernel to multiply a vector by a scalar
  cl_program mult_program; cl_kernel mult_kernel;
  build_kernel("vector_multiply_by_scalar.cl", "vector_multiply_by_scalar", 
                    &context, device_id, &mult_program, &mult_kernel);
  // build the kernel to multiply a vector by a vector
  cl_program multv_program; cl_kernel multv_kernel;
  build_kernel("vector_mult.cl", "vector_mult", &context, 
                            device_id, &multv_program, &multv_kernel);
  // build the kernel to divide a vector by another
  cl_program divide_program; cl_kernel divide_kernel;
  build_kernel("vector_divide.cl", "vector_divide", &context, 
                            device_id, &divide_program, &divide_kernel);

  // seed random
  srand(time(NULL));

  for (int j = 0; j < NB_TEST; j++) {
    
    // Create the two input vectors
    int i;
    int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
    int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
    for(i = 0; i < LIST_SIZE; i++) {
      A[i] = rand()%(MAX_VALUE*MAX_VALUE)+1;
      B[i] = rand()%MAX_VALUE+1;
    }
    int scalar = rand();
    scalar++;

    // test on gpu
    start(&tm_global);
    {
      vector_operation_on_gpu(&context, &command_queue, &add_kernel, A, B, gpu_add,
                              gpu_add_alloc, gpu_add_write, gpu_add_calc, gpu_add_read, gpu_add_release, j);
      vector_operation_on_gpu(&context, &command_queue, &multv_kernel, A, B, gpu_mult,
                              gpu_mult_alloc, gpu_mult_write, gpu_mult_calc, gpu_mult_read, gpu_mult_release, j);
      vector_operation_on_gpu(&context, &command_queue, &divide_kernel, A, B, gpu_div,
                              gpu_div_alloc, gpu_div_write, gpu_div_calc, gpu_div_read, gpu_div_release, j);
    }
    stop(&tm_global, gpu, j);
    // test on cpu
    start(&tm_global); 
    {
      start(&tm_local);
      vector_add_on_cpu(A, B);
      stop(&tm_local, cpu_add, j);

      start(&tm_local);
      vector_mult_on_cpu(A, B);
      stop(&tm_local, cpu_mult, j);

      start(&tm_local);
      vector_div_on_cpu(A, B);
      stop(&tm_local, cpu_div, j);
    }
    stop(&tm_global, cpu, j);
    
    // cleanup
    free(A);
    free(B);
  }

  printf("\n");
  printf("  avg time cpu_add %f\n", avg(cpu_add));
  printf("  avg time cpu_mult %f\n", avg(cpu_mult));
  printf("  avg time cpu_div %f\n", avg(cpu_div));
  printf("\n");
  printf("avg time cpu %f\n", avg(cpu));
  printf("\n");
  printf("\n");
  printf("    avg time gpu_add_alloc %f\n", avg(gpu_add_alloc));
  printf("    avg time gpu_add_write %f\n", avg(gpu_add_write));
  printf("    avg time gpu_add_calc %f\n", avg(gpu_add_calc));
  printf("    avg time gpu_add_read %f\n", avg(gpu_add_read));
  printf("    avg time gpu_add_release %f\n", avg(gpu_add_release));
  printf("  avg time gpu_add %f\n", avg(gpu_add));
  printf("\n");
  printf("    avg time gpu_mult_alloc %f\n", avg(gpu_mult_alloc));
  printf("    avg time gpu_mult_write %f\n", avg(gpu_mult_write));
  printf("    avg time gpu_mult_calc %f\n", avg(gpu_mult_calc));
  printf("    avg time gpu_mult_read %f\n", avg(gpu_mult_read));
  printf("    avg time gpu_mult_release %f\n", avg(gpu_mult_release));
  printf("  avg time gpu_mult %f\n", avg(gpu_mult));
  printf("\n");
  printf("    avg time gpu_div_alloc %f\n", avg(gpu_div_alloc));
  printf("    avg time gpu_div_write %f\n", avg(gpu_div_write));
  printf("    avg time gpu_div_calc %f\n", avg(gpu_div_calc));
  printf("    avg time gpu_div_read %f\n", avg(gpu_div_read));
  printf("    avg time gpu_div_release %f\n", avg(gpu_div_release));
  printf("  avg time gpu_div %f\n", avg(gpu_div));
  printf("\n");
  printf("avg time gpu %f\n", avg(gpu));
  printf("\n");

  // Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(add_kernel);
  ret = clReleaseProgram(add_program);
  ret = clReleaseKernel(mult_kernel);
  ret = clReleaseProgram(mult_program);
  ret = clReleaseKernel(multv_kernel);
  ret = clReleaseProgram(multv_program);
  ret = clReleaseKernel(divide_kernel);
  ret = clReleaseProgram(divide_program);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);
  return 0;
}

