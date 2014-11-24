Benchmark
=========

Doing addition, multiplication and division on two random vectors, comparison between CPU and GPU process times.
Each operation is done only once after the vector data is sent on the GPU memory.

#Typical results : 

```

  avg time cpu_add 11.771
  avg time cpu_mult 11.709
  avg time cpu_div 18.155

avg time cpu 41.757


    avg time gpu_add_alloc 40.65
    avg time gpu_add_calc 5.816
    avg time gpu_add_read 24.305
    avg time gpu_add_release 5.507
  avg time gpu_add 76.999

    avg time gpu_mult_alloc 40.176
    avg time gpu_mult_calc 5.565
    avg time gpu_mult_read 23.771
    avg time gpu_mult_release 5.426
  avg time gpu_mult 75.63

    avg time gpu_div_alloc 40.098
    avg time gpu_div_calc 5.608
    avg time gpu_div_read 24.285
    avg time gpu_div_release 5.458
  avg time gpu_div 76.164

avg time gpu 228.984

```

The cost to send and read data on the GPU seems crippling for unitary operations.
