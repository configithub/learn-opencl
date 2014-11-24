Benchmark
=========

Doing addition, multiplication and division on two random vectors, comparison between CPU and GPU process times.
Each operation is done only once after the vector data is sent on the GPU memory.

###Setup of the benchmark :

- Test repetitions : 10
- Vector size : 4194304
- Local item size : 1024

###Typical results : 

```

  avg time cpu_add 44695.1
  avg time cpu_mult 45790.4
  avg time cpu_div 68664.6

avg time cpu 159151.5


    avg time gpu_add_alloc 17.4
    avg time gpu_add_write 23014.
    avg time gpu_add_calc 440.5
    avg time gpu_add_read 16950.9
    avg time gpu_add_release 627.9
  avg time gpu_add 41257.1

    avg time gpu_mult_alloc 11.1
    avg time gpu_mult_write 22798.3
    avg time gpu_mult_calc 427.4
    avg time gpu_mult_read 14922.4
    avg time gpu_mult_release 616.5
  avg time gpu_mult 38785.3

    avg time gpu_div_alloc 10.5
    avg time gpu_div_write 22795.4
    avg time gpu_div_calc 414.2
    avg time gpu_div_read 13884.6
    avg time gpu_div_release 604.9
  avg time gpu_div 37716.1

avg time gpu 117760.9

```

The cost to send and read data on the GPU seems crippling for unitary operations.
