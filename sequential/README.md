Sequential
=========

Add two vectors, then multiply the result by two and output the value.


Concepts introduced:
=========

##Send scalar to the GPU

This can be done directly by giving the scalar value as an argument to the kernel :
```
  ret = clSetKernelArg(mult_kernel, 1, sizeof(int), &scalar);
```

##Send vector to the GPU

This has to be carried out in 3 steps : 

- Allocation on the standard memory
- Create memory buffer on GPU memory, represented by a cl_mem virtual object :
```
  cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, 
      LIST_SIZE * sizeof(int), NULL, &ret);
```
- Write to buffer :
```
  ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
      LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
```

The vector thus allocated on the GPU memory can then be given to a kernel as an argument : 
```
  ret = clSetKernelArg(add_kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
```

##Cl event :

These objects can be given to clEnqueue functions to make them wait for other functions to finish before being called on the GPU.

``` 
ret = clEnqueueNDRangeKernel(command_queue, mult_kernel, 1, NULL, 
    &global_item_size, &local_item_size, 1, &add_finished, &mult_finished);
``` 

This will wait for the even "add_finished" and fire "mult_finished" when over
It will wait for exactly one event.


##Data parallel mode : 

The operations will be parallelised along the data.
Kernels must be written using get_global_id to get the index of the data (which is also the id of the thread the is working) on which the operations will be carried on.
This is the case in this example.
Works best when a lot of data are being treated with the same task.

Data parallel kernel executions are queued with this function :
```
clEnqueueNDRangeKernel
```

##Task parallel mode : 

The operations will be parallelised between different tasks, and not along the data.
Kernels can be written in a more classical manner, and performance are better if many kernels doing different tasks can work at the same time.

Task parallel kernel executions are queued with this function :
```
clEnqueueTask
```

To use this mode, out of order execution must be activated in the command queue : 
```
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 
                        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);
```
Else the queued tasks will be executed one after the other.
