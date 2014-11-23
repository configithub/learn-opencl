Sequential
=========

Add two vectors, then multiply the result by two and output the value.


Concepts introduced:
=========

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
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &ret);
```
Else the queued tasks will be executed one after the other.
