Deploy
=========

I wanted to check if we can bypass the costly write step to the gpu by sending a scalar value, then calculating the vector values on the gpu instead of the cpu, as this can save us the write cost when this is possible.

In this example :

- The process allocates a vector on the GPU (which is quite fast, way faster than writing)
- It sends a scalar value to the GPU
- It calculates a series of data from this scalar, to be stored on the previously allocated vector
- Finally, it reads the resulting vector (which unfortunately is very slow)

I also added the read of a single element of the vector to see if it was doable and it is, so we can also bypass the costly read step if we only want a certain value of the result vector.


