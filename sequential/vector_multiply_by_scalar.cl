__kernel void vector_multiply_by_scalar(__global int *vec) {
//__kernel void vector_multiply_by_scalar(__global int *vec, int scalar) {
    
    // Get the index of the current element
    int i = get_global_id(0);

    // Do the operation
    vec[i] *= 2;
}
