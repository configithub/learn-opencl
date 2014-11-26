__kernel void deploy(int scalar, int size, __global int * vec) {
    
    // Get the index of the current element
    int i = get_global_id(0);

    if(i == 0) { // only do that one time
      for(int j = 0; j < size; ++j) {
        vec[j] = j * scalar + 7;
      }
    }
}
