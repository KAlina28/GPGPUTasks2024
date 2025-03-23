#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void matrix_transpose_naive(__global const float *as, __global float *as_t, unsigned int m, unsigned int k) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i < k && j < m) {
        as_t[i * m + j] = as[j * k + i];
    }
}

__kernel void matrix_transpose_local_bad_banks(__global const float *as, __global float *as_t, unsigned int m, unsigned int k)
{
    __local float buf[TILE_SIZE][TILE_SIZE];
    unsigned int gid = get_global_id(0);
    unsigned int gjd = get_global_id(1);
    unsigned int lid = get_local_id(0);
    unsigned int ljd = get_local_id(1);

    buf[ljd][lid] = as[gjd * m + gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[(get_group_id(0) * TILE_SIZE + ljd) * k + get_group_id(1) * TILE_SIZE + lid] = buf[lid][ljd];
}

__kernel void matrix_transpose_local_good_banks(__global const float *as, __global float *as_t, unsigned int m, unsigned int k)
{
    __local float buf[TILE_SIZE + 1][TILE_SIZE];
    unsigned int gid = get_global_id(0);
    unsigned int gjd = get_global_id(1);
    unsigned int lid = get_local_id(0);
    unsigned int ljd = get_local_id(1);

    buf[ljd][lid] = as[gjd * m + gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[(get_group_id(0) * TILE_SIZE + ljd) * k + get_group_id(1) * TILE_SIZE + lid] = buf[lid][ljd];
}
