#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

// TILE_SIZE и WORK_PER_THREAD задаются через поле 'defines' в кернел конфиге

__kernel void matrix_multiplication_naive(__global const float *a, __global const float *b, __global float *c, unsigned int m, unsigned int k, unsigned int n)
{
    unsigned int gid_row = get_global_id(0);
    unsigned int gjd_col = get_global_id(1);

    if (gid_row >= m || gjd_col >= n) {
        return;
    }
    float ans = 0.0f;
    for (int i = 0; i < k; i++) {
        ans += a[gjd_col * k + i]*b[n*i + gid_row];
    }
    c[gjd_col * n + gid_row] = ans;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local(__global const float *a, __global const float *b, __global float *c, unsigned int m, unsigned int k, unsigned int n)
{
    __local float buf_a[TILE_SIZE][TILE_SIZE];
    __local float buf_b[TILE_SIZE][TILE_SIZE];
    unsigned int gid_row = get_global_id(0);
    unsigned int gjd_col = get_global_id(1);
    unsigned int lid = get_local_id(0);
    unsigned int ljd = get_local_id(1);

    float ans = 0.0f;
    for (int i = 0; i < k; i += TILE_SIZE) {
        buf_a[ljd][lid] = a[gjd_col * k + i + lid];
        buf_b[ljd][lid] = b[(ljd + i) * n + gid_row];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < TILE_SIZE; j++) {
            ans += buf_a[ljd][j]*buf_b[j][lid];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[gjd_col * n + gid_row] = ans;
}
#endif

#if defined(TILE_SIZE) && defined(WORK_PER_THREAD)
__kernel void matrix_multiplication_local_wpt(__global const float *a, __global const float *b, __global float *c, unsigned int m, unsigned int k, unsigned int n)
{
    __local float buf_a[TILE_SIZE][TILE_SIZE];
    __local float buf_b[TILE_SIZE][TILE_SIZE];
    unsigned int lid = get_local_id(0);
    unsigned int ljd = get_local_id(1) * WORK_PER_THREAD;
    unsigned int grid_row = get_group_id(0) * TILE_SIZE + lid;
    unsigned int grjd_col = get_group_id(1) * TILE_SIZE + ljd;
    

    float sum[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; ++w) {
        sum[w] = 0.0f;
    }

    for (int i = 0; i < k; i += TILE_SIZE) {
        for (int w = 0; w < WORK_PER_THREAD; ++w) {
            buf_a[ljd + w][lid] = a[(grjd_col + w) * k + i + lid];
            buf_b[ljd + w][lid] = b[(w + i + ljd) * n + grid_row];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < TILE_SIZE; j++) {
            for (int w = 0; w < WORK_PER_THREAD; ++w) {
                sum[w] += buf_a[ljd + w][j]*buf_b[j][lid];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WORK_PER_THREAD; w++) {
        c[(grjd_col + w)*n + grid_row] = sum[w];
    }
}
#endif
