#define VALUES_PER_WORKITEM 32
#define WORKGROUP_SIZE 128

__kernel void sum(__global const unsigned int* array, unsigned int n, __global unsigned int* result) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int grid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int sum = 0;


#if VERSION == 0
    if (gid >= n)
        return;
    atomic_add(result, array[gid]);

#elif VERSION == 1
    for (int i = gid * WORK_PER_ITEM; i < (gid + 1) * WORK_PER_ITEM; i ++) {
        if (i >= n)
            break;
        sum += array[i];
    }
    atomic_add(result, sum);
#elif VERSION == 2
    for (int i = grid * grs * WORK_PER_ITEM + lid; i < (grid + 1) * grs * WORK_PER_ITEM; i += grs) {
        if (i >= n)
            break;
        sum += array[i];
    }
    atomic_add(result, sum);
#elif VERSION == 3
    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < n ? array[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid != 0) {
        return;
    }
    unsigned int local_res = 0;
    for (int i = 0; i < grs; ++i) {
        local_res += buf[i];
    }
    atomic_add(result, local_res);

#elif VERSION == 4
    __local unsigned int buf[WORKGROUP_SIZE];
    
    buf[lid] = gid < n ? array[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = 1; stride < WORKGROUP_SIZE; stride *= 2) {
        int idx = 2 * lid * stride;
        if (idx < WORKGROUP_SIZE) {
            buf[idx] += buf[idx + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (lid == 0) {
        atomic_add(result, buf[0]);
    }
#endif
}