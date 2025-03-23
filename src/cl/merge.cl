#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 5

int find_bound(__global const int* as, int size, int value, bool bound) { //true >=, false >
    int l = -1, r = size;
    while (r - l > 1) {
        int m = (l + r) / 2;
        if (as[m] > value || (as[m] == value && bound)) {
            r = m;
        } else {
            l = m;
        }
    }
    return r;
}

__kernel void merge_global(__global const int *as, __global int *bs, unsigned int block_size)
{
    unsigned int gid = get_global_id(0);
    int offset = gid % (2 * block_size);
    int num_block = gid / block_size;

    int cur_id;
    if (num_block % 2 == 0){
        cur_id = find_bound(as + (num_block + 1) * block_size, block_size, as[gid], 1);
    }else{
        cur_id = find_bound(as + (num_block - 1) * block_size, block_size, as[gid], 0);
    }

    bs[cur_id + (gid - offset) + (gid - num_block*block_size)] = as[gid];
}

__kernel void calculate_indices(__global const int *as, __global unsigned int *inds, unsigned int block_size)
{

}

__kernel void merge_local(__global const int *as, __global const unsigned int *inds, __global int *bs, unsigned int block_size)
{

}
