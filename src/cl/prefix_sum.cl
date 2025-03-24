// TODO
__kernel void prefix_sum(__global unsigned int* a, __global unsigned int* b,  int s, int n, int step, int jump) {
    int cur_id = s + get_global_id(0) * step;
    if (cur_id < n && cur_id >= 0) {
        b[cur_id] = a[cur_id] + ((cur_id >= jump) ? a[cur_id - jump] : 0);
    }
}