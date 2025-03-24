__kernel void bitonic(__global int* data, unsigned int k, unsigned int j)
{
    unsigned int gid = get_global_id(0);
    unsigned int l = gid ^ j;
    if (l > gid){
        unsigned int d = gid&k;
        bool swap = (!d && data[gid]>data[l]) || (d && data[gid]<data[l]);
        if (swap) {
            int temp = data[gid];
            data[gid] = data[l];
            data[l] = temp;
        }
    }
}
