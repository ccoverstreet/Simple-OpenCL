__kernel void copy_test(__global float *in, __global float *dst) {
	size_t gid = get_global_id(0);
	size_t lid = get_local_id(0);

	dst[gid] = in[gid];
}

__kernel void simple_reduce(__global float *in, __global uint *dst) {
	// Uses 3 decimal place precision
	size_t gid = get_global_id(0);		
	size_t lid = get_local_id(0);		

	uint res = (uint)(in[gid] * 1000) ;

	
	res = work_group_reduce_add(res);
	if (lid == 0) {
		atomic_add(dst, res);
	}
}
