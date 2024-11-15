@group(0) @binding(1)
var<storage, read_write> output : array<f32>;

var<workgroup> a : array<f32, 64>;

@compute @workgroup_size(64)
fn main(
    @builtin(global_invocation_id)
    gid : vec3<u32>,
    @builtin(local_invocation_id)
    lid : vec3<u32>,
) {
    let value = f32(gid.x) + f32(lid.x)/f32(100.0);
    a[lid.x ^ 1] = value;
    workgroupBarrier();
    if (gid.x < arrayLength(&output)) {
        output[gid.x] = a[lid.x];
    }
}
