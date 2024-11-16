// naive kernel for SGEMM row major, no shared memory
// global memory not coalesced

struct Matrix {
    size: vec2f,
    numbers: array<f32>,
}

@group(0) @binding(0) var<storage, read> A: Matrix;
@group(0) @binding(1) var<storage, read> B: Matrix;
@group(0) @binding(2) var<storage, read_write> C: Matrix;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
	let ti = gid.x;
	let tj = gid.y;
	let m = u32(A.size.x);
	let n = u32(B.size.y);
	let k = u32(A.size.y);

	if (gid.x >= m || gid.y >= n) {
		return;
	}

	C.size = vec2(A.size.x, B.size.y);

	let lda = k;
	let ldb = n;
	let ldc = n;

	var r = C.numbers[ti * ldc + tj];
	for (var i=0u; i<u32(A.size.y); i=i+1u) {
		r = r + A.numbers[ti * lda + i] * B.numbers[i * ldb + tj];
	}
	C.numbers[ti * ldc + tj] = r;
}
