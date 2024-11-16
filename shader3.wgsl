// Kernel 2: Using Shared Memory
struct Matrix {
    size: vec2f,
    numbers: array<f32>,
}

@group(0) @binding(0) var<storage, read> A: Matrix;
@group(0) @binding(1) var<storage, read> B: Matrix;
@group(0) @binding(2) var<storage, read_write> C: Matrix;

const lds = 17;
var<workgroup> As: array<f32, 16*lds>; // pading to avoid bank conflict; just need 16*16
var<workgroup> Bs: array<f32, 16*lds>;
var<workgroup> Cs: array<f32, 16*lds>;

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let m = u32(A.size.x);
	let n = u32(B.size.y);
	let k = u32(A.size.y);
	let lda = k;
	let ldb = n;
	let ldc = n;

	C.size = vec2(A.size.x, B.size.y);

	// this workgropu is supposed to compute a block in C.
	let bi = 2*gid.x/16*16;
	let bj = 2*gid.y/16*16;
	let li = gid.x%8;
	let lj = gid.y%8;

	for (var ki=0u; ki<k; ki+=16) {
	    // load A and B into shared memory
        As[li*lds+lj] = A.numbers[(bi+li)*lda+ki+lj];
        Bs[li*lds+lj] = B.numbers[(ki+li)*ldb+bj+lj];
        As[li*lds+lj+8] = A.numbers[(bi+li)*lda+ki+lj+8];
        Bs[li*lds+lj+8] = B.numbers[(ki+li)*ldb+bj+lj+8];
        As[(li+8)*lds+lj] = A.numbers[(bi+li+8)*lda+ki+lj];
        Bs[(li+8)*lds+lj] = B.numbers[(ki+li+8)*ldb+bj+lj];
        As[(li+8)*lds+lj+8] = A.numbers[(bi+li+8)*lda+ki+lj+8];
        Bs[(li+8)*lds+lj+8] = B.numbers[(ki+li+8)*ldb+bj+lj+8];

        workgroupBarrier();
        // compute C
        for (var kk=0u; kk<16; kk+=2) {
            Cs[li*lds+lj] += As[li*lds+kk] * Bs[kk*lds+lj];
            Cs[li*lds+lj+8] += As[li*lds+kk] * Bs[kk*lds+lj+8];
            Cs[(li+8)*lds+lj] += As[(li+8)*lds+kk] * Bs[kk*lds+lj];
            Cs[(li+8)*lds+lj+8] += As[(li+8)*lds+kk] * Bs[kk*lds+lj+8];
            Cs[li*lds+lj] += As[li*lds+kk+1] * Bs[(kk+1)*lds+lj];
            Cs[li*lds+lj+8] += As[li*lds+kk+1] * Bs[(kk+1)*lds+lj+8];
            Cs[(li+8)*lds+lj] += As[(li+8)*lds+kk+1] * Bs[(kk+1)*lds+lj];
            Cs[(li+8)*lds+lj+8] += As[(li+8)*lds+kk+1] * Bs[(kk+1)*lds+lj+8];
        }
        workgroupBarrier();
	}
    C.numbers[(bi+li)*ldc+bj+lj] = Cs[li*lds+lj];
    C.numbers[(bi+li)*ldc+bj+lj+8] = Cs[li*lds+lj+8];
    C.numbers[(bi+li+8)*ldc+bj+lj] = Cs[(li+8)*lds+lj];
    C.numbers[(bi+li+8)*ldc+bj+lj+8] = Cs[(li+8)*lds+lj+8];
}
