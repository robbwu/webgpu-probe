// https://github.com/zanussbaum/surfgrad/blob/main/src/shaders/matmul.ts
// adapted Nov 18, 2024

struct Matrix {
    size: vec2f,
    numbers: array<f32>,
}

@group(0) @binding(0) var<storage, read> A: Matrix;
@group(0) @binding(1) var<storage, read> B: Matrix;
@group(0) @binding(2) var<storage, read_write> C: Matrix;


const BLOCKSIZE: u32 = 16;
const TILE_M: u32 = 8;  // Tile size in M dimension
const TILE_N: u32 = 8;  // Tile size in N dimension

@compute @workgroup_size(BLOCKSIZE, BLOCKSIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let M = u32(A.size.x);
	let N = u32(B.size.y);
	let K = u32(A.size.y);
    let row = global_id.y * TILE_M;
    let col = global_id.x * TILE_N;

    var sums: array<array<f32, TILE_N>, TILE_M>;
    for (var i = 0u; i < TILE_M; i++) {
        for (var j = 0u; j < TILE_N; j++) {
            sums[i][j] = 0.0;
        }
    }

    // Compute the 2D tile
    for (var k = 0u; k < K; k++) {
      let a_00 = A.numbers[row * K + k];
      let a01 = A.numbers[(row + 1) * K + k];
      let a02 = A.numbers[(row + 2) * K + k];
      let a03 = A.numbers[(row + 3) * K + k];
      let a04 = A.numbers[(row + 4) * K + k];
      let a05 = A.numbers[(row + 5) * K + k];
      let a06 = A.numbers[(row + 6) * K + k];
      let a07 = A.numbers[(row + 7) * K + k];
      let b_00 = B.numbers[k * N + col];
      let b01 = B.numbers[k * N + (col + 1)];
      let b02 = B.numbers[k * N + (col + 2)];
      let b03 = B.numbers[k * N + (col + 3)];
      let b04 = B.numbers[k * N + (col + 4)];
      let b05 = B.numbers[k * N + (col + 5)];
      let b06 = B.numbers[k * N + (col + 6)];
      let b07 = B.numbers[k * N + (col + 7)];
      sums[0][0] += a_00 * b_00;
      sums[0][1] += a_00 * b01;
      sums[0][2] += a_00 * b02;
      sums[0][3] += a_00 * b03;
      sums[0][4] += a_00 * b04;
      sums[0][5] += a_00 * b05;
      sums[0][6] += a_00 * b06;
      sums[0][7] += a_00 * b07;
      sums[1][0] += a01 * b_00;
      sums[1][1] += a01 * b01;
      sums[1][2] += a01 * b02;
      sums[1][3] += a01 * b03;
      sums[1][4] += a01 * b04;
      sums[1][5] += a01 * b05;
      sums[1][6] += a01 * b06;
      sums[1][7] += a01 * b07;
      sums[2][0] += a02 * b_00;
      sums[2][1] += a02 * b01;
      sums[2][2] += a02 * b02;
      sums[2][3] += a02 * b03;
      sums[2][4] += a02 * b04;
      sums[2][5] += a02 * b05;
      sums[2][6] += a02 * b06;
      sums[2][7] += a02 * b07;
      sums[3][0] += a03 * b_00;
      sums[3][1] += a03 * b01;
      sums[3][2] += a03 * b02;
      sums[3][3] += a03 * b03;
      sums[3][4] += a03 * b04;
      sums[3][5] += a03 * b05;
      sums[3][6] += a03 * b06;
      sums[3][7] += a03 * b07;
      sums[4][0] += a04 * b_00;
      sums[4][1] += a04 * b01;
      sums[4][2] += a04 * b02;
      sums[4][3] += a04 * b03;
      sums[4][4] += a04 * b04;
      sums[4][5] += a04 * b05;
      sums[4][6] += a04 * b06;
      sums[4][7] += a04 * b07;
      sums[5][0] += a05 * b_00;
      sums[5][1] += a05 * b01;
      sums[5][2] += a05 * b02;
      sums[5][3] += a05 * b03;
      sums[5][4] += a05 * b04;
      sums[5][5] += a05 * b05;
      sums[5][6] += a05 * b06;
      sums[5][7] += a05 * b07;
      sums[6][0] += a06 * b_00;
      sums[6][1] += a06 * b01;
      sums[6][2] += a06 * b02;
      sums[6][3] += a06 * b03;
      sums[6][4] += a06 * b04;
      sums[6][5] += a06 * b05;
      sums[6][6] += a06 * b06;
      sums[6][7] += a06 * b07;
      sums[7][0] += a07 * b_00;
      sums[7][1] += a07 * b01;
      sums[7][2] += a07 * b02;
      sums[7][3] += a07 * b03;
      sums[7][4] += a07 * b04;
      sums[7][5] += a07 * b05;
      sums[7][6] += a07 * b06;
      sums[7][7] += a07 * b07;
    }
    C.numbers[row * N + col] = sums[0][0];
    C.numbers[row * N + (col + 1)] = sums[0][1];
    C.numbers[row * N + (col + 2)] = sums[0][2];
    C.numbers[row * N + (col + 3)] = sums[0][3];
    C.numbers[row * N + (col + 4)] = sums[0][4];
    C.numbers[row * N + (col + 5)] = sums[0][5];
    C.numbers[row * N + (col + 6)] = sums[0][6];
    C.numbers[row * N + (col + 7)] = sums[0][7];
    C.numbers[(row + 1) * N + col] = sums[1][0];
    C.numbers[(row + 1) * N + (col + 1)] = sums[1][1];
    C.numbers[(row + 1) * N + (col + 2)] = sums[1][2];
    C.numbers[(row + 1) * N + (col + 3)] = sums[1][3];
    C.numbers[(row + 1) * N + (col + 4)] = sums[1][4];
    C.numbers[(row + 1) * N + (col + 5)] = sums[1][5];
    C.numbers[(row + 1) * N + (col + 6)] = sums[1][6];
    C.numbers[(row + 1) * N + (col + 7)] = sums[1][7];
    C.numbers[(row + 2) * N + col] = sums[2][0];
    C.numbers[(row + 2) * N + (col + 1)] = sums[2][1];
    C.numbers[(row + 2) * N + (col + 2)] = sums[2][2];
    C.numbers[(row + 2) * N + (col + 3)] = sums[2][3];
    C.numbers[(row + 2) * N + (col + 4)] = sums[2][4];
    C.numbers[(row + 2) * N + (col + 5)] = sums[2][5];
    C.numbers[(row + 2) * N + (col + 6)] = sums[2][6];
    C.numbers[(row + 2) * N + (col + 7)] = sums[2][7];
    C.numbers[(row + 3) * N + col] = sums[3][0];
    C.numbers[(row + 3) * N + (col + 1)] = sums[3][1];
    C.numbers[(row + 3) * N + (col + 2)] = sums[3][2];
    C.numbers[(row + 3) * N + (col + 3)] = sums[3][3];
    C.numbers[(row + 3) * N + (col + 4)] = sums[3][4];
    C.numbers[(row + 3) * N + (col + 5)] = sums[3][5];
    C.numbers[(row + 3) * N + (col + 6)] = sums[3][6];
    C.numbers[(row + 3) * N + (col + 7)] = sums[3][7];
    C.numbers[(row + 4) * N + col] = sums[4][0];
    C.numbers[(row + 4) * N + (col + 1)] = sums[4][1];
    C.numbers[(row + 4) * N + (col + 2)] = sums[4][2];
    C.numbers[(row + 4) * N + (col + 3)] = sums[4][3];
    C.numbers[(row + 4) * N + (col + 4)] = sums[4][4];
    C.numbers[(row + 4) * N + (col + 5)] = sums[4][5];
    C.numbers[(row + 4) * N + (col + 6)] = sums[4][6];
    C.numbers[(row + 4) * N + (col + 7)] = sums[4][7];
    C.numbers[(row + 5) * N + col] = sums[5][0];
    C.numbers[(row + 5) * N + (col + 1)] = sums[5][1];
    C.numbers[(row + 5) * N + (col + 2)] = sums[5][2];
    C.numbers[(row + 5) * N + (col + 3)] = sums[5][3];
    C.numbers[(row + 5) * N + (col + 4)] = sums[5][4];
    C.numbers[(row + 5) * N + (col + 5)] = sums[5][5];
    C.numbers[(row + 5) * N + (col + 6)] = sums[5][6];
    C.numbers[(row + 5) * N + (col + 7)] = sums[5][7];
    C.numbers[(row + 6) * N + col] = sums[6][0];
    C.numbers[(row + 6) * N + (col + 1)] = sums[6][1];
    C.numbers[(row + 6) * N + (col + 2)] = sums[6][2];
    C.numbers[(row + 6) * N + (col + 3)] = sums[6][3];
    C.numbers[(row + 6) * N + (col + 4)] = sums[6][4];
    C.numbers[(row + 6) * N + (col + 5)] = sums[6][5];
    C.numbers[(row + 6) * N + (col + 6)] = sums[6][6];
    C.numbers[(row + 6) * N + (col + 7)] = sums[6][7];
    C.numbers[(row + 7) * N + col] = sums[7][0];
    C.numbers[(row + 7) * N + (col + 1)] = sums[7][1];
    C.numbers[(row + 7) * N + (col + 2)] = sums[7][2];
    C.numbers[(row + 7) * N + (col + 3)] = sums[7][3];
    C.numbers[(row + 7) * N + (col + 4)] = sums[7][4];
    C.numbers[(row + 7) * N + (col + 5)] = sums[7][5];
    C.numbers[(row + 7) * N + (col + 6)] = sums[7][6];
    C.numbers[(row + 7) * N + (col + 7)] = sums[7][7];

}
