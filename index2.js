"use strict";
async function loadShader(url) {
    const response = await fetch(url);
    const shaderCode = await response.text();
    return shaderCode;
}
(async () => {
    console.log("index2.js");
    let features = [];
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        return;
    }
    let hasTimestamp = false;
    if (adapter.features.has("timestamp-query")) {
        hasTimestamp = true;
        features.push("timestamp-query");
    }
    const device = await adapter.requestDevice({
        requiredFeatures: features,
    });
    console.log(device.features);
    const outputDiv = document.getElementById("output");
    // add user agent
    const sessionDiv = document.getElementById("session-info");
    const userAgent = document.createElement("div");
    userAgent.innerText = navigator.userAgent;
    sessionDiv.appendChild(userAgent);
    // add control radio button and run button, a matrix size text input
    const controlDiv = document.getElementById("control");
    const sizeInput = document.createElement("input");
    sizeInput.type = "text";
    sizeInput.value = "2048";
    controlDiv.appendChild(sizeInput);
    const runButton = document.createElement("button");
    runButton.innerText = "Run";
    controlDiv.appendChild(runButton);
    // checkbox for verifiying correctness
    const verifyCheckbox = document.createElement("input");
    verifyCheckbox.type = "checkbox";
    verifyCheckbox.checked = true;
    controlDiv.appendChild(verifyCheckbox);
    const cboxLabel = document.createElement("label");
    cboxLabel.innerText = "Verify (slow)";
    controlDiv.appendChild(cboxLabel);
    controlDiv.appendChild(document.createElement("br"));
    // add a dropdown for selecting the shader
    const shaderSelect = document.createElement("select");
    const shaderList = ["shader1.wgsl", "shader2.wgsl", "shader3.wgsl"];
    for (let i = 0; i < shaderList.length; i++) {
        const option = document.createElement("option");
        option.value = shaderList[i];
        option.text = shaderList[i];
        shaderSelect.appendChild(option);
    }
    controlDiv.appendChild(shaderSelect);
    runButton.onclick = async () => {
        const size = parseInt(sizeInput.value);
        console.log("Size", size);
        const [A, B, C] = await gpuCompute(size, shaderSelect.value);
        if (verifyCheckbox.checked) {
            cpuComputeAndCheck(A, B, C);
        }
    };
    const gpuCompute = async (n, shader) => {
        let BLOCK_SIZE = 16;
        if (shader == "shader2.wgsl" || shader == "shader1.wgsl") {
            BLOCK_SIZE = 8;
        }
        const M = n;
        const N = n;
        const K = n;
        const first = new Float32Array(2 + M * K);
        for (let i = 2; i < first.length; i++) {
            first[i] = Math.random();
        }
        first[0] = M;
        first[1] = K;
        const gpuBufferFirst = device.createBuffer({
            mappedAtCreation: true,
            size: first.byteLength,
            usage: GPUBufferUsage.STORAGE,
        });
        const arrayBufferFirst = gpuBufferFirst.getMappedRange();
        new Float32Array(arrayBufferFirst).set(first);
        gpuBufferFirst.unmap();
        const second = new Float32Array(2 + K * N);
        for (let i = 2; i < second.length; i++) {
            second[i] = Math.random();
        }
        second[0] = K;
        second[1] = N;
        const gpuBufferSecondMatrix = device.createBuffer({
            mappedAtCreation: true,
            size: second.byteLength,
            usage: GPUBufferUsage.STORAGE,
        });
        const arrayBufferSecondMatrix = gpuBufferSecondMatrix.getMappedRange();
        new Float32Array(arrayBufferSecondMatrix).set(second);
        gpuBufferSecondMatrix.unmap();
        const resultMatrixBufferSize = Float32Array.BYTES_PER_ELEMENT * (2 + first[0] * second[1]);
        const resultMatrixBuffer = device.createBuffer({
            size: resultMatrixBufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        const bindGroupLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "read-only-storage",
                    },
                },
                {
                    binding: 2,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    },
                },
            ],
        });
        const shaderCode = await loadShader(shader);
        console.log(shaderCode);
        const shaderModule = device.createShaderModule({
            code: shaderCode,
        });
        const computePipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [bindGroupLayout],
            }),
            compute: {
                module: shaderModule,
                entryPoint: "main",
            },
        });
        const bindGroup = device.createBindGroup({
            // layout: bindGroupLayout,
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: gpuBufferFirst,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: gpuBufferSecondMatrix,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: resultMatrixBuffer,
                    },
                },
            ],
        });
        const commandEncoder = device.createCommandEncoder();
        var passEncoder;
        var querySet = device.createQuerySet({
            type: "occlusion",
            count: 2,
        });
        const size = 2 * BigInt64Array.BYTES_PER_ELEMENT;
        console.log("BigINT SIZE", size);
        const queryBuffer = device.createBuffer({
            size: size,
            usage: GPUBufferUsage.QUERY_RESOLVE |
                // GPUBufferUsage.STORAGE |
                GPUBufferUsage.COPY_SRC,
        });
        if (hasTimestamp) {
            querySet = device.createQuerySet({
                type: "timestamp",
                count: 2,
            });
            const timestampWrites = {
                querySet,
                beginningOfPassWriteIndex: 0, // Write timestamp in index 0 when pass begins.
                endOfPassWriteIndex: 1, // Write timestamp in index 1 when pass ends.
            };
            passEncoder = commandEncoder.beginComputePass({ timestampWrites });
        }
        else {
            passEncoder = commandEncoder.beginComputePass();
        }
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        const blockX = Math.ceil(first[0] / BLOCK_SIZE);
        const blockY = Math.ceil(second[1] / BLOCK_SIZE);
        passEncoder.dispatchWorkgroups(blockX, blockY);
        // passEncoder.writeTimestamp(querySet, 1);
        passEncoder.end();
        // commandEncoder.writeTimestamp(querySet, 1);
        if (hasTimestamp)
            commandEncoder.resolveQuerySet(querySet, 0, 2, queryBuffer, 0);
        const gpuReadBuffer = device.createBuffer({
            size: resultMatrixBufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        commandEncoder.copyBufferToBuffer(resultMatrixBuffer, 0, gpuReadBuffer, 0, resultMatrixBufferSize);
        const queryReadBuffer = device.createBuffer({
            size: size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        commandEncoder.copyBufferToBuffer(queryBuffer, 0, queryReadBuffer, 0, size);
        const gpuCommand = commandEncoder.finish();
        let t0 = Date.now();
        device.queue.submit([gpuCommand]);
        await queryReadBuffer.mapAsync(GPUMapMode.READ);
        const queryArrayBuffer = queryReadBuffer.getMappedRange();
        const T = new BigUint64Array(queryArrayBuffer);
        console.log(Number(T[0]), Number(T[1]));
        const ns = T[1] - T[0];
        const GFLOPS = (2 * M * N * K) / Number(ns);
        console.log(`exec time ${Number(ns) / 1e6} ms`);
        console.log(`${GFLOPS} GFLOPSs`);
        queryReadBuffer.unmap();
        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        let t1 = Date.now();
        const arrayBuffer = gpuReadBuffer.getMappedRange();
        console.log(`Time taken ${t1 - t0} ms`);
        console.log(`GFLOPS js timer: ${(2 * M * N * K) / (t1 - t0) / 1e6}`);
        const result = new Float32Array(arrayBuffer);
        console.log(result);
        console.log(adapter.info);
        if (outputDiv) {
            let info = "unknown";
            if (adapter.info)
                info = `${adapter.info.vendor} ${adapter.info.architecture}`;
            const outputPre = document.createElement("pre");
            outputPre.textContent = `
    ${shader}: Matmul FP32 ${M}x${N}x${K} on ${info}
    exec time (js timer): ${Number(t1 - t0)} ms
    exec time (gpu/shader timer): ${Number(ns) / 1e6} ms
    GFLOPS (js timer): ${(2 * M * N * K) / (t1 - t0) / 1e6}
    `;
            outputDiv.prepend(outputPre);
        }
        return [first, second, result];
    };
    // check the accuracy
    const cpuComputeAndCheck = (A, B, C0) => {
        const M = A[0];
        const N = B[1];
        const K = A[1];
        const C = new Float32Array(M * N);
        const t0 = Date.now();
        sgemm("row-major", "no-transpose", "no-transpose", M, N, K, 1.0, A.slice(2), K, B.slice(2), N, 0.0, C, N);
        const t1 = Date.now();
        console.log(`IKJ CPU SGEMM: ${t1 - t0} ms`);
        console.log(`IKJ GFLOPS: ${(2 * M * N * K) / (t1 - t0) / 1e6}`);
        // console.log(C);
        let mismatch = false;
        for (var i = 0; i < M; i++) {
            for (var j = 0; j < N; j++) {
                if (Math.abs(C[i * N + j] - C0[2 + i * N + j]) > 1e-3) {
                    mismatch = true;
                    console.error(`mismatch at ${i} ${j} ${C[i * N + j]} ${C0[i * N + j]}`);
                    break;
                }
            }
            if (mismatch)
                break;
        }
        if (!mismatch) {
            console.log("All good");
        }
        else {
            console.log("Mismatch");
        }
    };
})();
// alpha can only be 1.0; beta can only be 0.0
function sgemm(major, transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc) {
    for (var i = 0; i < M; i++) {
        for (let k = 0; k < K; k++) {
            const aik = A[i * lda + k];
            for (var j = 0; j < N; j++) {
                let sum = beta * C[i * ldc + j];
                C[i * ldc + j] += aik * B[k * ldb + j];
            }
        }
    }
}
