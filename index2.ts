async function loadShader(url: string) {
  const response = await fetch(url);
  const shaderCode = await response.text();
  return shaderCode;
}

(async () => {
  console.log("index2.js");
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    return;
  }
  if (!adapter.features.has("timestamp-query")) {
    console.log("no timestamp!");
    return;
  }

  const device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query"],
  });
  console.log(device.features);

  const M = 512;
  const N = 512;
  const K = 512;

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

  const resultMatrixBufferSize =
    Float32Array.BYTES_PER_ELEMENT * (2 + first[0] * second[1]);
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

  const shaderCode = await loadShader("./shader2.wgsl");
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

  const querySet = device.createQuerySet({
    type: "timestamp",
    count: 2,
  });
  const size = 2 * BigInt64Array.BYTES_PER_ELEMENT;
  console.log("BigINT SIZE", size);
  const queryBuffer = device.createBuffer({
    size: size,
    usage:
      GPUBufferUsage.QUERY_RESOLVE |
      // GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC,
  });
  const timestampWrites = {
    querySet,
    beginningOfPassWriteIndex: 0, // Write timestamp in index 0 when pass begins.
    endOfPassWriteIndex: 1, // Write timestamp in index 1 when pass ends.
  };

  const commandEncoder = device.createCommandEncoder();
  // commandEncoder.writeTimestamp(querySet, 0);
  const passEncoder = commandEncoder.beginComputePass({ timestampWrites });

  passEncoder.setPipeline(computePipeline);
  passEncoder.setBindGroup(0, bindGroup);
  const blockX = Math.ceil(first[0] / 8);
  const blockY = Math.ceil(second[1] / 8);
  // passEncoder.writeTimestamp(querySet, 0);

  passEncoder.dispatchWorkgroups(blockX, blockY);

  // passEncoder.writeTimestamp(querySet, 1);

  passEncoder.end();
  // commandEncoder.writeTimestamp(querySet, 1);
  commandEncoder.resolveQuerySet(querySet, 0, 2, queryBuffer, 0);

  const gpuReadBuffer = device.createBuffer({
    size: resultMatrixBufferSize,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  commandEncoder.copyBufferToBuffer(
    resultMatrixBuffer,
    0,
    gpuReadBuffer,
    0,
    resultMatrixBufferSize,
  );
  const queryReadBuffer = device.createBuffer({
    size: size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });
  commandEncoder.copyBufferToBuffer(queryBuffer, 0, queryReadBuffer, 0, size);

  const gpuCommand = commandEncoder.finish();
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
  const arrayBuffer = gpuReadBuffer.getMappedRange();
  console.log(new Float32Array(arrayBuffer));
})();
