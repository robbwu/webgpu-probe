
async function loadShader(url) {
    const response = await fetch(url);
    const shaderCode = await response.text();
    return shaderCode;
}

(async() => {

	const shaderCode = await loadShader('./shader.wgsl');
	// console.log(shaderCode);

	if (!navigator.gpu) throw Error("WebGPU not supported.");

	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) throw Error("Couldn’t request WebGPU adapter.");

	const required_features = [];
	if (adapter.features.has('shader-f16')) {
		required_features.push('shader-f16');
		console.log('shader-f16 supported');
	} else {
		alert('need a browser that supports shader-f16');
		return;
	}
	if (adapter.features.has('timestamp-query')) {
		required_features.push('timestamp-query');
		console.log('timestamp-query supported');
	} else {
		alert('need a browser that supports timestamp-query');
		return;
	}

	// console.log(adapter.info);
	// console.log("required_features", required_features);

	const device = await adapter.requestDevice({
		requiredFeatures: required_features,
	});
	if (!device) throw Error("Couldn’t request WebGPU logical device.");

	// console.log(device.features);


	const module = device.createShaderModule({
		code: shaderCode,
	});

	const bindGroupLayout =
		  device.createBindGroupLayout({
			  entries: [{
				  binding: 1,
				  visibility: GPUShaderStage.COMPUTE,
				  buffer: {
					  type: "storage",
				  },
			  }],
		  });
	
	const pipeline = device.createComputePipeline({
		layout: device.createPipelineLayout({
			bindGroupLayouts: [bindGroupLayout],
		}),
		compute: {
			module,
			entryPoint: "main",
		},
	});


	const BUFFER_SIZE = 1000000;

	const output = device.createBuffer({
		size: BUFFER_SIZE,
		usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
	});

	const stagingBuffer = device.createBuffer({
		size: BUFFER_SIZE,
		usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
	});

	const bindGroup = device.createBindGroup({
		layout: bindGroupLayout,
		entries: [{
			binding: 1,
			resource: {
				buffer: output,
			},
		}],
	});





	const commandEncoder = device.createCommandEncoder();

	const passEncoder = commandEncoder.beginComputePass();
	passEncoder.setPipeline(pipeline);
	passEncoder.setBindGroup(0, bindGroup);
	passEncoder.dispatchWorkgroups(1);
	passEncoder.dispatchWorkgroups(Math.ceil(BUFFER_SIZE / 64));
	passEncoder.end();

	commandEncoder.copyBufferToBuffer(
		output,
		0, // Source offset
		stagingBuffer,
		0, // Destination offset
		BUFFER_SIZE
	);
	const commands = commandEncoder.finish();
	device.queue.submit([commands]);

	await stagingBuffer.mapAsync(
		GPUMapMode.READ,
		0, // Offset
		BUFFER_SIZE // Length
	);
	const copyArrayBuffer =
		  stagingBuffer.getMappedRange(0, BUFFER_SIZE);
	const data = copyArrayBuffer.slice();
	stagingBuffer.unmap();
	// console.log(new Float16Array(data));

	const dataView = new DataView(data);
	const outputArray = new Array(BUFFER_SIZE / 4); // Since each f16 is 2 bytes
	for (let i = 0; i < BUFFER_SIZE / 4; i++) {
		outputArray[i] = dataView.getFloat32(i * 4, true);
	}
	console.log(outputArray);
	
})()
