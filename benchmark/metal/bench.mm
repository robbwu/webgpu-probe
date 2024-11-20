#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <iostream>
#include <chrono>

class MetalMatrixMultiply {
private:
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;

public:
    MetalMatrixMultiply() {
        NSError* error = nil;
        // Get default Metal device
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to create system default device, attempting to enumerate devices..." << std::endl;
            // Try to enumerate all devices
            NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
            if (!devices || [devices count] == 0) {
                throw std::runtime_error("No Metal devices found in the system after enumeration");
            }
            std::cout << "Found " << [devices count] << " Metal devices:" << std::endl;
            for (id<MTLDevice> dev in devices) {
                std::cout << "  - " << [[dev name] UTF8String] << std::endl;
            }

            // Take the first device
            device = [devices firstObject];
        }

        if (!device) {
            throw std::runtime_error("Failed to create Metal device");
        }

        // Create command queue
        commandQueue = [device newCommandQueue];
        if (!commandQueue) {
            throw std::runtime_error("Failed to create command queue");
        }
    }

    void multiplyMatrices(const float* matrixA, const float* matrixB, float* result,
                         int M, int N, int K) {
        // Create Metal buffers
        id<MTLBuffer> bufferA = [device newBufferWithBytes:matrixA
                                                   length:M * K * sizeof(float)
                                                  options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferB = [device newBufferWithBytes:matrixB
                                                   length:K * N * sizeof(float)
                                                  options:MTLResourceStorageModeShared];

        id<MTLBuffer> bufferC = [device newBufferWithBytes:result
                                                   length:M * N * sizeof(float)
                                                  options:MTLResourceStorageModeShared];

        // Create descriptor for matrix multiplication
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M
            columns:K
            rowBytes:K * sizeof(float)
            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K
            columns:N
            rowBytes:N * sizeof(float)
            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M
            columns:N
            rowBytes:N * sizeof(float)
            dataType:MPSDataTypeFloat32];

        // Create matrices
        MPSMatrix* matA = [[MPSMatrix alloc] initWithBuffer:bufferA
                                                descriptor:descA];
        MPSMatrix* matB = [[MPSMatrix alloc] initWithBuffer:bufferB
                                                descriptor:descB];
        MPSMatrix* matC = [[MPSMatrix alloc] initWithBuffer:bufferC
                                                descriptor:descC];

        // Create matrix multiplication kernel
        MPSMatrixMultiplication* matMul = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
            transposeLeft:false
            transposeRight:false
            resultRows:M
            resultColumns:N
            interiorColumns:K
            alpha:1.0
            beta:0.0];

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

        // Encode matrix multiplication
        [matMul encodeToCommandBuffer:commandBuffer
                        leftMatrix:matA
                        rightMatrix:matB
                        resultMatrix:matC];

        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back
        memcpy(result, [bufferC contents], M * N * sizeof(float));
    }
};

using namespace std;
using namespace chrono;
// Example usage
int main() {
    try {
        const int M = 2048;
        const int N = 2048;
        const int K = 2048;

        // Allocate and initialize matrices
        float* matrixA = new float[M * K];
        float* matrixB = new float[K * N];
        float* result = new float[M * N];

        // Initialize matrices with some values
        for (int i = 0; i < M * K; i++) matrixA[i] = 1.0f;
        for (int i = 0; i < K * N; i++) matrixB[i] = 2.0f;

        // this is for warm up
        MetalMatrixMultiply multiplier;

        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);


        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);
        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);
        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);
        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);
        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);
        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);
        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);
        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);
        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);
        multiplier.multiplyMatrices(matrixA, matrixB, result, M, N, K);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();

        // Use result...
        std::cout << "First element of result: " << result[0] << std::endl;
        std::cout << "Metal Shader Performance Library: " << std::endl;
        std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
        std::cout << "Average over 10 runs:" << std::endl;
        std::cout << "Time: " << duration_cast<microseconds>(t2 - t1).count()/10 << " us" << std::endl;
        std::cout << "GFLOPS: " << 10*2.0 * M * N * K / duration_cast<microseconds>(t2 - t1).count()/1e3 << std::endl;
        // Cleanup
        delete[] matrixA;
        delete[] matrixB;
        delete[] result;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
