#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <OpenCL/opencl.h>
#include "clblast_c.h"

// Utility function to check OpenCL errors
void checkError(cl_int error, const char* message) {
    if (error != CL_SUCCESS) {
        std::cerr << "Error: " << message << " (" << error << ")" << std::endl;
        exit(error);
    }
}

// Utility function to get device info string
std::string getDeviceInfo(cl_device_id device, cl_device_info param) {
    size_t size;
    cl_int err = clGetDeviceInfo(device, param, 0, nullptr, &size);
    checkError(err, "Getting device info size");

    std::vector<char> info(size);
    err = clGetDeviceInfo(device, param, size, info.data(), nullptr);
    checkError(err, "Getting device info");

    return std::string(info.data());
}
// Utility function to get device info value
template<typename T>
T getDeviceInfoValue(cl_device_id device, cl_device_info param) {
    T value;
    cl_int err = clGetDeviceInfo(device, param, sizeof(T), &value, nullptr);
    checkError(err, "Getting device info value");
    return value;
}

void printDeviceInfo(cl_device_id device) {
    std::cout << "\nDevice Information:" << std::endl;
    std::cout << "Name: " << getDeviceInfo(device, CL_DEVICE_NAME) << std::endl;
    std::cout << "Vendor: " << getDeviceInfo(device, CL_DEVICE_VENDOR) << std::endl;
    std::cout << "Version: " << getDeviceInfo(device, CL_DEVICE_VERSION) << std::endl;

    // Get compute units
    cl_uint compute_units = getDeviceInfoValue<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
    std::cout << "Compute Units: " << compute_units << std::endl;

    // Get max work group size
    size_t max_work_group_size = getDeviceInfoValue<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    std::cout << "Max Work Group Size: " << max_work_group_size << std::endl;

    // Get global memory size (in GB)
    cl_ulong global_mem_size = getDeviceInfoValue<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
    std::cout << "Global Memory Size: " << global_mem_size / (1024.0 * 1024.0 * 1024.0) << " GB" << std::endl;

    // Get max clock frequency
    cl_uint max_clock_freq = getDeviceInfoValue<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
    std::cout << "Max Clock Frequency: " << max_clock_freq << " MHz" << std::endl;

    std::cout << "\n";
}


int main() {
    // Initialize OpenCL
    cl_int err;
    cl_platform_id platform;
    cl_uint num_platforms;
    err = clGetPlatformIDs(1, &platform, &num_platforms);
    checkError(err, "Getting platform ID");

    cl_device_id device;
    cl_uint num_devices;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &num_devices);
    checkError(err, "Getting device ID");

    // Print device information
    printDeviceInfo(device);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "Creating context");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");

    // Matrix dimensions
    const size_t M = 2048; // Rows of A and C
    const size_t N = 2048; // Columns of B and C
    const size_t K = 2048; // Columns of A and rows of B

    // Initialize matrices with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N, 0.0f);

    for (size_t i = 0; i < M * K; ++i) A[i] = dis(gen);
    for (size_t i = 0; i < K * N; ++i) B[i] = dis(gen);

    // Create OpenCL buffers
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * M * K, A.data(), &err);
    checkError(err, "Creating buffer A");

    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * K * N, B.data(), &err);
    checkError(err, "Creating buffer B");

    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(float) * M * N, C.data(), &err);
    checkError(err, "Creating buffer C");

    // SGEMM parameters
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Warmup run
    cl_event event;
    err = CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
                       M, N, K, alpha, bufA, 0, K, bufB, 0, N, beta,
                       bufC, 0, N, &queue, &event);
    checkError(err, "Running SGEMM warmup");
    clWaitForEvents(1, &event);
    clReleaseEvent(event);

    // Benchmark runs
    const int num_runs = 10;
    std::vector<double> times;

    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        err = CLBlastSgemm(CLBlastLayoutRowMajor, CLBlastTransposeNo, CLBlastTransposeNo,
                          M, N, K, alpha, bufA, 0, K, bufB, 0, N, beta,
                          bufC, 0, N, &queue, &event);
        checkError(err, "Running SGEMM");

        clWaitForEvents(1, &event);
        clReleaseEvent(event);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        times.push_back(diff.count());
    }

    // Calculate statistics
    double total_time = 0.0;
    double min_time = times[0];
    for (double time : times) {
        total_time += time;
        min_time = std::min(min_time, time);
    }
    double avg_time = total_time / num_runs;

    // Calculate GFLOPS
    double gflops = (2.0 * M * N * K) / (min_time * 1e9); // Best performance
    double avg_gflops = (2.0 * M * N * K) / (avg_time * 1e9); // Average performance

    std::cout << "Matrix dimensions: " << M << "x" << K << " * " << K << "x" << N << std::endl;
    std::cout << "Average time: " << avg_time * 1000 << " ms" << std::endl;
    std::cout << "Best time: " << min_time * 1000 << " ms" << std::endl;
    std::cout << "Average Performance: " << avg_gflops << " GFLOPS" << std::endl;
    std::cout << "Peak Performance: " << gflops << " GFLOPS" << std::endl;

    // Cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
