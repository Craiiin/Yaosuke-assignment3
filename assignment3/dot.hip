#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

// 学生需要完成的核心函数：在 DCU 上计算两个向量的点积。
// TODO_STUDENT: 每个线程先累加自己负责的乘积，再让每个线程组输出一个部分和。
__global__ void dot_kernel(int n, const float* x, const float* y,
                           float* partial) {
  // n 是向量长度；x、y 位于 DCU 显存；partial 每个位置保存一个线程组的结果。
  // 实现思路：确定线程负责的元素，计算 x[i] * y[i]，再合并组内线程的结果。
  (void)n;
  (void)x;
  (void)y;
  (void)partial;
  return;
}

int main() {
  constexpr int n = 1 << 20;
  constexpr int threads = 256;
  constexpr int blocks = 256;
  std::vector<float> x(n, 1.0f), y(n, 2.0f), partial(blocks, 0.0f);
  float *dx = nullptr, *dy = nullptr, *dpartial = nullptr;

  // 为两个输入向量和每个线程组的部分结果申请 DCU 显存。
  hipMalloc(&dx, n * sizeof(float));
  hipMalloc(&dy, n * sizeof(float));
  hipMalloc(&dpartial, blocks * sizeof(float));
  // 将输入数据从 CPU 复制到 DCU，并把部分结果清零。
  hipMemcpy(dx, x.data(), n * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dy, y.data(), n * sizeof(float), hipMemcpyHostToDevice);
  hipMemset(dpartial, 0, blocks * sizeof(float));

  hipEvent_t start = nullptr, stop = nullptr;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  constexpr int repeats = 100;
  hipEventRecord(start);
  // 重复启动学生补全的 dot_kernel，统计单次平均计算时间。
  for (int repeat = 0; repeat < repeats; ++repeat) {
    hipLaunchKernelGGL(dot_kernel, dim3(blocks), dim3(threads), 0, 0,
                       n, dx, dy, dpartial);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  hipEventElapsedTime(&elapsed_ms, start, stop);
  elapsed_ms /= repeats;
  // 清空部分结果后再运行一次，确保检查的是一次完整点积。
  hipMemset(dpartial, 0, blocks * sizeof(float));
  hipLaunchKernelGGL(dot_kernel, dim3(blocks), dim3(threads), 0, 0,
                     n, dx, dy, dpartial);
  // 取回每个线程组的部分结果，供 CPU 合并成最终点积。
  hipMemcpy(partial.data(), dpartial, blocks * sizeof(float),
            hipMemcpyDeviceToHost);

  // 将 DCU 返回的各组结果相加，并与 CPU 直接计算的结果比较。
  const float gpu = std::accumulate(partial.begin(), partial.end(), 0.0f);
  double cpu = 0.0;
  for (int i = 0; i < n; ++i) cpu += static_cast<double>(x[i]) * y[i];
  const double error = std::abs(static_cast<double>(gpu) - cpu);
  std::cout << "Dot GPU=" << gpu << " CPU=" << cpu
            << " abs_error=" << error << " repeats=" << repeats
            << " elapsed_ms=" << elapsed_ms << '\n';

  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipFree(dx);
  hipFree(dy);
  hipFree(dpartial);
  if (error > 1e-2) {
    std::cerr << "FAIL: 请完成 dot_kernel 中的 TODO_STUDENT。\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
