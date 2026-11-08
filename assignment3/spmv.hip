#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// CSR 只保存非零元素：row_ptr 标出每行范围，col_idx 保存列号，values 保存数值。
struct CsrMatrix {
  uint32_t rows = 0, cols = 0, nnz = 0;
  std::vector<uint32_t> row_ptr, col_idx;
  std::vector<float> values;
};

// 从实验提供的二进制文件读取 CSR 矩阵，兼容单精度和双精度输入。
static bool load_csr(const std::string& path, CsrMatrix& a) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  in.read(reinterpret_cast<char*>(&a.rows), sizeof(a.rows));
  in.read(reinterpret_cast<char*>(&a.cols), sizeof(a.cols));
  in.read(reinterpret_cast<char*>(&a.nnz), sizeof(a.nnz));
  a.row_ptr.resize(static_cast<size_t>(a.rows) + 1);
  a.col_idx.resize(a.nnz);
  a.values.resize(a.nnz);
  in.read(reinterpret_cast<char*>(a.row_ptr.data()), a.row_ptr.size() * sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(a.col_idx.data()), a.col_idx.size() * sizeof(uint32_t));
  const std::streampos values_begin = in.tellg();
  in.seekg(0, std::ios::end);
  const std::streamoff value_bytes = in.tellg() - values_begin;
  in.seekg(values_begin);
  if (value_bytes == static_cast<std::streamoff>(a.nnz * sizeof(float))) {
    in.read(reinterpret_cast<char*>(a.values.data()), a.values.size() * sizeof(float));
  } else if (value_bytes == static_cast<std::streamoff>(a.nnz * sizeof(double))) {
    std::vector<double> values64(a.nnz);
    in.read(reinterpret_cast<char*>(values64.data()), values64.size() * sizeof(double));
    std::transform(values64.begin(), values64.end(), a.values.begin(),
                   [](double value) { return static_cast<float>(value); });
  } else {
    return false;
  }
  return static_cast<bool>(in);
}

// 学生需要完成的核心函数：每个 DCU 线程计算稀疏矩阵中的一行。
// TODO_STUDENT: 遍历当前行的非零元素，累加 values[k] * x[col_idx[k]]。
__global__ void spmv_kernel(int rows, const uint32_t* row_ptr,
                            const uint32_t* col_idx, const float* values,
                            const float* x, float* y) {
  // 当前行的非零元素范围是 [row_ptr[row], row_ptr[row + 1])。
  // col_idx[k] 才是输入向量 x 的下标，计算结果写入 y[row]。
  (void)rows;
  (void)row_ptr;
  (void)col_idx;
  (void)values;
  (void)x;
  (void)y;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "用法: spmv <matrix_file.bin>\n";
    return 2;
  }
  CsrMatrix a;
  if (!load_csr(argv[1], a)) {
    std::cerr << "无法读取矩阵: " << argv[1] << '\n';
    return 2;
  }

  // CPU 先计算一份正确结果，用于检查学生实现的 DCU 计算。
  std::vector<float> x(a.cols, 1.0f), gpu(a.rows, 0.0f), cpu(a.rows, 0.0f);
  for (uint32_t r = 0; r < a.rows; ++r) {
    for (uint32_t k = a.row_ptr[r]; k < a.row_ptr[r + 1]; ++k)
      cpu[r] += a.values[k] * x[a.col_idx[k]];
  }

  // 将 CSR 的三个数组以及输入、输出向量分别放入 DCU 显存。
  uint32_t *drow = nullptr, *dcol = nullptr;
  float *dval = nullptr, *dx = nullptr, *dy = nullptr;
  hipMalloc(&drow, a.row_ptr.size() * sizeof(uint32_t));
  hipMalloc(&dcol, a.col_idx.size() * sizeof(uint32_t));
  hipMalloc(&dval, a.values.size() * sizeof(float));
  hipMalloc(&dx, x.size() * sizeof(float));
  hipMalloc(&dy, gpu.size() * sizeof(float));
  hipMemcpy(drow, a.row_ptr.data(), a.row_ptr.size() * sizeof(uint32_t), hipMemcpyHostToDevice);
  hipMemcpy(dcol, a.col_idx.data(), a.col_idx.size() * sizeof(uint32_t), hipMemcpyHostToDevice);
  hipMemcpy(dval, a.values.data(), a.values.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(dx, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMemset(dy, 0, gpu.size() * sizeof(float));

  // 每个线程负责一行，因此线程总数必须覆盖矩阵的全部行。
  constexpr int threads = 256;
  const int blocks = (static_cast<int>(a.rows) + threads - 1) / threads;
  hipEvent_t start = nullptr, stop = nullptr;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  constexpr int repeats = 100;
  hipEventRecord(start);
  // 重复启动学生手写的 spmv_kernel，统计单次平均计算时间。
  for (int repeat = 0; repeat < repeats; ++repeat) {
    hipLaunchKernelGGL(spmv_kernel, dim3(blocks), dim3(threads), 0, 0,
                       static_cast<int>(a.rows), drow, dcol, dval, dx, dy);
  }
  hipEventRecord(stop);
  hipEventSynchronize(stop);
  float elapsed_ms = 0.0f;
  hipEventElapsedTime(&elapsed_ms, start, stop);
  elapsed_ms /= repeats;
  // 恢复输出向量后再运行一次，用于正确性检查。
  hipMemset(dy, 0, gpu.size() * sizeof(float));
  hipLaunchKernelGGL(spmv_kernel, dim3(blocks), dim3(threads), 0, 0,
                     static_cast<int>(a.rows), drow, dcol, dval, dx, dy);
  hipMemcpy(gpu.data(), dy, gpu.size() * sizeof(float), hipMemcpyDeviceToHost);

  // 逐行比较 CPU 与 DCU 结果，找出最大绝对误差。
  float max_error = 0.0f;
  for (uint32_t r = 0; r < a.rows; ++r)
    max_error = std::max(max_error, std::abs(gpu[r] - cpu[r]));
  std::cout << "rows=" << a.rows << " cols=" << a.cols << " nnz=" << a.nnz
            << " max_abs_error=" << max_error << " repeats=" << repeats
            << " elapsed_ms=" << elapsed_ms << '\n';

  hipEventDestroy(start);
  hipEventDestroy(stop);
  hipFree(drow); hipFree(dcol); hipFree(dval); hipFree(dx); hipFree(dy);
  if (max_error > 1e-4f) {
    std::cerr << "FAIL: 请完成 spmv_kernel 中的 TODO_STUDENT。\n";
    return 1;
  }
  std::cout << "PASS\n";
  return 0;
}
