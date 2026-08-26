// Fused PWL v2 forward for CPU (mode 0/1/2/3).
//
// Build via torch.utils.cpp_extension.load at runtime.

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <c10/util/irange.h>

namespace {

inline float clampf(float x, float lo, float hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

at::Tensor pwl_v2_forward_cpu(
    const at::Tensor& x,              // [N, D] float32 contiguous
    const at::Tensor& modes,          // [G] int64
    const at::Tensor& const_vals,     // [G] float32
    const at::Tensor& lin_m,          // [G] float32
    const at::Tensor& lin_b,          // [G] float32
    const at::Tensor& split_stacked,  // [G, max_k+1] float32
    const at::Tensor& slope_stacked,  // [G, max_k] float32
    const at::Tensor& intercept_stacked, // [G, max_k] float32
    const at::Tensor& n_intervals,    // [G] int64
    double x_min,
    double x_max) {

  TORCH_CHECK(x.device().is_cpu(), "x must be CPU");
  TORCH_CHECK(x.dtype() == at::kFloat, "x must be float32");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [N, D]");

  TORCH_CHECK(modes.device().is_cpu() && modes.dtype() == at::kLong && modes.is_contiguous(), "modes must be CPU int64 contiguous");
  TORCH_CHECK(const_vals.device().is_cpu() && const_vals.dtype() == at::kFloat && const_vals.is_contiguous(), "const_vals must be CPU float32 contiguous");
  TORCH_CHECK(lin_m.device().is_cpu() && lin_m.dtype() == at::kFloat && lin_m.is_contiguous(), "lin_m must be CPU float32 contiguous");
  TORCH_CHECK(lin_b.device().is_cpu() && lin_b.dtype() == at::kFloat && lin_b.is_contiguous(), "lin_b must be CPU float32 contiguous");
  TORCH_CHECK(split_stacked.device().is_cpu() && split_stacked.dtype() == at::kFloat && split_stacked.is_contiguous(), "split_stacked must be CPU float32 contiguous");
  TORCH_CHECK(slope_stacked.device().is_cpu() && slope_stacked.dtype() == at::kFloat && slope_stacked.is_contiguous(), "slope_stacked must be CPU float32 contiguous");
  TORCH_CHECK(intercept_stacked.device().is_cpu() && intercept_stacked.dtype() == at::kFloat && intercept_stacked.is_contiguous(), "intercept_stacked must be CPU float32 contiguous");
  TORCH_CHECK(n_intervals.device().is_cpu() && n_intervals.dtype() == at::kLong && n_intervals.is_contiguous(), "n_intervals must be CPU int64 contiguous");

  const auto N = x.size(0);
  const auto D = x.size(1);
  const auto G = modes.size(0);
  TORCH_CHECK(G > 0, "num_groups must be > 0");
  TORCH_CHECK(D % G == 0, "D must be divisible by G");
  const int64_t C = D / G;

  TORCH_CHECK(split_stacked.size(0) == G, "split_stacked shape mismatch");
  TORCH_CHECK(slope_stacked.size(0) == G, "slope_stacked shape mismatch");
  TORCH_CHECK(intercept_stacked.size(0) == G, "intercept_stacked shape mismatch");
  TORCH_CHECK(split_stacked.size(1) == slope_stacked.size(1) + 1, "split_stacked must be (G, max_k+1)");
  TORCH_CHECK(slope_stacked.sizes() == intercept_stacked.sizes(), "slope/intercept shape mismatch");

  const int64_t max_k = slope_stacked.size(1);
  const float xmin = static_cast<float>(x_min);
  const float xmax = static_cast<float>(x_max);

  auto y = at::empty_like(x);

  const float* x_ptr = x.data_ptr<float>();
  float* y_ptr = y.data_ptr<float>();

  const int64_t* modes_ptr = modes.data_ptr<int64_t>();
  const float* const_ptr = const_vals.data_ptr<float>();
  const float* m_ptr = lin_m.data_ptr<float>();
  const float* b_ptr = lin_b.data_ptr<float>();
  const float* split_ptr = split_stacked.data_ptr<float>();      // row-major
  const float* slope_ptr = slope_stacked.data_ptr<float>();
  const float* inter_ptr = intercept_stacked.data_ptr<float>();
  const int64_t* k_ptr = n_intervals.data_ptr<int64_t>();

  const int64_t total = N * D;

  at::parallel_for(0, total, 4096, [&](int64_t begin, int64_t end) {
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t d = idx % D;
      const int64_t g = d / C;
      const int64_t mode = modes_ptr[g];
      float xv = clampf(x_ptr[idx], xmin, xmax);
      float out = 0.0f;
      if (mode == 0) {
        out = 0.0f;
      } else if (mode == 1) {
        out = const_ptr[g];
      } else if (mode == 2) {
        out = m_ptr[g] * xv + b_ptr[g];
      } else {
        // pwl
        int64_t K = k_ptr[g];
        if (K <= 0) {
          out = 0.0f;
        } else {
          if (K > max_k) K = max_k;
          const float* split_g = split_ptr + g * (max_k + 1);
          int64_t count = 0;
          // count how many starts <= x (starts are split[0..K-1])
          for (int64_t i = 0; i < K; ++i) {
            count += (xv >= split_g[i]) ? 1 : 0;
          }
          int64_t seg = count - 1;
          if (seg < 0) seg = 0;
          if (seg >= K) seg = K - 1;
          const float* slope_g = slope_ptr + g * max_k;
          const float* inter_g = inter_ptr + g * max_k;
          out = slope_g[seg] * xv + inter_g[seg];
        }
      }
      y_ptr[idx] = out;
    }
  });

  return y;
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("pwl_v2_forward_cpu", &pwl_v2_forward_cpu, "PWL v2 forward (CPU, fused)");
}

