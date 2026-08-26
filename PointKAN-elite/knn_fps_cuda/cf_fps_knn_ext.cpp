#include <torch/extension.h>

#include <vector>
#include <string>

std::vector<at::Tensor> fps_knn_coarse_fine_forward_cuda(
    const at::Tensor& X,            // [B, N, 3] float32 cuda
    int64_t K,
    int64_t k_neighbors,
    double init_threshold,
    int64_t max_iter,
    double min_candidates_ratio,
    const std::string& seed,
    bool exclude_self,
    int64_t block_size,
    int64_t mmax_factor);

std::vector<at::Tensor> fps_knn_coarse_fine_forward(
    const at::Tensor& X,
    int64_t K,
    int64_t k_neighbors,
    double init_threshold,
    int64_t max_iter,
    double min_candidates_ratio,
    const std::string& seed,
    bool exclude_self,
    int64_t block_size,
    int64_t mmax_factor) {
  TORCH_CHECK(X.is_cuda(), "X must be CUDA");
  TORCH_CHECK(X.dtype() == at::kFloat, "X must be float32");
  TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
  TORCH_CHECK(X.dim() == 3 && X.size(2) == 3, "X must be [B,N,3]");
  TORCH_CHECK(K > 0, "K must be > 0");
  TORCH_CHECK(k_neighbors > 0, "k_neighbors must be > 0");
  TORCH_CHECK(max_iter > 0, "max_iter must be > 0");
  TORCH_CHECK(min_candidates_ratio >= 1.0, "min_candidates_ratio must be >= 1.0");
  TORCH_CHECK(block_size > 0, "block_size must be > 0");
  TORCH_CHECK(mmax_factor >= 1, "mmax_factor must be >= 1");

  return fps_knn_coarse_fine_forward_cuda(
      X, K, k_neighbors, init_threshold, max_iter, min_candidates_ratio,
      seed, exclude_self, block_size, mmax_factor);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fps_knn_coarse_fine_forward_cuda", &fps_knn_coarse_fine_forward,
        "FPS + coarse-fine KNN forward (CUDA)");
}

