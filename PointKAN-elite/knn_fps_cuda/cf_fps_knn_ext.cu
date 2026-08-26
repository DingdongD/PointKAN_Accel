#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>

namespace {

__device__ __forceinline__ float fmaxf2(float a, float b) { return a > b ? a : b; }

__global__ void init_min_dist_kernel(float* min_dist, int64_t B, int64_t N) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * N;
  if (idx < total) {
    min_dist[idx] = INFINITY;
  }
}

__global__ void seed_first_kernel(int64_t* farthest, int64_t B) {
  int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (b < B) farthest[b] = 0;
}

__global__ void seed_max_norm_kernel(const float* X, int64_t* farthest, int64_t B, int64_t N) {
  // one block per batch
  int b = blockIdx.x;
  if (b >= B) return;
  // simple reduction in global memory (not super fast, but ok for seed)
  float best = -1.0f;
  int64_t best_i = 0;
  for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
    const float x = X[(b * N + i) * 3 + 0];
    const float y = X[(b * N + i) * 3 + 1];
    const float z = X[(b * N + i) * 3 + 2];
    float n2 = x * x + y * y + z * z;
    if (n2 > best) {
      best = n2;
      best_i = i;
    }
  }
  // reduce within block
  __shared__ float s_best[256];
  __shared__ int64_t s_idx[256];
  int tid = threadIdx.x;
  if (tid < 256) {
    s_best[tid] = best;
    s_idx[tid] = best_i;
  }
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (s_best[tid + stride] > s_best[tid]) {
        s_best[tid] = s_best[tid + stride];
        s_idx[tid] = s_idx[tid + stride];
      }
    }
    __syncthreads();
  }
  if (tid == 0) farthest[b] = s_idx[0];
}

__global__ void compute_dist_kernel(
    const float* __restrict__ X,
    const int64_t* __restrict__ farthest,
    float* __restrict__ dist,
    int64_t B, int64_t N) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * N;
  if (idx >= total) return;
  int64_t b = idx / N;
  int64_t i = idx - b * N;
  int64_t c = farthest[b];
  const float cx = X[(b * N + c) * 3 + 0];
  const float cy = X[(b * N + c) * 3 + 1];
  const float cz = X[(b * N + c) * 3 + 2];
  const float x = X[(b * N + i) * 3 + 0];
  const float y = X[(b * N + i) * 3 + 1];
  const float z = X[(b * N + i) * 3 + 2];
  float dx = x - cx;
  float dy = y - cy;
  float dz = z - cz;
  dist[idx] = sqrtf(dx * dx + dy * dy + dz * dz);
}

__global__ void update_min_dist_kernel(
    const float* __restrict__ dist,
    const int64_t* __restrict__ farthest,
    float* __restrict__ min_dist,
    int64_t B, int64_t N) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * N;
  if (idx >= total) return;
  int64_t b = idx / N;
  int64_t i = idx - b * N;
  float d = dist[idx];
  float old = min_dist[idx];
  float nd = d < old ? d : old;
  min_dist[idx] = nd;
  // center distance set to 0
  if (i == farthest[b]) min_dist[idx] = 0.0f;
}

__global__ void argmax_min_dist_kernel(
    const float* __restrict__ min_dist,
    int64_t* __restrict__ farthest_out,
    int64_t B, int64_t N) {
  // one block per batch
  int b = blockIdx.x;
  if (b >= B) return;
  float best = -1.0f;
  int64_t best_i = 0;
  for (int64_t i = threadIdx.x; i < N; i += blockDim.x) {
    float v = min_dist[b * N + i];
    if (v > best) {
      best = v;
      best_i = i;
    }
  }
  __shared__ float s_best[256];
  __shared__ int64_t s_idx[256];
  int tid = threadIdx.x;
  s_best[tid] = best;
  s_idx[tid] = best_i;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if (tid < stride) {
      if (s_best[tid + stride] > s_best[tid]) {
        s_best[tid] = s_best[tid + stride];
        s_idx[tid] = s_idx[tid + stride];
      }
    }
    __syncthreads();
  }
  if (tid == 0) farthest_out[b] = s_idx[0];
}

__global__ void build_candidates_kernel(
    const float* __restrict__ dist,
    const int64_t* __restrict__ farthest,
    const float* __restrict__ thresholds, // [B]
    bool exclude_self,
    int64_t* __restrict__ cand_idx,     // [B, Mmax]
    int32_t* __restrict__ cand_fill,    // [B] filled <= Mmax
    int32_t* __restrict__ cand_count,   // [B] full count (may > Mmax)
    int64_t B, int64_t N, int32_t Mmax) {
  int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t total = B * N;
  if (idx >= total) return;
  int64_t b = idx / N;
  int64_t i = idx - b * N;
  float d = dist[idx];
  float threshold = thresholds[b];
  if (exclude_self && i == farthest[b]) return;
  if (d < threshold) {
    int32_t c = atomicAdd(&cand_count[b], 1);
    int32_t f = atomicAdd(&cand_fill[b], 1);
    if (f < Mmax) {
      cand_idx[b * (int64_t)Mmax + f] = i;
    }
    (void)c;
  }
}

__global__ void select_topk_from_candidates_kernel(
    const float* __restrict__ dist,
    const int64_t* __restrict__ cand_idx,
    const int32_t* __restrict__ cand_fill,
    int64_t* __restrict__ out_knn_base,  // [B, K, k] (contiguous)
    int64_t B, int64_t N, int64_t K, int64_t step_t, int32_t Mmax, int32_t k) {
  int b = blockIdx.x;
  if (b >= B) return;
  if (threadIdx.x != 0) return; // single-thread per batch (Mmax small)
  int32_t m = cand_fill[b];
  if (m > Mmax) m = Mmax;
  // simple insertion into top-k buffers
  // store best distances + indices
  float best_d[32];
  int64_t best_i[32];
  for (int j = 0; j < k; ++j) {
    best_d[j] = INFINITY;
    best_i[j] = 0;
  }
  if (m >= k) {
    for (int32_t ci = 0; ci < m; ++ci) {
      int64_t idx = cand_idx[b * (int64_t)Mmax + ci];
      float d = dist[b * N + idx];
      // insert if smaller than worst
      if (d >= best_d[k - 1]) continue;
      int pos = k - 1;
      while (pos > 0 && d < best_d[pos - 1]) {
        best_d[pos] = best_d[pos - 1];
        best_i[pos] = best_i[pos - 1];
        --pos;
      }
      best_d[pos] = d;
      best_i[pos] = idx;
    }
  } else {
    // Fallback path for insufficient candidates:
    // use full-set top-k so we don't emit default 0 indices.
    for (int64_t idx = 0; idx < N; ++idx) {
      float d = dist[b * N + idx];
      if (d >= best_d[k - 1]) continue;
      int pos = k - 1;
      while (pos > 0 && d < best_d[pos - 1]) {
        best_d[pos] = best_d[pos - 1];
        best_i[pos] = best_i[pos - 1];
        --pos;
      }
      best_d[pos] = d;
      best_i[pos] = idx;
    }
  }
  for (int j = 0; j < k; ++j) {
    out_knn_base[((int64_t)b * K + step_t) * k + j] = best_i[j];
  }
}

} // namespace

std::vector<at::Tensor> fps_knn_coarse_fine_forward_cuda(
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
  const c10::cuda::CUDAGuard device_guard(X.device());
  auto stream = at::cuda::getDefaultCUDAStream();

  const auto B = X.size(0);
  const auto N = X.size(1);
  K = std::min<int64_t>(K, N);
  int64_t k_actual = std::min<int64_t>(k_neighbors, N);
  int64_t min_candidates = std::max<int64_t>(k_actual + 1, (int64_t)(k_actual * min_candidates_ratio));
  int32_t Mmax = (int32_t)std::min<int64_t>(N, min_candidates * std::max<int64_t>(1, mmax_factor));

  auto idx = at::empty({B, K}, X.options().dtype(at::kLong));
  auto knn_idx = at::empty({B, K, k_actual}, X.options().dtype(at::kLong));

  auto dist = at::empty({B, N}, X.options().dtype(at::kFloat));
  auto min_dist = at::empty({B, N}, X.options().dtype(at::kFloat));
  auto farthest = at::empty({B}, X.options().dtype(at::kLong));

  const int threads = 256;
  {
    int64_t total = B * N;
    int blocks = (int)((total + threads - 1) / threads);
    init_min_dist_kernel<<<blocks, threads, 0, stream>>>(
        min_dist.data_ptr<float>(), B, N);
  }

  if (seed == "first") {
    int blocks = (int)((B + threads - 1) / threads);
    seed_first_kernel<<<blocks, threads, 0, stream>>>(farthest.data_ptr<int64_t>(), B);
  } else if (seed == "max_norm") {
    seed_max_norm_kernel<<<(int)B, threads, 0, stream>>>(X.data_ptr<float>(), farthest.data_ptr<int64_t>(), B, N);
  } else {
    // random seed: use torch randint on GPU for simplicity
    farthest.copy_(at::randint(0, N, {B}, X.options().dtype(at::kLong)));
  }

  auto cand_idx = at::empty({B, Mmax}, X.options().dtype(at::kLong));
  auto cand_fill = at::empty({B}, X.options().dtype(at::kInt));
  auto cand_count = at::empty({B}, X.options().dtype(at::kInt));
  auto thresholds = at::empty({B}, X.options().dtype(at::kFloat));

  for (int64_t t = 0; t < K; ++t) {
    // write idx[:, t] = farthest
    idx.select(1, t).copy_(farthest);

    // 1) distance to current center
    {
      int64_t total = B * N;
      int blocks = (int)((total + threads - 1) / threads);
      compute_dist_kernel<<<blocks, threads, 0, stream>>>(
          X.data_ptr<float>(), farthest.data_ptr<int64_t>(), dist.data_ptr<float>(), B, N);
    }

    // 2) coarse-fine threshold loop, done = count >= min_candidates
    int iters = (int)max_iter;
    // per-b thresholds to match python behavior
    thresholds.fill_((float)init_threshold);
    for (int i = 0; i < iters; ++i) {
      cand_fill.zero_();
      cand_count.zero_();
      int64_t total = B * N;
      int blocks = (int)((total + threads - 1) / threads);
      build_candidates_kernel<<<blocks, threads, 0, stream>>>(
          dist.data_ptr<float>(), farthest.data_ptr<int64_t>(),
          thresholds.data_ptr<float>(), exclude_self,
          cand_idx.data_ptr<int64_t>(),
          cand_fill.data_ptr<int32_t>(),
          cand_count.data_ptr<int32_t>(),
          B, N, Mmax);
      // check done on GPU by copying counts to CPU (cheap for small B)
      auto counts_cpu = cand_count.to(at::kCPU);
      auto counts_ptr = counts_cpu.data_ptr<int32_t>();
      bool done_all = true;
      for (int64_t b = 0; b < B; ++b) {
        if ((int64_t)counts_ptr[b] < min_candidates) {
          done_all = false;
          break;
        }
      }
      if (done_all) break;
      // update thresholds for unfinished batches (python: threshold *= 1.1 where not done)
      auto thr_cpu = thresholds.to(at::kCPU);
      auto thr_ptr = thr_cpu.data_ptr<float>();
      for (int64_t b = 0; b < B; ++b) {
        if ((int64_t)counts_ptr[b] < min_candidates) thr_ptr[b] *= 1.1f;
      }
      thresholds.copy_(thr_cpu);
    }

    // 3) select top-k from candidates (uses cand_fill limited to Mmax)
    select_topk_from_candidates_kernel<<<(int)B, 1, 0, stream>>>(
        dist.data_ptr<float>(),
        cand_idx.data_ptr<int64_t>(),
        cand_fill.data_ptr<int32_t>(),
        knn_idx.data_ptr<int64_t>(),
        B, N, K, t, Mmax, (int32_t)k_actual);

    // 4) FPS update min_dist and next farthest
    {
      int64_t total = B * N;
      int blocks = (int)((total + threads - 1) / threads);
      update_min_dist_kernel<<<blocks, threads, 0, stream>>>(
          dist.data_ptr<float>(),
          farthest.data_ptr<int64_t>(),
          min_dist.data_ptr<float>(),
          B, N);
    }
    if (t < K - 1) {
      argmax_min_dist_kernel<<<(int)B, threads, 0, stream>>>(
          min_dist.data_ptr<float>(), farthest.data_ptr<int64_t>(), B, N);
    }
  }

  return {idx, knn_idx};
}

