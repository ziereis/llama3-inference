#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include <numeric>
#include <iomanip>
#include "checkpoint_reader.h"

typedef double f64;
typedef float f32;
typedef int64_t i64;
typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;



struct q4_K_tensor {
  std::string name;
  std::vector<uint32_t> shape;
  std::vector<float> sb_scales;
  std::vector<float> sb_mins;
  std::vector<uint8_t> scales;
  std::vector<uint8_t> mins;
  std::vector<uint8_t> quants;

  explicit q4_K_tensor(std::vector<uint32_t> shape, std::string name) {
    uint64_t n = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
    this->shape = std::move(shape);
    this->name = std::move(name);
    quants.resize(n/2);
    scales.resize(n/32);
    mins.resize(n/32);
    sb_scales.resize((n/256) * 2);
    sb_mins.resize((n/256) * 2);
  }
};

struct Tensor {
    std::string name;
    std::vector<uint32_t> shape;
    std::vector<float> data;

    explicit Tensor(std::vector<uint32_t> shape, std::string name) {
      uint64_t n = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
      this->shape = std::move(shape);
      this->name = std::move(name);
      data.resize(n);
    }
};

void q_leastsquares(int nmax, int niter, float alpha, int n, float * x,
                       uint8_t* quants, float* qscale, float* qmin) {
  float min = x[0];
  float max = x[0];
  for (int i = 1; i < n; i++) {
    min = std::min(x[i], min);
    max = std::max(x[i], max);
  }
  if (min > 0)
    min = 0;
  float iscale = nmax / (max - min);
  float scale = 1 / iscale;

  for (int itry = 0; itry < niter; ++itry) {
    float sumqx = 0;
    float suml2 = 0;
    bool did_change = false;
    for (int i = 0; i < n; i++) {
      int q = std::round(iscale * (x[i] - min));
      q = std::clamp(q, 0, nmax);
      if (q != quants[i]) {
        quants[i] = q;
        did_change = true;
      }
      sumqx += (x[i] - min) * q;
      suml2 += q * q;
    }
    scale = sumqx / suml2;
    float diff = 0;
    for (int i = 0; i < n; i++) {
      diff += x[i] - scale * quants[i];
    }
    min = alpha * min + (1-alpha)*diff/n;
    if (min > 0)
      min = 0;
    iscale = 1/scale;
    if (!did_change) break;
  }
  *qscale = scale;
  *qmin = min;
}

void quantize_q4_k(const TensorInfo& x, q4_K_tensor& q4_k) {
  int n = x.size / sizeof(float);
  assert(n % 256 == 0);
  
  uint8_t quants[256];
  float scales[8];
  float mins[8];
  uint8_t qscales[8];
  uint8_t qmins[8];
  float sb_scales[2];
  float sb_mins[2];

  int n_blocks = n / 256;
  int s_blocks = 256/32;

  auto* data = x.data;

  for (int64_t b = 0; b < n_blocks; b++) {
    for (int64_t bb = 0; bb < 8; bb++) {
        float* data_begin = data + b * 256 + bb * 32;
        uint8_t* quants_begin = quants + bb * 32;
        q_leastsquares(15, 9, 0.5, 32, data_begin, quants_begin, scales + bb, mins + bb);
    }
    q_leastsquares(127, 9, 0.5, 8, scales, qscales, sb_scales, sb_mins);
    q_leastsquares(127, 9, 0.5, 8, mins, qmins, sb_scales +1, sb_mins +1);

    for (int64_t i = 0; i < 256/2; i++) {
      q4_k.quants[b*(256/2) + i] = ((quants[i * 2] << 4) & 0xF0) | (quants[i * 2 + 1]);
    }
    for (int64_t i = 0; i < 8; i++) {
      q4_k.scales[b * 8 + i] = qscales[i]; 
    }
    for (int64_t i = 0; i < 8; i++) {
      q4_k.mins[b * 8 + i] = qmins[i]; 
    }
    for (int64_t i = 0; i < 2; i++) {
      q4_k.sb_scales[b * 2 + i] = sb_scales[i]; 
    }
    for (int64_t i = 0; i < 2; i++) {
      q4_k.sb_mins[b * 2 + i] = sb_mins[i];
    }
  }
}



#define unpack_left(in) ((in >> 4) & 0x0f) - 8
#define unpack_right(in) (in & 0x0f) - 8
#define pack_q4(left, right) (((((u8)left) + 8) << 4) | (((u8)right) + 8))
void quantize_q4(i8  * out, f32  * scales, f32  * in,
                 u64 n, i32 group_size) {
  assert(n % 2 == 0);
  assert(group_size % 2 == 0);
  u64 n_groups = n / group_size;
  for (u64 i = 0; i < n_groups; i++) {
    u64 start_idx = i * group_size;
    u64 end_idx = (i + 1) * (group_size);

    f32 max = fabs(in[start_idx]);
    for (u64 j = start_idx; j < end_idx; j++) {
      max = fmax(max, fabs(in[j]));
    }
    f32 scale = max / 7.0f;

    for (u64 j = start_idx / 2; j < end_idx / 2; j++) {
      i8 left = (i8)round(in[j * 2] / scale);
      i8 right = (i8)round(in[(j * 2) + 1] / scale);
      out[j] = pack_q4(left, right);
    }
    scales[i] = scale;
  }
}

void dequantize_q4(f32  * out, i8  * in, f32  * scales,
                   u64 n, i32 group_size) {
  u64 n_groups = n / group_size;
  for (u64 i = 0; i < n_groups; i++) {
    u64 start_idx = i * group_size / 2;
    u64 end_idx = (i + 1) * (group_size / 2);
    for (u64 j = start_idx; j < end_idx; j++) {
      u8 p = (u8)in[j];
      i32 left = unpack_left(p);
      i32 right = unpack_right(p);
      out[j * 2] = ((f32)left) * scales[i];
      out[(j * 2) + 1] = ((f32)right) * scales[i];
    }
  }
}



void dequantize_q4_k(const q4_K_tensor& q4_k, Tensor& out) {
  int row_size = q4_k.shape[1];
  int n_blocks_per_row = row_size / 256;
  for (int row = 0; row < q4_k.shape[0]; row++) {
    for (int b = 0; b < n_blocks_per_row; b++) {
      float* out_b = out.data.data() + row_size * row + b * 256;
      const uint8_t* w_b = q4_k.quants.data() + row * (row_size/2) + b * 128;
      const uint8_t* s_b = q4_k.scales.data() + row * (row_size/32) + b * 8;
      const uint8_t* m_b = q4_k.mins.data() + row * (row_size/32) + b * 8;
      const float* sbs_b = q4_k.sb_scales.data() + row * (row_size/256 * 2) + b * 2;
      const float* sbm_b = q4_k.sb_mins.data() + row * (row_size/256 * 2) + b * 2;
      for (int i = 0; i < 8; i++) {
        float* out_sb = out_b + i * 32;
        const uint8_t* w_sb = w_b + i * 16;
        float scale = s_b[i] * sbs_b[0] + sbm_b[0];
        float min = m_b[i] * sbs_b[1] + sbm_b[1];
        for (int j = 0; j < 16; j++) {
          float left = (w_sb[j] & 0xF0) >> 4;
          float right = (w_sb[j] & 0x0F);
          out_sb[j*2] = left * scale + min;
          out_sb[j*2+1] =  right * scale + min;
        }
      }
    }
  }
}

// m x p  @ n x p = m x n
// W is stored transposed
void matmul(const float *x, const float *W, float *out, int64_t m, int64_t n, int64_t p) {
  for (int64_t row = 0; row < m; row++) {
    for (int64_t col = 0; col < n; col++) {
      float sum = 0;
      for (int64_t k = 0; k < p; k++) {
        sum += x[row * p + k] * W[col * p + k];
      }
      out[row * n + col] = sum;
    }
  }
}

void q4k_matmul(float* x, q4_K_tensor& W, float* out, int64_t m, int64_t n, int64_t p) {
  int nblocks = p / 256;
  for (int64_t row = 0; row < m; row++) {
    for (int64_t col = 0; col < n; col++) {
      float sum = 0;
      for (int64_t b = 0; b < nblocks; b++) {
        const uint8_t* w_b = W.quants.data() + col * (p/2) + b * 128;
        const uint8_t* s_b = W.scales.data() + col * (p/32) + b * 8;
        const uint8_t* m_b = W.mins.data() + col * (p/32) + b * 8;
        const float* sbs_b = W.sb_scales.data() + col * (p/256 * 2) + b * 2;
        const float* sbm_b = W.sb_mins.data() + col * (p/256 * 2) + b * 2;
        float* x_b = x + row * p + b * 256;
        for (int sb = 0 ; sb < 8; sb++) {
          float* x_sb = x_b + sb * 32;
          const uint8_t* w_sb = w_b + sb * 16;
          float scale = s_b[sb] * sbs_b[0] + sbm_b[0];
          float min = m_b[sb] * sbs_b[1] + sbm_b[1];
          for (int k = 0; k < 16; k++) {
            float left = (w_sb[k] & 0xF0) >> 4;
            float right = (w_sb[k] & 0x0F);
            sum += x_sb[k*2] * (left * scale + min);
            sum += x_sb[k*2 + 1] * (right * scale + min);
          }
        }
      }
      out[row * n + col] = sum;
    }
  }
}


int main(int argc, char* argv[]) {
  CheckpointReader reader("../output_checkpoint.bin");
  auto& tensors = reader.get_tensor_map();
  auto& embedding = tensors.at("tok_embeddings.weight");
  auto& wq = tensors.at("layers.0.attention.wq.weight");
  auto wq_q4 = q4_K_tensor(wq.shape, wq.name);
  quantize_q4_k(wq, wq_q4);

  std::vector<float> raw_matmul_result(20 * 4096);
  matmul(embedding.data, wq.data, raw_matmul_result.data(), 20, 4096, 4096);

  Tensor dequantized_wq(wq_q4.shape, wq_q4.name + "_dequantized");
  dequantize_q4_k(wq_q4, dequantized_wq);

  std::vector<float> dequantized_matmul_result(20 * 4096);
  matmul(embedding.data, dequantized_wq.data.data(), dequantized_matmul_result.data(), 20, 4096, 4096);

  std::vector<float> q4k_matmul_result(20 * 4096);
  q4k_matmul(embedding.data, wq_q4, q4k_matmul_result.data(), 20, 4096, 4096);

  float mse_dequantized = 0.0f, mse_q4k = 0.0f;
  float max_abs_error_dequantized = 0.0f, max_abs_error_q4k = 0.0f;
  for (int64_t i = 0; i < 20 * 4096; ++i) {
    float diff_dequantized = raw_matmul_result[i] - dequantized_matmul_result[i];
    float diff_q4k = raw_matmul_result[i] - q4k_matmul_result[i];
    mse_dequantized += diff_dequantized * diff_dequantized;
    mse_q4k += diff_q4k * diff_q4k;
    max_abs_error_dequantized = std::max(max_abs_error_dequantized, std::abs(diff_dequantized));
    max_abs_error_q4k = std::max(max_abs_error_q4k, std::abs(diff_q4k));
  }
  mse_dequantized /= (20 * 4096);
  mse_q4k /= (20 * 4096);

  float rmse_dequantized = std::sqrt(mse_dequantized);
  float rmse_q4k = std::sqrt(mse_q4k);

  std::cout << "Comparison between raw matmul and dequantized matmul:" << std::endl;
  std::cout << "Root Mean Square Error (RMSE): " << rmse_dequantized << std::endl;
  std::cout << "Maximum Absolute Error: " << max_abs_error_dequantized << std::endl;

  std::cout << "\nComparison between raw matmul and q4k_matmul:" << std::endl;
  std::cout << "Root Mean Square Error (RMSE): " << rmse_q4k << std::endl;
  std::cout << "Maximum Absolute Error: " << max_abs_error_q4k << std::endl;

  std::cout << "\nFirst 20 elements of raw matmul result:" << std::endl;
  for(int i = 0; i < 20; i++) {
    std::cout << raw_matmul_result[i] << ", ";
  }
  std::cout << std::endl;

  std::cout << "\nFirst 20 elements of dequantized matmul result:" << std::endl;
  for(int i = 0; i < 20; i++) {
    std::cout << dequantized_matmul_result[i] << ", ";
  }
  std::cout << std::endl;

  std::cout << "\nFirst 20 elements of q4k matmul result:" << std::endl;
  for(int i = 0; i < 20; i++) {
    std::cout << q4k_matmul_result[i] << ", ";
  }
  std::cout << std::endl;

  return 0;
}