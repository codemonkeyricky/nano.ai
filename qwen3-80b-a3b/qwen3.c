#include <assert.h>
#include <fcntl.h>
#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

struct RotaryPosEmb {
    float *inv_freq;
};

struct Derived {
    struct RotaryPosEmb rope;
};

struct Config {
    int vocab_size;
    int hidden_size;
    int n_heads;
    int kv_heads;
    int n_layers;
    int max_position_embeddings;
    int intermediate_size;
    int head_dim;
    int num_experts;
    int num_experts_per_token;
    int moe_intermediate_size;

    struct Derived d;
};

struct Expert {
    const __bf16 *gate_proj;
    const __bf16 *up_proj;
    const __bf16 *down_proj;
};

struct SharedExpert {
    const __bf16 *gate;
    const __bf16 *gate_proj;
    const __bf16 *up_proj;
    const __bf16 *down_proj;
};

struct Layer {
    const __bf16 *gate;
    const __bf16 *input_layernorm;
    const __bf16 *post_attn_layernorm;
    /* self attention */
    const __bf16 *q_proj_w;
    const __bf16 *q_norm;
    const __bf16 *k_proj_w;
    const __bf16 *k_norm;
    const __bf16 *v_proj_w;
    const __bf16 *o_proj_w;
    /* linear attention */
    const __bf16 *linear_attn_in_proj_ba_w;
    const __bf16 *linear_attn_in_proj_qkvz_w;
    const __bf16 *linear_attn_out_proj_w;
    const __bf16 *linear_attn_conv1d_w;
    const __bf16 *linear_attn_dt_b;
    const __bf16 *linear_attn_a_log;
    const __bf16 *linear_attn_norm;
    const __bf16 *linear_attn_out_proj;
    struct Expert *experts;
    struct SharedExpert *shared_expert;
};

struct Mmapping {
    const __bf16 *embeddings;
    struct Layer *layers;
    const __bf16 *final_layernorm;
    const __bf16 *lm_head;
};

typedef struct {
    __bf16 *cache;
} Head;

struct Attention {
    float attn[32][64];
};

struct AttentionChunk {
    float attn[32][64][64];
};

struct Projection {
    float attn[32][128];
};

struct ProjectionChunk {
    float attn[32][64][128];
};

struct DecayChunk {
    float decay[32][64];
};

struct RecurrentState {
    float decay[32][128][128];
};

// struct Cache {
//     __bf16 cache[2][256];
// };

struct RLayer {
    /* self attention */
    Head *key_cache;
    Head *value_cache;

    /* linear attention */
    __bf16 mixed_qkv_raw[64][8192]; /* +4 for the convolution kernel size */
    __bf16 mixed_qkv[64][8192];
    struct Projection *query;
    struct AttentionChunk *attn_cache;
    struct ProjectionChunk *k_cache;
    struct ProjectionChunk *v_cache;
    struct ProjectionChunk *k_beta_cache;
    struct ProjectionChunk *k_beta_exp_cache;
    struct ProjectionChunk *v_beta_cache;
    struct ProjectionChunk *k_cumdecay;
    struct ProjectionChunk *value;
    struct DecayChunk *beta;
    struct DecayChunk *g;
    struct AttentionChunk *decay_mask;
    float v_prime[32][64][128];
    float v_new[32][64][128];
    float v_new_transposed[32][128][64];
    float last_recurrent_state[32][128][128];
    float last_recurrent_state_transposed[32][128][128];
    float attn_inter[32][64][128];
    float g_exp[32][64];
    float g_tmp[32][64];
    float core_attn_out[32][64][128];
    __bf16 core_attn_out_bf16[32][64][128];
    __bf16 z[64][32][128];
};

struct Runtime {
    __bf16 *qg;
    __bf16 *gate;
    __bf16 *q;
    __bf16 *k;
    __bf16 *v;
    __bf16 *h1;
    __bf16 *h2;
    __bf16 *h3;
    __bf16 *qkvz;
    __bf16 *ba;
    float (*g)[32]; // (*g)[64];

    struct RLayer *layers;
    char **lookup;
};

struct Transformer {
    struct Config config;
    struct Mmapping mmapping;
    struct Runtime runtime;
};

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void subtract_f32(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}

void mul_f32(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (float)a[i] * (float)b[i];
    }
}

void mul(__bf16 *out, const __bf16 *a, const __bf16 *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (__bf16)((float)a[i] * (float)b[i]);
    }
}

void mul_scalar(__bf16 *out, const __bf16 *a, float b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (__bf16)((float)a[i] * b);
    }
}

void silu_array(__bf16 *output, const __bf16 *input, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] / (1.0f + expf(-input[i]));
    }
}

void rope_init(struct Config *c) {

#if 0
    const float t = 5000000.0;
    const int d = c->head_dim; // c->hidden_size / c->n_heads;

    // Calculate inverse frequencies
    for (int i = 0; i < d / 2; i++) {
        float r = (float)(i * 2); // r = 0, 2, 4, ..., d-2
        float exponent = r / (float)d;
        c->d.rope.inv_freq[i] = 1.0f / powf(t, exponent);
    }
#endif
}

void rope_forward(struct Config *c, struct RotaryPosEmb *rope, int p, __bf16 *cos, __bf16 *sin) {
    const int d = c->head_dim;
    for (int f = 0; f < 32; f++) {
        float freq = (float)p * rope->inv_freq[f % 32];
        cos[f] = cos[32 + f] = cosf(freq);
        sin[f] = sin[32 + f] = sinf(freq);
    }
}

void mmap_layer_expert(struct Transformer *x, int layer, int expert) {

    struct Expert *ex = &x->mmapping.layers[layer].experts[expert];

    struct mmap_lookup {
        const char *path;
        const __bf16 **mmap;
    };

    struct mmap_lookup lookup[] = {
        {"weights/layer_%d_expert_%d_gate_proj.bin", &ex->gate_proj},
        {"weights/layer_%d_expert_%d_up_proj.bin", &ex->up_proj},
        {"weights/layer_%d_expert_%d_down_proj.bin", &ex->down_proj},
    };

    for (ssize_t i = 0; i < sizeof(lookup) / sizeof(lookup[0]); i++) {

        char path[FILENAME_MAX];
        snprintf(path, FILENAME_MAX, lookup[i].path, layer, expert);

        int fd = open(path, O_RDONLY);
        assert(fd > -1);
        int file_size = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);
        *lookup[i].mmap = (const __bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        assert(*lookup[i].mmap != MAP_FAILED);
        mlock(*lookup[i].mmap, file_size);
        close(fd);
    }
}

void mmap_layer_shared_expert(struct Transformer *x, int layer) {

    struct SharedExpert *ex = x->mmapping.layers[layer].shared_expert;

    struct mmap_lookup {
        const char *path;
        const __bf16 **mmap;
    };

    struct mmap_lookup lookup[] = {
        {"weights/layer_%d_shared_expert_gate.bin", &ex->gate},
        {"weights/layer_%d_shared_expert_gate_proj.bin", &ex->gate_proj},
        {"weights/layer_%d_shared_expert_up_proj.bin", &ex->up_proj},
        {"weights/layer_%d_shared_expert_down_proj.bin", &ex->down_proj},
    };

    for (ssize_t i = 0; i < sizeof(lookup) / sizeof(lookup[0]); i++) {

        char path[FILENAME_MAX];
        snprintf(path, FILENAME_MAX, lookup[i].path, layer);

        int fd = open(path, O_RDONLY);
        assert(fd > -1);
        int file_size = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);
        *lookup[i].mmap = (const __bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        assert(*lookup[i].mmap != MAP_FAILED);
        mlock(*lookup[i].mmap, file_size);
        close(fd);
    }
}

void mmap_layer(struct Transformer *x, int layer) {

    struct Layer *l = &x->mmapping.layers[layer];

    struct mmap_lookup {
        const char *path;
        const __bf16 **mmap;
    };

    struct mmap_lookup lookup[] = {
        {"weights/layer_%d_mlp_gate.bin", &l->gate},
        {"weights/layer_%d_input_layernorm.bin", &l->input_layernorm},
        {"weights/layer_%d_self_attn_q_proj_w.bin", &l->q_proj_w},
        {"weights/layer_%d_self_attn_q_norm.bin", &l->q_norm},
        {"weights/layer_%d_self_attn_k_proj_w.bin", &l->k_proj_w},
        {"weights/layer_%d_self_attn_k_norm.bin", &l->k_norm},
        {"weights/layer_%d_self_attn_v_proj_w.bin", &l->v_proj_w},
        {"weights/layer_%d_self_attn_o_proj_w.bin", &l->o_proj_w},
        {"weights/layer_%d_linear_attn_in_proj_qkvz_w.bin", &l->linear_attn_in_proj_qkvz_w},
        {"weights/layer_%d_linear_attn_in_proj_ba_w.bin", &l->linear_attn_in_proj_ba_w},
        {"weights/layer_%d_linear_attn_out_proj_w.bin", &l->linear_attn_out_proj_w},
        {"weights/layer_%d_linear_attn_conv1d_w.bin", &l->linear_attn_conv1d_w},
        {"weights/layer_%d_linear_attn_dt_b.bin", &l->linear_attn_dt_b},
        {"weights/layer_%d_linear_attn_a_log.bin", &l->linear_attn_a_log},
        {"weights/layer_%d_linear_attn_norm.bin", &l->linear_attn_norm},
        {"weights/layer_%d_linear_attn_out_proj.bin", &l->linear_attn_out_proj},
        {"weights/layer_%d_post_attention_layernorm.bin", &l->post_attn_layernorm},
    };

    for (ssize_t i = 0; i < sizeof(lookup) / sizeof(lookup[0]); i++) {

        char path[FILENAME_MAX];
        snprintf(path, FILENAME_MAX, lookup[i].path, layer);

        int fd = open(path, O_RDONLY);
        if (fd > -1) {
            int file_size = lseek(fd, 0, SEEK_END);
            lseek(fd, 0, SEEK_SET);
            *lookup[i].mmap = (const __bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
            assert(*lookup[i].mmap != MAP_FAILED);
            mlock(*lookup[i].mmap, file_size);
            close(fd);
        }
    }
}

void mmap_init(struct Config *config, struct Mmapping *mmapping) {
    int fd = open("weights/embeddings.bin", O_RDONLY);
    assert(fd > -1);
    int file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    mmapping->embeddings = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(mmapping->embeddings != MAP_FAILED);
    mlock(mmapping->embeddings, file_size);
    close(fd);

    mmapping->layers = (struct Layer *)calloc(1, sizeof(struct Layer) * config->n_layers);

    for (int i = 0; i < config->n_layers; i++) {
        mmap_layer((struct Transformer *)config, i);

        mmapping->layers[i].experts = (struct Expert *)malloc(sizeof(struct Expert) * 512);
        for (int ex = 0; ex < 512; ++ex) {
            mmap_layer_expert((struct Transformer *)config, i, ex);
        }

        mmapping->layers[i].shared_expert = (struct SharedExpert *)calloc(1, sizeof(struct SharedExpert));
        mmap_layer_shared_expert((struct Transformer *)config, i);
    }

    fd = open("weights/norm.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    mmapping->final_layernorm = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(mmapping->final_layernorm != MAP_FAILED);
    mlock(mmapping->final_layernorm, file_size);
    close(fd);

    fd = open("weights/lm_head.bin", O_RDONLY);
    if (fd != -1) {
        file_size = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);
        mmapping->lm_head = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        assert(mmapping->lm_head != MAP_FAILED);
        mlock(mmapping->lm_head, file_size);
        close(fd);
    }
}

static float tensor_array[] = {
    1.0000e+00f, 6.0430e-01f, 3.6517e-01f, 2.2067e-01f, 1.3335e-01f, 8.0584e-02f, 4.8697e-02f, 2.9427e-02f,
    1.7783e-02f, 1.0746e-02f, 6.4938e-03f, 3.9242e-03f, 2.3714e-03f, 1.4330e-03f, 8.6596e-04f, 5.2330e-04f,
    3.1623e-04f, 1.9110e-04f, 1.1548e-04f, 6.9783e-05f, 4.2170e-05f, 2.5483e-05f, 1.5399e-05f, 9.3057e-06f,
    5.6234e-06f, 3.3982e-06f, 2.0535e-06f, 1.2409e-06f, 7.4989e-07f, 4.5316e-07f, 2.7384e-07f, 1.6548e-07f};

void config_init(struct Config *config) {

    char path[FILENAME_MAX];
    snprintf(path, FILENAME_MAX, "weights/config.bin");

    FILE *f = fopen(path, "rb");
    assert(f != NULL);

    int vals[11];
    size_t n = fread(vals, sizeof(int), 11, f);
    assert(n == 11);
    fclose(f);

    config->vocab_size = vals[0];
    config->hidden_size = vals[1];
    config->n_heads = vals[2];
    config->kv_heads = vals[5];
    config->n_layers = vals[3];
    config->max_position_embeddings = vals[4];
    config->intermediate_size = vals[6];
    config->head_dim = vals[7];
    config->num_experts = vals[8];
    config->num_experts_per_token = vals[9];
    config->moe_intermediate_size = vals[10];

    // config->d.rope.inv_freq = (float *)malloc(sizeof(float) * (config->head_dim) / 2);
    config->d.rope.inv_freq = tensor_array;

    rope_init(config);
}

void norm(__bf16 *out, const __bf16 *in, const __bf16 *weight, const int len, struct Transformer *x) {

    struct Config *c = &x->config;
    struct Mmapping *m = &x->mmapping;

    float mean = 0.0f;
    for (int i = 0; i < len; i++) {
        mean += (float)in[i];
    }
    mean /= (float)len;

    float variance = 0.0f;
    for (int i = 0; i < len; i++) {
        float diff = (float)in[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)len;

    float denom = 1.0f / sqrtf(variance + 1e-6f);

    for (int i = 0; i < len; i++) {
        out[i] = ((__bf16)((float)in[i] * denom)) * (float)weight[i];
    }
}

void l2norm_forward(__bf16 *out, __bf16 *in, int dim) {
    const float eps = 1e-6f; // Small epsilon value to avoid division by zero

    // Calculate sum of squares (convert to float for precision)
    __bf16 sum_squares = 0.0f;
    for (int i = 0; i < dim; i++) {
        // float val = (float)in[i]; // Convert bf16 to float
        sum_squares += in[i] * in[i];
    }

    // Calculate reciprocal square root of (sum_squares + eps)
    __bf16 rsqrt_val = 1.0f / sqrtf(sum_squares + (__bf16)eps);
    for (int i = 0; i < dim; i++) {
        out[i] = in[i] * rsqrt_val;
    }
}

void rmsnorm_forward(__bf16 *out, __bf16 *in, int dim) {
    const float eps = 1e-6f; // Small epsilon value to avoid division by zero

    // Calculate sum of squares (convert to float for precision)
    float sum_squares = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = (float)in[i]; // Convert bf16 to float
        sum_squares += val * val;
    }

    // Calculate mean of squares
    float mean_squares = sum_squares / dim;

    // Calculate reciprocal square root of (mean_squares + eps)
    float rsqrt_val = 1.0f / sqrtf(mean_squares + eps);

    // Apply normalization: out = in * rsqrt(mean(in^2) + eps)
    for (int i = 0; i < dim; i++) {
        float val = (float)in[i];           // Convert bf16 to float
        out[i] = (__bf16)(val * rsqrt_val); // Convert back to bf16
    }
}

void rmsnorm_f32(float *out, float *in, int dim) {
    const float eps = 1e-6f; // Small epsilon value to avoid division by zero

    // Calculate sum of squares (convert to float for precision)
    float sum_squares = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = (float)in[i]; // Convert bf16 to float
        sum_squares += val * val;
    }

    // Calculate mean of squares
    float mean_squares = sum_squares / dim;

    // Calculate reciprocal square root of (mean_squares + eps)
    float rsqrt_val = 1.0f / sqrtf(mean_squares + eps);

    // Apply normalization: out = in * rsqrt(mean(in^2) + eps)
    for (int i = 0; i < dim; i++) {
        float val = (float)in[i]; // Convert bf16 to float
        out[i] = val * rsqrt_val; // Convert back to bf16
    }
}

void conv1d_simple(__bf16 *output, const __bf16 *input, const __bf16 *weight, int length, int stride, int padding) {

    int kernel_size = 4;
    int output_length = (length + 2 * padding - (kernel_size - 1) - 1) / stride + 1;

    // Perform convolution - each filter processes its corresponding input channel
    for (int ol = 0; ol < output_length; ol++) {
        int input_start = ol * stride - padding;

        for (int k = 0; k < kernel_size; k++) {
            int input_pos = input_start + k;

            if (input_pos >= 0 && input_pos < length) {
                int input_idx = input_pos;
                int weight_idx = k;
                int output_idx = ol;

                output[output_idx] += input[input_idx] * weight[weight_idx];
            }
        }
    }
}

void layernorm_n(__bf16 *out, const __bf16 *in, const __bf16 *weight, const int n, struct Transformer *x) {

    struct Config *c = &x->config;

    float tmp[n], tmp2[n];
    for (int i = 0; i < n; i++) {
        tmp[i] = (float)in[i];
    }

    rmsnorm_f32(tmp, tmp, n);
    for (size_t i = 0; i < n; i++) {
        tmp2[i] = (float)weight[i] + 1.0f;
    }

    for (size_t i = 0; i < n; i++) {
        out[i] = (__bf16)(tmp2[i] * tmp[i]);
    }
}

void layernorm(__bf16 *out, const __bf16 *in, const __bf16 *weight, struct Transformer *x) {

    struct Config *c = &x->config;
    int n = c->hidden_size;

    float tmp[n], tmp2[n];
    for (int i = 0; i < n; i++) {
        tmp[i] = (float)in[i];
    }

    rmsnorm_f32(tmp, tmp, n);
    for (size_t i = 0; i < n; i++) {
        tmp2[i] = (float)weight[i] + 1.0f;
    }

    for (size_t i = 0; i < n; i++) {
        out[i] = (__bf16)(tmp2[i] * tmp[i]);
    }
}

float dot_f32(float *__restrict x, float *__restrict w, int n) {

    assert(n % 32 == 0);

    // x = __builtin_assume_aligned(x, 64);
    // w = __builtin_assume_aligned(w, 64);

    float out = 0;
    for (int j = 0; j < n; ++j) {
        out += x[j] * w[j];
    }
    return out;
}

__bf16 dot(__bf16 *__restrict x, __bf16 *__restrict w, int n) {

    assert(n % 32 == 0);

    x = __builtin_assume_aligned(x, 64);
    w = __builtin_assume_aligned(w, 64);

    __m512 acc = _mm512_setzero_ps();
    for (int j = 0; j < n; j += 32) {
        __m512bh a = (__m512bh)_mm512_loadu_si512((const __m512i *)&x[j]);
        __m512bh b = (__m512bh)_mm512_loadu_si512((const __m512i *)&w[j]);
        acc = _mm512_dpbf16_ps(acc, a, b);
    }
    return _mm512_reduce_add_ps(acc);
}

void transpose(float *out, const float *in, int n, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            out[j * n + i] = in[i * m + j];
        }
    }
}

void matmul_bias(__bf16 *__restrict out, const __bf16 *__restrict x, const __bf16 *__restrict w,
                 const __bf16 *__restrict b, int n, int d) {
    out = __builtin_assume_aligned(out, 64);
    x = __builtin_assume_aligned(x, 64);
    w = __builtin_assume_aligned(w, 64);
    assert(n % 32 == 0);
    // assert(d % 32 == 0);
    int i;
    // #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        __bf16 tmp = dot(x, &w[i * n], n) + (b ? (float)b[i] : 0);
        out[i] = tmp;
    }
}

void matmul(__bf16 *__restrict out, const __bf16 *__restrict x, const __bf16 *__restrict w, int n, int d) {
    matmul_bias(out, x, w, 0, n, d);
}

void matmul_bias_f32(float *__restrict out, const float *__restrict x, const float *__restrict w,
                     const float *__restrict b, int n, int d) {
    // out = __builtin_assume_aligned(out, 64);
    // x = __builtin_assume_aligned(x, 64);
    // w = __builtin_assume_aligned(w, 64);
    assert(n % 32 == 0);
    // assert(d % 32 == 0);
    int i;
    // #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float tmp = dot_f32(x, &w[i * n], n) + (b ? (float)b[i] : 0);
        out[i] = tmp;
    }
}

void matmul_f32(float *__restrict out, const float *__restrict x, const float *__restrict w, int n, int d) {
    matmul_bias_f32(out, x, w, 0, n, d);
}

void add(__bf16 *out, const __bf16 *a, const __bf16 *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (__bf16)((float)a[i] + (float)b[i]);
    }
}

void rotate_half(__bf16 *out, const __bf16 *x, int D) {
    int half = D / 2;
    for (int i = 0; i < half; i++) {
        out[i] = -x[i + half]; // Negate the second half and put it in the first half
        out[i + half] = x[i];  // Copy the first half to the second half
    }
}

void rotary_positional_embedding(__bf16 *emb, __bf16 *cos, __bf16 *sin) {
    int n = 64;

    __bf16 first[64], second[64];
    memcpy(first, emb, 64 * sizeof(__bf16));
    memcpy(second, emb, 64 * sizeof(__bf16));

    for (int i = 0; i < 64; ++i) {
        first[i] *= cos[i];
    }

    rotate_half(second, second, 64);
    for (int i = 0; i < 64; ++i) {
        second[i] *= sin[i];
    }

    for (int i = 0; i < 64; ++i) {
        emb[i] = first[i] + second[i];
    }
}

__bf16 qg_proj[64][8192];
__bf16 gate[64][4096];
__bf16 query_state[16][64][256];
__bf16 key_state[2][64][256];
__bf16 value_state[2][64][256];

void self_attention(__bf16 xout[64][2048], __bf16 x[64][2048], const struct Transformer *xfmr, const int layer,
                    const int n, __bf16 sin[64][64], __bf16 cos[64][64]) {
    struct query_gate {
        __bf16 q[256];
        __bf16 gate[256];
    };

    struct projection {
        __bf16 block[256];
    };

    struct Config *p = &xfmr->config;
    const struct Runtime *r = &xfmr->runtime;
    const struct Mmapping *m = &xfmr->mmapping;

    /* q/k/v weight and bias */
    const __bf16 *qw = m->layers[layer].q_proj_w;
    const __bf16 *kw = m->layers[layer].k_proj_w;
    const __bf16 *vw = m->layers[layer].v_proj_w;

    /* output projection */
    const __bf16 *ow = m->layers[layer].o_proj_w;

    // int attn_sz = p->n_heads * p->head_dim;
    // int sz = p->kv_heads * p->head_dim;

    /* query gate projection */
    for (int i = 0; i < n; ++i) {
        matmul(qg_proj[i], x[i], qw, 2048, 8192);
    }

    for (int i = 0; i < n; ++i) {
        for (int h = 0; h < 16; ++h) {
            memcpy(query_state[h][i], &qg_proj[i][512 * h], 256 * sizeof(__bf16));
            memcpy(&gate[i][h * 256], &qg_proj[i][512 * h + 256], 256 * sizeof(__bf16));
        }
    }

    /* query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2) */
    for (int h = 0; h < 16; ++h) {
        for (int i = 0; i < n; ++i) {
            layernorm_n(query_state[h][i], query_state[h][i], m->layers[layer].q_norm, 256, xfmr);
        }
    }

    /* key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2) */
    {
        /* projection */
        for (int i = 0; i < n; ++i) {
            __bf16 tmp[2][256] = {};
            matmul((__bf16 *)tmp, x[i], kw, 2048, 512);
            for (int h = 0; h < 2; ++h) {
                memcpy(key_state[h][i], tmp[h], 256 * sizeof(__bf16));
            }
        }

        /* normalization */
        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < n; ++i) {
                layernorm_n(key_state[h][i], key_state[h][i], m->layers[layer].k_norm, 256, xfmr);
            }
        }
    }

    /* value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) */
    {
        /* projection */
        for (int i = 0; i < n; ++i) {
            __bf16 tmp[2][256] = {};
            matmul((__bf16 *)tmp, x[i], vw, 2048, 512);
            for (int h = 0; h < 2; ++h) {
                memcpy(value_state[h][i], tmp[h], 256 * sizeof(__bf16));
            }
        }
    }

    for (int h = 0; h < 16; ++h) {
        for (int i = 0; i < n; ++i) {
            rotary_positional_embedding(query_state[h][i], cos[i], sin[i]);
        }
    }

    for (int h = 0; h < 2; ++h) {
        for (int i = 0; i < n; ++i) {
            rotary_positional_embedding(key_state[h][i], cos[i], sin[i]);
        }
    }

    // __bf16 query_state[16][64][256];
    // __bf16 key_state[2][64][256];
    // __bf16 value_state[2][64][256];

    static float y[16][64][256];
    {
        static float q[16][64][256];
        static float k[2][64][256];
        static float v[2][64][256];

        memset(v, 0, sizeof(v));

        for (int h = 0; h < 16; ++h) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < 256; ++j) {
                    q[h][i][j] = (float)query_state[h][i][j];
                }
            }
        }

        for (int h = 0; h < 2; ++h) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < 256; ++j) {
                    k[h][i][j] = (float)key_state[h][i][j];
                    v[h][i][j] = (float)value_state[h][i][j];
                }
            }
        }

        float attn[16][64][64] = {};

        /* att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) */
        for (int h = 0; h < 16; ++h) {
            for (int i = 0; i < n; ++i) {
                /* 1x256 @ 64x256 */
                matmul_f32(attn[h][i], q[h][i], (float *)k[h / 8], 256, 64);
            }
        }

        /* scale */
        for (int h = 0; h < 16; ++h) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < 64; ++j) {
                    attn[h][i][j] *= 0.0625f;
                }
            }
        }

        /* clear upper right for easier debug */
        for (int h = 0; h < 16; ++h) {
            for (int i = 0; i < n; ++i) {
                for (int j = i + 1; j < 64; ++j) {
                    attn[h][i][j] = 0;
                }
            }
        }

        /* soft max */
        for (int h = 0; h < 16; ++h) {
            for (int i = 0; i < n; ++i) {
                float max_att = attn[h][i][0];
                for (int j = 0; j <= i; j++) {
                    if (attn[h][i][j] > max_att)
                        max_att = attn[h][i][j];
                }
                float sum_exp = 0.0f;
                for (int j = 0; j <= i; j++) {
                    attn[h][i][j] = expf(attn[h][i][j] - max_att);
                    sum_exp += attn[h][i][j];
                }
                for (int j = 0; j <= i; j++) {
                    attn[h][i][j] /= sum_exp;
                }
            }
        }

        /* attention is 16x64x64, v is 2x64x256 */
        float v_t[2][256][64];
        for (int h = 0; h < 2; ++h) {
            transpose((float *)v_t[h], (float *)v[h], 64, 256);
        }

        for (int h = 0; h < 16; ++h) {
            for (int i = 0; i < n; ++i) {
                matmul_f32((float *)y[h][i], attn[h][i], (float *)v_t[h / 8], 64, 256);
            }
        }
    }

    static __bf16 tmp[64][16][256];
    memset(tmp, 0, sizeof(tmp));

    /* convert and transpose */
    for (int h = 0; h < 16; ++h) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < 256; ++j) {
                tmp[i][h][j] = (__bf16)y[h][i][j];
            }
        }
    }

    /*
    # manual implementation of attention
    from torch.nn import functional as F
    q = query
    k = key
    v = value
    k_expanded = k.repeat_interleave(8, dim=1)
    att = (q @ k_expanded.transpose(-2, -1)) * 0.0625
    # TODO mask out upper triangle
    seq_len = q.size(-2)
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    att = att.masked_fill(mask, float('-inf'))
    att = F.softmax(att, dim=-1)
    v_expanded = v.repeat_interleave(8, dim=1)
    y = att @ v_expanded
    */

    static __bf16 tmp2[64][4096];
    /* attn_output = attn_output * torch.sigmoid(gate) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < 4096; j++) {
            tmp2[i][j] = ((__bf16 *)tmp[i])[j] * sigmoid(gate[i][j]);
        }
    }

    /* attn_output = self.o_proj(attn_output) */
    for (int i = 0; i < n; ++i) {
        matmul(xout[i], tmp2[i], ow, 4096, 2048);
    }
}

struct qkvz {
    union {
        __bf16 qkv[512]; /* 128 + 128 + 256*/
        struct {
            __bf16 q[128];
            __bf16 k[128];
            __bf16 v[256];
        };
    };
    __bf16 z[256];
};

struct projected_qkvz {
    struct qkvz head[16];
};

struct ba {
    __bf16 b[2];
    __bf16 a[2];
};

struct projected_ba {
    struct ba ba[16];
};

struct conv1d_w {
    __bf16 w[4];
};

// struct linear_conv1d_w {
//     struct conv1d_w d[2048];
// };

float softplus(float x) {
    // For large positive x, use approximation to avoid overflow
    if (x > 20.0f) {
        return x;
    }
    // For large negative x, return 0 to avoid underflow
    if (x < -20.0f) {
        return 0.0f;
    }
    return log1pf(expf(x));
}

void rmsnorm_gated(__bf16 *xout, __bf16 *tmp, __bf16 *zz, __bf16 *w, int n_heads, int head_dim) {
    const float variance_epsilon = 1e-6f;

    float hidden[n_heads * head_dim] = {};
    float gate[n_heads * head_dim] = {};

    // Convert bfloat16 to float for computation
    for (int i = 0; i < n_heads * head_dim; i++) {
        hidden[i] = tmp[i];
        gate[i] = zz[i];
    }

    // Calculate variance per head (across head_dim dimension)
    for (int head = 0; head < n_heads; head++) {
        int start_idx = head * head_dim;

        // Calculate variance for this head
        float variance = 0.0f;
        for (int i = start_idx; i < start_idx + head_dim; i++) {
            variance += hidden[i] * hidden[i];
        }
        variance /= head_dim;

        // Calculate reciprocal square root for this head
        float rsqrt_val = 1.0f / sqrtf(variance + variance_epsilon);

        // Apply normalization and gate for this head
        for (int i = start_idx, j = 0; i < start_idx + head_dim; ++i, ++j) {
            // Normalize
            hidden[i] = hidden[i] * rsqrt_val;

            /* TODO: multiply weight */
            hidden[i] *= w[j];

            // Apply gate with SiLU activation: x * sigmoid(x)
            float sigmoid_gate = 1.0f / (1.0f + expf(-gate[i]));
            hidden[i] = hidden[i] * (gate[i] * sigmoid_gate); // SiLU: gate * sigmoid(gate)
        }
    }

    // Convert back to bfloat16
    for (int i = 0; i < n_heads * head_dim; i++) {
        xout[i] = hidden[i];
    }
}

#pragma GCC push_options
#pragma GCC optimize("O0")

void recurrent_gated_delta_rule(float q[32][128], float k[32][128], float v[32][128], float g[32], float beta[32],
                                int layer, int pos, int n, const struct Transformer *xfmr) {
    const struct Config *c = &xfmr->config;
    struct Runtime *r = &xfmr->runtime;
    const struct Mmapping *m = &xfmr->mmapping;

    float g_exp[32] = {};
    for (int h = 0; h < 32; ++h) {
        g_exp[h] = expf(g[h]);
    }

    /* last_recurrent_state = last_recurrent_state * g_t */
    for (int h = 0; h < 32; ++h) {
        float *recurrent = (float *)r->layers[layer].last_recurrent_state[h]; /* 128x128 */
        for (int j = 0; j < 128 * 128; ++j) {
            recurrent[j] *= g_exp[h];
        }
    }

    float kv_mem_tmp[32][128][128] = {};
    for (int h = 0; h < 32; ++h) {
        float (*recurrent)[128] = r->layers[layer].last_recurrent_state[h]; /* 128x128 */
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                kv_mem_tmp[h][i][j] = k[h][i] * recurrent[i][j];
            }
        }
    }

    float kv_mem[32][128] = {};
    for (int h = 0; h < 32; ++h) {
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                kv_mem[h][j] += kv_mem_tmp[h][i][j];
            }
        }
    }

    /* verified? */

    static float delta[32][128] = {};
    for (int h = 0; h < 32; ++h) {
        for (int j = 0; j < 128; ++j) {
            delta[h][j] = v[h][j] - kv_mem[h][j];
        }
    }

    for (int h = 0; h < 32; ++h) {
        for (int j = 0; j < 128; ++j) {
            delta[h][j] *= beta[h];
        }
    }

    static float tmp[32][128][128] = {};
    for (int h = 0; h < 32; ++h) {
        float *k_t = k[h];
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                tmp[h][i][j] = k_t[i] * delta[h][j];
            }
        }
    }

    for (int h = 0; h < 32; ++h) {
        float (*recurrent)[128] = r->layers[layer].last_recurrent_state[h]; /* 128x128 */
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                recurrent[i][j] += tmp[h][i][j];
            }
        }
    }

    float (*recurrent)[128][128] = (float (*)[128][128])r->layers[layer].last_recurrent_state; /* 32x128x128 */
    static float core_tmp[32][128][128] = {};

    for (int h = 0; h < 32; ++h) {
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                core_tmp[h][i][j] = recurrent[h][i][j] * q[h][i];
            }
        }
    }

    float (*core)[64][128] = (float (*)[64][128]) & r->layers[layer].core_attn_out; /* 32x64x128 */
    for (int h = 0; h < 32; ++h) {
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                core[h][0][j] += core_tmp[h][i][j];
            }
        }
    }

    volatile int dummy = 0;
}

#pragma GCC pop_options

void linear_attention(__bf16 xout[64][2048], __bf16 x[64][2048], const struct Transformer *xfmr, const int layer,
                      const int p, const int n, __bf16 sin[64][64], __bf16 cos[64][64], int prefill) {
    const struct Config *c = &xfmr->config;
    struct Runtime *r = &xfmr->runtime;
    const struct Mmapping *m = &xfmr->mmapping;
    // int pp = pos % 4;

    __bf16 (*qkvz)[64][12288] = (__bf16 (*)[64][12288])r->qkvz;
    __bf16 (*ba)[16][4] = (__bf16 (*)[16][4])r->ba;

    /* qkvz and ba projection */
    for (int i = 0; i < n; ++i) {
        matmul((*qkvz)[i], x[i], m->layers[layer].linear_attn_in_proj_qkvz_w, c->hidden_size, 12288);
    }

    /* ba is 64x(16x2x2) */
    for (int i = 0; i < n; ++i) {
        matmul((__bf16 *)ba[i], x[i], m->layers[layer].linear_attn_in_proj_ba_w, c->hidden_size, 64);
    }

    /* Convert from interleaved to concatenated format */

    __bf16 (*raw)[8192] = r->layers[layer].mixed_qkv_raw;

    for (int kk = p; kk < p + n; ++kk) {

        int k = kk % 64;

        struct projected_qkvz *p_qkvz = (struct projected_qkvz *)(*qkvz)[k];

        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 128; j++) {
                raw[k][0 + i * 128 + j] = p_qkvz->head[i].q[j];
            }
        }
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 128; j++) {
                raw[k][2048 + i * 128 + j] = p_qkvz->head[i].k[j];
            }
        }
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 256; j++) {
                raw[k][4096 + i * 256 + j] = p_qkvz->head[i].v[j];
            }
        }
    }

    /* TODO: save most recent 4 mixed_qkv[pp] for next iterations */

    __bf16 (*mixed)[8192] = r->layers[layer].mixed_qkv;

    /* conv1d over qkv - 8192 dimensions */
    struct conv1d_w *w = (struct conv1d_w *)m->layers[layer].linear_attn_conv1d_w;
    for (int pp = p; pp < p + n; ++pp) {
        for (int i = 0; i < 8192; i++) {
            __bf16 tmp = 0;
            for (int k = 0; k < 4; ++k) {
                __bf16 c1 = raw[(64 - 3 + pp + k) % 64][i]; /* use the zero padding at the end */
                __bf16 c2 = w[i].w[k];
                tmp += c1 * c2;
            }
            mixed[pp][i] = tmp;
        }
    }

    for (int pp = p; pp < p + n; ++pp) {

        int ppp = pp % 64;

        /* silu on all 8192 elements */
        silu_array(mixed[ppp], mixed[ppp], 8192);
    }

    float beta[64][32] = {};
    for (int k = 0; k < 64; ++k) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 2; ++j) {
                beta[k][i * 2 + j] = sigmoid(ba[k][i][0 + j]);
            }
        }
    }

    float a[64][32] = {};
    for (int k = 0; k < 64; ++k) {
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 2; ++j) {
                a[k][i * 2 + j] = ba[k][i][2 + j];
            }
        }
    }

    for (int k = 0; k < 64; ++k) {
        for (int i = 0; i < 32; ++i) {
            a[k][i] = a[k][i] + (float)m->layers[layer].linear_attn_dt_b[i];
        }
        for (int i = 0; i < 32; ++i) {
            a[k][i] = softplus(a[k][i]);
        }
    }

    float nalogexp[64][32] = {};
    for (int k = 0; k < 64; ++k) {
        for (int i = 0; i < 32; ++i) {
            nalogexp[k][i] = -expf((float)m->layers[layer].linear_attn_a_log[i]);
        }
    }

    float g[32][64] = {};
    for (int h = 0; h < 32; ++h) {
        for (int k = 0; k < n; ++k) {
            g[h][k] = nalogexp[k][h] * a[k][h];
        }
    }

    for (int kk = p; kk < p + n; ++kk) {

        int k = kk % 64;

        __bf16 *query, *key, *value;
        query = mixed[k] + 0;
        key = mixed[k] + 2048;
        value = mixed[k] + 4096;

        /* query is normalized and scaled */
        for (int i = 0; i < 2048; i += 128) {
            l2norm_forward(query + i, query + i, 128);
        }

        /* key is normalized not scaled */
        for (int i = 0; i < 2048; i += 128) {
            l2norm_forward(key + i, key + i, 128);
        }
    }

    /*
     * Expand q and k to 32 heads
     * Convert qkv floats
     */

    float q[64][4096] = {}, k[64][4096] = {}, v[64][4096] = {};
    for (int pp = 0; pp < n; ++pp) {

        int index = (p + pp) % 64;

        __bf16 *query, *key, *value;
        query = mixed[index] + 0;
        key = mixed[index] + 2048;
        value = mixed[index] + 4096;

        for (int i = 0; i < 4096; ++i) {
            v[pp][i] = value[i];
        }

        float scale = 1.0f / sqrtf(128);
        for (int h = 0; h < 16; ++h) {
            for (int i = 0; i < 128; ++i) {
                int k1 = (h * 2 + 0) * 128 + i;
                int k2 = (h * 2 + 1) * 128 + i;
                k[pp][k1] = k[pp][k2] = key[h * 128 + i];
                q[pp][k1] = q[pp][k2] = scale * (float)query[h * 128 + i]; /* scale query */
            }
        }
    }

    float v_beta[64][32][128] = {};
    float k_beta[64][32][128] = {};
    for (int p = 0; p < n; ++p) {
        for (int h = 0; h < 32; ++h) {
            for (int j = 0; j < 128; ++j) {
                v_beta[p][h][j] = v[p][h * 128 + j] * beta[p][h];
                k_beta[p][h][j] = k[p][h * 128 + j] * beta[p][h];
            }
        }
    }

    /*
     * g = g.cumsum(dim=-1)
     * g is cumulative sum of decay factors
     */
    /* float g[32][64] = {}; */
    for (int h = 0; h < 32; ++h) {
        for (int i = 1; i < 64; ++i) {
            g[h][i] += g[h][i - 1];
        }
    }

    if (prefill) {
        /* chunk gated delta rule */

        /*
         * decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
         */
        float decay_mask[32][64][64] = {};
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j <= i; ++j) {
                    decay_mask[h][i][j] = expf(g[h][i] - g[h][j]);
                }
            }
        }

        /*
         *  mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
         *  attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
         *  for i in range(1, chunk_size):
         *      row = attn[..., i, :i].clone()
         *      sub = attn[..., :i, :i].clone()
         *      attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
         *  attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
         *
         *  TODO: implement the above more efficient recurrent calculation
         */

        /* tranpose head and seq_len */
        static float qq[32][64][128] = {};
        static float kk[32][64][128] = {};
        static float vv[32][64][128] = {};
        for (int i = 0; i < 64; ++i) {
            for (int h = 0; h < 32; ++h) {
                for (int j = 0; j < 128; ++j) {
                    qq[h][i][j] = q[i][h * 128 + j];
                    kk[h][i][j] = k[i][h * 128 + j];
                    vv[h][i][j] = v[i][h * 128 + j];
                }
            }
        }

        static float kk_beta[32][64][128] = {};
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 128; ++j) {
                    kk_beta[h][i][j] = kk[h][i][j] * beta[i][h];
                }
            }
        }

        static float vv_beta[32][64][128] = {};
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 128; ++j) {
                    vv_beta[h][i][j] = vv[h][i][j] * beta[i][h];
                }
            }
        }

        float attn[32][64][64] = {};

        for (int h = 0; h < 32; ++h) {

            /*
             * k_beta attending to all keys
             * deliberately stop before offset because upper right is masked out
             */
            for (int i = 0; i < 64; ++i) {
                matmul_f32(attn[h][i], kk_beta[h][i], (float *)kk[h], 128, 64);
            }

            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 64; ++j) {
                    attn[h][i][j] = -(attn[h][i][j] * decay_mask[h][i][j]);
                }
            }

            for (int i = 0; i < 64; ++i) {
                attn[h][i][i] = 0.0f;
            }
        }

        for (int i = 1; i < 64; i++) {
            for (int head = 0; head < 32; head++) {
                float row[64];
                for (int j = 0; j < i; j++) {
                    row[j] = attn[head][i][j];
                }

                float sub[64][64];
                for (int k = 0; k < i; k++) {
                    for (int j = 0; j < i; j++) {
                        sub[k][j] = attn[head][k][j];
                    }
                }

                for (int j = 0; j < i; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < i; k++) {
                        sum += row[k] * sub[k][j];
                    }
                    attn[head][i][j] = row[j] + sum;
                }
            }
        }

        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                attn[h][i][i] += 1.0f;
            }
        }

        static float value[32][64][128] = {};
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                float tmp[128][64] = {};
                transpose((float *)tmp, (float *)vv_beta[h], 64, 128);
                matmul_f32(value[h][i], attn[h][i], (float *)tmp, 64, 128);
            }
        }

        /* k_beta * g.exp() */
        /* float g[32][64] = {}; */
        /* static float kk_beta[32][64][128] = {}; */
        static float kk_beta_exp[32][64][128] = {};
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 128; ++j) {
                    kk_beta_exp[h][i][j] = kk_beta[h][i][j] * expf(g[h][i]);
                }
            }
        }

        static float k_cumdecay[32][64][128] = {};
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                float tmp[128][64] = {};
                transpose((float *)tmp, (float *)kk_beta_exp[h], 64, 128);
                matmul_f32(k_cumdecay[h][i], attn[h][i], (float *)tmp, 64, 128);
            }
        }

        /*
         * TODO:
         * last_current_state is 32x128x128
         * core_attn_out is 32x64x128
         * initialized to all zeroes at start.
         */

        static float attn2[32][64][64] = {};
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                matmul_f32(attn2[h][i], qq[h][i], (float *)kk[h], 128, 64);
            }

            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 64; ++j) {
                    attn2[h][i][j] *= decay_mask[h][i][j];
                }
            }
        }

        float (*lrs)[128][128] = r->layers[layer].last_recurrent_state;
        float (*lrs_t)[128][128] = r->layers[layer].last_recurrent_state_transposed;

        for (int h = 0; h < 32; ++h) {
            transpose((float *)lrs_t[h], (float *)lrs[h], 128, 128);
        }

        float (*v_prime)[64][128] = r->layers[layer].v_prime;
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                matmul_f32(v_prime[h][i], k_cumdecay[h][i], (float *)lrs_t[h], 64, 128); /* 1x64 @ 64x128 = 1x128*/
            }
        }

        float (*v_new)[64][128] = r->layers[layer].v_new;
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                subtract_f32(v_new[h][i], value[h][i], v_prime[h][i], 128);
            }
        }

        float (*g_exp)[64] = r->layers[layer].g_exp;
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                g_exp[h][i] = expf(g[h][i]);
            }
        }

        float (*attn_inter)[64][128] = r->layers[layer].attn_inter;
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 128; ++j) {
                    attn_inter[h][i][j] = qq[h][i][j] * g_exp[h][i];
                }
            }
        }

        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                matmul_f32(attn_inter[h][i], attn_inter[h][i], (float *)lrs_t[h], 128, 128);
            }
        }

        /* attn @ v_new */
        float (*core_attn_out)[64][128] = r->layers[layer].core_attn_out;
        {
            float (*v_new_t)[128][64] = r->layers[layer].v_new_transposed;
            for (int h = 0; h < 32; ++h) {
                transpose((float *)v_new_t[h], (float *)v_new[h], 64, 128);
            }

            static float tmp[32][64][128] = {};
            for (int h = 0; h < 32; ++h) {
                for (int i = 0; i < 64; ++i) {
                    matmul_f32(tmp[h][i], attn2[h][i], (float *)v_new_t[h], 64, 128); /* 1x64 @ 64x128 = 1x128*/
                }
            }

            for (int h = 0; h < 32; ++h) {
                for (int i = 0; i < 64; ++i) {
                    for (int j = 0; j < 128; ++j) {
                        core_attn_out[h][i][j] = attn_inter[h][i][j] + tmp[h][i][j];
                    }
                }
            }
        }

        /*
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
            */
        {

            float (*g_tmp)[64] = r->layers[layer].g_tmp;

            /* (g[:, :, i, -1, None] - g[:, :, i]) */
            for (int h = 0; h < 32; ++h) {
                for (int j = 0; j < 64; ++j) {
                    g_tmp[h][j] = g[h][63] - g[h][j];
                }
            }

            for (int h = 0; h < 32; ++h) {
                for (int j = 0; j < 64; ++j) {
                    g_tmp[h][j] = expf(g_tmp[h][j]);
                }
            }

            /* k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp() */
            for (int h = 0; h < 32; ++h) {
                for (int i = 0; i < 64; ++i) {
                    for (int j = 0; j < 128; ++j) {
                        kk[h][i][j] *= g_tmp[h][i];
                    }
                }
            }

            /* lrs_tmp = (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new */
            static float lrs_tmp[32][128][128] = {};
            for (int h = 0; h < 32; ++h) {
                static float tmp[128][64] = {};
                static float v_new_t[128][64] = {};
                transpose((float *)tmp, (float *)kk[h], 64, 128);
                transpose((float *)v_new_t, (float *)v_new[h], 64, 128);
                /* 128x64 @ 64x128 = 128x128 */
                for (int i = 0; i < 128; ++i) {
                    matmul_f32((float *)lrs_tmp[h][i], (float *)tmp[i], (float *)v_new_t, 64, 128); /* 1x128 */
                }
            }

            /* last_recurrent_state * g[:, :, i, -1, None, None].exp() */
            for (int h = 0; h < 32; ++h) {
                for (int i = 0; i < 128; ++i) {
                    for (int j = 0; j < 128; ++j) {
                        lrs[h][i][j] *= expf(g[h][63]);
                    }
                }
            }

            for (int h = 0; h < 32; ++h) {
                for (int i = 0; i < 128; ++i) {
                    for (int j = 0; j < 128; ++j) {
                        lrs[h][i][j] += lrs_tmp[h][i][j];
                    }
                }
            }
        }
    } else {
        /* recurrent gated delta rule */

        float gg[32];
        for (int h = 0; h < 32; ++h) {
            gg[h] = g[h][0];
        }

        recurrent_gated_delta_rule((float (*)[128])q, (float (*)[128])k, (float (*)[128])v, gg, beta[0], layer, p, 1,
                                   xfmr);
    }

    /* convert back to bf16 */
    float (*cao)[64][128] = r->layers[layer].core_attn_out;
    __bf16 (*cao_bf16)[64][128] = r->layers[layer].core_attn_out_bf16;
    for (int h = 0; h < 32; ++h) {
        for (int i = 0; i < 64; ++i) {
            for (int j = 0; j < 128; ++j) {
                cao_bf16[h][i][j] = (__bf16)cao[h][i][j];
            }
        }
    }

    __bf16 (*z)[32][128] = r->layers[layer].z;
    for (int i = 0; i < 64; ++i) {
        struct projected_qkvz *p_qkvz = (struct projected_qkvz *)(*qkvz)[i];
        for (int h = 0; h < 16; ++h) {
            memcpy(z[i][h * 2], p_qkvz->head[h].z, 256 * sizeof(__bf16));
        }
    }

    /* linear attention normalize core_attn_out */

    __bf16 post_lan[64][4096]; /* post linear attention norm */
    for (int i = 0; i < n; ++i) {
        /* Pull out head and dim per token */
        __bf16 tmp[32][128] = {};
        for (int h = 0; h < 32; ++h) {
            for (int j = 0; j < 128; ++j) {
                tmp[h][j] = cao_bf16[h][i][j];
            }
        }

        /* post_lan[i] = 4096, tmp = 32x128, z[i] = 32x128, */
        rmsnorm_gated(post_lan[i], (__bf16 *)tmp, (__bf16 *)z[i], m->layers[layer].linear_attn_norm, 32, 128);
    }

    /* out projection */

    for (int i = 0; i < n; ++i) {
        matmul(xout[i], post_lan[i], m->layers[layer].linear_attn_out_proj_w, 4096, 2048);
    }
}

void *aligned_malloc(size_t alignment, size_t size) {

    if ((alignment & (alignment - 1)) != 0) {
        return NULL;
    }

    if (alignment < sizeof(void *))
        alignment = sizeof(void *);

    void *raw = calloc(1, size + alignment + sizeof(void *) - 1);
    if (!raw)
        return NULL;
    uintptr_t aligned = ((uintptr_t)raw + sizeof(void *) + alignment - 1) & ~(alignment - 1);
    ((void **)aligned)[-1] = raw;

    return (void *)aligned;
}

void aligned_free(void *aligned) {
    if (!aligned)
        return;
    void *raw = ((void **)aligned)[-1];
    free(raw);
}

void runtime_init(struct Transformer *xfmr) {

    const struct Config *c = &xfmr->config;
    struct Runtime *r = &xfmr->runtime;

    int attn_sz = c->n_heads * c->head_dim; // == hidden_size
    int sz = c->head_dim * c->kv_heads;

    r->qg = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * 8192);
    r->q = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * 8192);
    r->gate = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * 8192);
    r->k = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * 8192);
    r->v = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * 8192);

    r->h1 = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * 256);
    r->h2 = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * 256);
    r->h3 = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * 256);

    r->qkvz = (__bf16 *)aligned_malloc(64, 64 * sizeof(__bf16) * 12288);
    r->ba = (__bf16 *)aligned_malloc(64, 64 * sizeof(__bf16) * 64);

    r->layers = (struct RLayer *)calloc(sizeof(struct RLayer), c->n_layers);

    /* chunk decay - multiple of 64 */
    r->g = (float (*)[32])aligned_malloc(64, sizeof(float) * 32 * 12288);

    int chunks = 12288 / 64;
    for (size_t i = 0; i < c->n_layers; i++) {
        r->layers[i].k_cache = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));
        r->layers[i].v_cache = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));
        r->layers[i].v_beta_cache = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));
        r->layers[i].k_beta_cache = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));
        r->layers[i].k_beta_exp_cache = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));
        r->layers[i].value = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));
        r->layers[i].query = (struct Projection *)aligned_malloc(1, sizeof(struct Projection));

        r->layers[i].k_cumdecay = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));
        // r->layers[i].v_prime = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));
        // r->layers[i].v_new = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));

        // r->layers[i].last_recurrent_state = (struct RecurrentState *)calloc(1, sizeof(struct RecurrentState));

        // r->layers[i].core_attn_out = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));

        r->layers[i].g = (struct DecayChunk *)calloc(chunks, sizeof(struct DecayChunk));
        r->layers[i].beta = (struct DecayChunk *)calloc(chunks, sizeof(struct DecayChunk));

        r->layers[i].decay_mask = (struct AttentionChunk *)calloc(chunks, sizeof(struct AttentionChunk));

        r->layers[i].attn_cache = (struct AttentionChunk *)calloc(chunks, sizeof(struct AttentionChunk));
        // r->layers[i].attn_cache2 = (struct AttentionChunk *)calloc(chunks, sizeof(struct AttentionChunk));

        /* self attention */
        r->layers[i].key_cache = (Head *)calloc(sizeof(Head), c->kv_heads);
        r->layers[i].value_cache = (Head *)calloc(sizeof(Head), c->kv_heads);
        for (size_t j = 0; j < c->kv_heads; ++j) {
            r->layers[i].key_cache[j].cache =
                (__bf16 *)aligned_malloc(64, sizeof(__bf16) * c->max_position_embeddings * c->head_dim);
            r->layers[i].value_cache[j].cache =
                (__bf16 *)aligned_malloc(64, sizeof(__bf16) * c->max_position_embeddings * c->head_dim);
        }
    }
}

void token_init(struct Transformer *xfmr, const char *tokenizer_path) {

    char path[FILENAME_MAX];
    snprintf(path, FILENAME_MAX, "weights/tokenizer.bin");

    FILE *f = fopen(path, "rb");
    assert(f != NULL);

    unsigned int vocab_size;
    fread(&vocab_size, sizeof(unsigned int), 1, f);

    xfmr->runtime.lookup = malloc(vocab_size * sizeof(char *));
    assert(xfmr->runtime.lookup);

    for (unsigned int i = 0; i < vocab_size; i++) {
        unsigned int len;
        fread(&len, sizeof(unsigned int), 1, f);

        xfmr->runtime.lookup[i] = malloc(len + 1);
        assert(xfmr->runtime.lookup[i]);

        fread(xfmr->runtime.lookup[i], 1, len, f);
        xfmr->runtime.lookup[i][len] = 0;
    }
    fclose(f);
}

struct ExpertRank {
    int index;
    float score;
};

/* Comparison function for qsort: descending order by score */
int compare_expert_rank_desc(const void *a, const void *b) {
    const struct ExpertRank *ea = (const struct ExpertRank *)a;
    const struct ExpertRank *eb = (const struct ExpertRank *)b;
    if (eb->score > ea->score)
        return 1;
    if (eb->score < ea->score)
        return -1;
    return 0;
}

/* Softmax for float array */
void softmax(float *scores, size_t n) {
    float max = scores[0];
    for (size_t i = 1; i < n; ++i) {
        if (scores[i] > max)
            max = scores[i];
    }
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        scores[i] = expf(scores[i] - max);
        sum += scores[i];
    }
    for (size_t i = 0; i < n; ++i) {
        scores[i] /= sum;
    }
}

__bf16 mlp[64][4096] __attribute__((aligned(64)));
__bf16 mlp1[4096] __attribute__((aligned(64)));
__bf16 mlp2[4096] __attribute__((aligned(64)));
__bf16 mlp3[4096] __attribute__((aligned(64)));
__bf16 mlp4[4096] __attribute__((aligned(64)));
__bf16 expert_output[4096] __attribute__((aligned(64)));
__bf16 seo[64][2048] __attribute__((aligned(64)));
__bf16 logits[64][151936] = {};

int main() {

    struct Transformer xfmr = {};

    struct Config *c = &xfmr.config;
    struct Mmapping *m = &xfmr.mmapping;
    struct Transformer *x = &xfmr;

    config_init(c);
    mmap_init(c, m);
    runtime_init(x);
    token_init(x, 0);

    int attn_hiddn_size = c->n_heads * c->head_dim;

    __bf16 residual[64][2048] __attribute__((aligned(64)));
    __bf16 emb[64][2048] __attribute__((aligned(64)));
    __bf16 emb2[64][2048] __attribute__((aligned(64)));
    __bf16 emb3[64][2048] __attribute__((aligned(64)));
    __bf16 emb4[64][2048] __attribute__((aligned(64)));

    int stop_tokens[3] = {151645, 151644, 151643};

#if 0
    int *tokens = calloc(c->max_position_embeddings, sizeof(__bf16));
    // Read tokens from stdin
    char input_buffer[16384] = {};
    if (fgets(input_buffer, sizeof(input_buffer), stdin) != NULL) {
        char *saveptr;
        char *tok = strtok_r(input_buffer, " \n", &saveptr);
        int idx = 0;
        while (tok && idx < 16384) {
            tokens[idx++] = atoi(tok);
            tok = strtok_r(NULL, " \n", &saveptr);
        }
    }
#else
    int tokens[4096] = {151644, 872, 198, 285, 625, 1535, 264, 11580, 151645, 198, 151644, 77091, 198};
    int prompt_len = 13;
#endif

    __bf16 cos[64][64] = {}, sin[64][64] = {};

    clock_t start_time = clock(); // Start timing

    int pos = 0, prefill = 1;
    while (pos + 1 < c->max_position_embeddings) {

        int n = prefill ? prompt_len : 1;
        for (int i = 0; i < n; ++i) {
            memcpy(emb[i], m->embeddings + tokens[pos + i] * c->hidden_size, c->hidden_size * sizeof(__bf16));
        }

        for (int i = 0; i < n; ++i) {
            rope_forward(c, &c->d.rope, pos + i, cos[i], sin[i]);
        }

        for (int k = 0; k < x->config.n_layers; k++) {

            /* save residual */
            for (int i = 0; i < n; ++i) {
                memcpy(residual[i], emb[i], c->hidden_size * sizeof(__bf16));
            }

            for (int i = 0; i < n; ++i) {
                layernorm(emb2[i], emb[i], m->layers[k].input_layernorm, x);
            }

            if (m->layers[k].q_proj_w) {
                self_attention(emb, emb2, x, k, n, sin, cos);
            } else {
                /* core_attn_out is 32x128, and we allocated 16x256 ... */
                linear_attention(emb, emb2, x, k, pos, n, sin, cos, prefill);
            }

            /* residual connection */
            for (int i = 0; i < n; ++i) {
                add(emb2[i], emb[i], residual[i], c->hidden_size);
            }

            /* save residual */
            for (int i = 0; i < n; ++i) {
                memcpy(residual[i], emb2[i], 2048 * sizeof(__bf16));
            }

            /* post layer norm */
            for (int i = 0; i < n; ++i) {
                layernorm(emb[i], emb2[i], m->layers[k].post_attn_layernorm, x);
            }

            /* gate projection to find experts */
            for (int i = 0; i < n; ++i) {
                matmul(mlp[i], emb[i], m->layers[k].gate, 2048, 512);
            }

            /* top k experts */
            struct ExpertRank ranks[64][c->num_experts];
            for (int i = 0; i < n; ++i) {
                for (size_t j = 0; j < c->num_experts; ++j) {
                    ranks[i][j].index = j;
                    ranks[i][j].score = (float)mlp[i][j];
                }
                qsort(ranks[i], c->num_experts, sizeof(struct ExpertRank), compare_expert_rank_desc);
            }

            /* softmax over top k experts */
            float routing_weights[64][c->num_experts_per_token] = {};
            for (int i = 0; i < n; ++i) {
                for (size_t j = 0; j < c->num_experts_per_token; ++j) {
                    routing_weights[i][j] = ranks[i][j].score;
                }
                softmax(routing_weights[i], c->num_experts_per_token);
            }

            int tpe[512][64] = {};
            int tpe_cnt[512] = {};
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < 10; ++j) {
                    int expert = ranks[i][j].index;
                    tpe[expert][tpe_cnt[expert]++] = i + 1;
                }
            }

            __bf16 final_hidden[64][2048] __attribute__((aligned(64))) = {};

            for (int ex = 0; ex < 512; ++ex) {
                for (int t = 0; t < tpe_cnt[ex]; ++t) {
                    int i = tpe[ex][t] - 1;

                    /* gate projection */
                    matmul(mlp1, emb[i], m->layers[k].experts[ex].gate_proj, 2048, 512);
                    silu_array(mlp2, mlp1, 512);

                    /* up projection */
                    matmul(mlp3, emb[i], m->layers[k].experts[ex].up_proj, 2048, 512);

                    /* hidden */
                    mul(mlp1, mlp2, mlp3, 512);

                    /* down */
                    __bf16 final[2048] = {};
                    matmul(final, mlp1, m->layers[k].experts[ex].down_proj, 512, 2048);

                    /* search through the top k experts for this token to find the routing */
                    int routing = -1;
                    for (int j = 0; j < 10; ++j) {
                        if (ranks[i][j].index == ex) {
                            routing = j;
                            break;
                        }
                    }

                    /* scale by routing weight */
                    float scale = routing_weights[i][routing];
                    for (int j = 0; j < 2048; ++j) {
                        final[j] *= scale;
                    }

                    /* accumulate */
                    for (int j = 0; j < 2048; ++j) {
                        final_hidden[i][j] += final[j];
                    }
                }
            }

            for (int i = 0; i < n; ++i) {

                /* shared_expert_output = self.shared_expert(hidden_states) */

                /* gate projection + silu */
                matmul(mlp1, emb[i], m->layers[k].shared_expert->gate_proj, c->hidden_size, c->moe_intermediate_size);
                silu_array(mlp2, mlp1, c->moe_intermediate_size);

                /* up projection */
                matmul(mlp1, emb[i], m->layers[k].shared_expert->up_proj, c->hidden_size, c->moe_intermediate_size);

                /* product */
                mul(mlp3, mlp2, mlp1, c->moe_intermediate_size);

                /* down projection  */
                matmul(seo[i], mlp3, m->layers[k].shared_expert->down_proj, c->moe_intermediate_size, c->hidden_size);
            }

            /*
             * embeddings -> hidden
             * embeddings2 -> shared_expert_output
             */

            __bf16 gate[64] = {};
            for (int i = 0; i < n; ++i) {
                /* shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output */
                /* gate projection */
                matmul(&gate[i], emb[i], m->layers[k].shared_expert->gate, 2048, 1);
                gate[i] = sigmoid(gate[i]);
            }

            for (int i = 0; i < n; ++i) {
                mul_scalar(seo[i], seo[i], gate[i], c->hidden_size);
            }

            for (int i = 0; i < n; ++i) {
                /* final_hidden_states = final_hidden_states + shared_expert_output */
                add(final_hidden[i], final_hidden[i], seo[i], c->hidden_size);
            }

            for (int i = 0; i < n; ++i) {
                add(emb[i], final_hidden[i], residual[i], c->hidden_size);
            }

            volatile int z = 0;
        }

        for (int i = 0; i < n; ++i) {
            layernorm(emb2[i], emb[i], m->final_layernorm, x);
        }

        const __bf16 *weights = !m->lm_head ? m->embeddings : m->lm_head;
        for (int i = 0; i < n; ++i) {
            matmul(logits[i], emb2[i], weights, c->hidden_size, c->vocab_size);
        }

        // Generate prediction: pick argmax from logits
        int predict[64] = {};
        for (int i = 0; i < n; ++i) {
            float max = (float)logits[i][0];
            for (int j = 1; j < c->vocab_size; j++) {
                if ((float)logits[i][j] > max) {
                    max = (float)logits[i][j];
                    predict[i] = j;
                }
            }
        }
        if (tokens[pos + n] == 0) {
            tokens[pos + n] = predict[n - 1];
            prefill = 0;

            if (tokens[pos + n] == stop_tokens[0] || tokens[pos + 1] == stop_tokens[1] ||
                tokens[pos + n] == stop_tokens[2]) {
                // terminate if we see stop token
                break;
            }
        }

        printf("%s", x->runtime.lookup[tokens[pos + n]]);
        fflush(stdout);

        pos += n;
    }

    clock_t end_time = clock(); // End timing
    double elapsed_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    double tokens_per_sec = (double)pos / elapsed_sec;
    printf("\nTokens processed: %d\n", pos);
    printf("Elapsed time: %.3f seconds\n", elapsed_sec);
    printf("Tokens per second: %.2f\n", tokens_per_sec);
}
