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
    __bf16 mixed_qkv_raw[64][8192];
    struct Projection *query;
    struct AttentionChunk *attn_cache;
    struct ProjectionChunk *k_cache;
    struct ProjectionChunk *v_cache;
    struct ProjectionChunk *k_beta_cache;
    struct ProjectionChunk *k_beta_exp_cache;
    struct ProjectionChunk *v_beta_cache;
    struct ProjectionChunk *k_cumdecay;
    struct ProjectionChunk *v_prime;
    struct ProjectionChunk *v_new;
    struct ProjectionChunk *value;
    struct RecurrentState *last_recurrent_state;
    struct ProjectionChunk *core_attn_out;
    struct DecayChunk *beta;
    struct DecayChunk *g;
    struct AttentionChunk *decay_mask;
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

void rotary_positional_embedding(__bf16 *emb, __bf16 *cos, __bf16 *sin, const struct Transformer *x, int heads) {

    __bf16 *in = emb; //  *out = x->runtime.h1, *out2 = x->runtime.h2, *out3 = x->runtime.h3;
    __bf16 out[256], out2[256], out3[256];
    __bf16 cos_padded[256], sin_padded[256];
    for (int i = 0; i < 4; ++i) {
        memcpy(&cos_padded[i * 64], cos, 64 * sizeof(__bf16));
        memcpy(&sin_padded[i * 64], sin, 64 * sizeof(__bf16));
    }

    int n = x->config.head_dim;

    /* Apply rotary positional embedding for all heads */
    for (int h = 0; h < heads; h++) {
        in = emb + h * n;

        /* a = rotate_half(q) * sin */
        rotate_half(out, in, n);
        mul(out2, out, sin_padded, n);

        /* b = q * cos */
        mul(out, in, cos_padded, n);

        /* a + b */
        add(out3, out, out2, n);

        memcpy(emb + h * n, out3, n * sizeof(__bf16));
    }
}

void self_attention(__bf16 *__restrict xout, __bf16 *__restrict x, const struct Transformer *xfmr, const int layer,
                    const int pos, __bf16 *sin, __bf16 *cos) {
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
    matmul(r->qg, x, qw, 2048, 8192);

    /* separate query and gate */
    struct query_gate *qg = (struct query_gate *)r->qg;
    for (int i = 0; i < 16; ++i) {
        memcpy(r->q + i * 256, qg[i].q, 256 * sizeof(__bf16));
        memcpy(r->gate + i * 256, qg[i].gate, 256 * sizeof(__bf16));
    }

    /* query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2) */
    struct projection *query_state = (struct projection *)r->q;
    for (int h = 0; h < 16; ++h) {
        layernorm_n(query_state[h].block, query_state[h].block, m->layers[layer].q_norm, 256, xfmr);
    }

    /* key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2) */
    matmul(r->k, x, kw, 2048, 512);
    struct projection *key_states = (struct projection *)r->k;
    for (int h = 0; h < 2; ++h) {
        layernorm_n(key_states[h].block, key_states[h].block, m->layers[layer].k_norm, 256, xfmr);
    }

    /* value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) */
    matmul(r->v, x, vw, 2048, 512);

    rotary_positional_embedding(r->q, cos, sin, xfmr, 16);
    rotary_positional_embedding(r->k, cos, sin, xfmr, 2);

    volatile int dummy = 0;

    /* insert to kv cache */
    int attn_sz = p->n_heads * p->head_dim;
    int sz = p->kv_heads * p->head_dim;

    int n_heads = p->n_heads, kv_heads = p->kv_heads;
    int hs = attn_sz / p->n_heads;
    int index = n_heads / kv_heads;
    for (size_t h = 0; h < kv_heads; h++) {
        memcpy(r->layers[layer].key_cache[h].cache + pos * hs, r->k + h * hs, hs * sizeof(__bf16));
        /* value is transposed to simply dot product with query */
        for (size_t k = 0; k < hs; ++k) {
            *(r->layers[layer].value_cache[h].cache + p->max_position_embeddings * k + pos) = r->v[h * hs + k];
        }
    }

    /* Calculate attention score */
    __bf16 att[(pos + 1 + 31) / 32 * 32] = {};
    __bf16 *y = xout;
    memset(y, 0, attn_sz * sizeof(__bf16)); // clear output buffer

    for (int h = 0; h < p->n_heads; h++) {

        /* current token query at head h */
        const __bf16 *qq = r->q + h * hs; // (1, hs)
        for (int t = 0; t <= pos; t++) {
            /* send query to all previous keys including current */
            __bf16 *kk = r->layers[layer].key_cache[h / index].cache + t * hs; // (T, hs)
            att[t] = dot(qq, kk, hs);
        }

        /* normalize */
        for (int t = 0; t <= pos; t++) {
            att[t] = att[t] / sqrtf(hs) * 0.0625f;
        }

        /* soft max */
        float max_att = att[0];
        for (int t = 1; t <= pos; t++) {
            if (att[t] > max_att)
                max_att = att[t];
        }
        float sum_exp = 0.0f;
        for (int t = 0; t <= pos; t++) {
            att[t] = expf(att[t] - max_att);
            sum_exp += att[t];
        }
        for (int t = 0; t <= pos; t++) {
            att[t] /= sum_exp;
        }

        /* y = att @ v // (1, T) x (T, hs) -> (1, hs) */
        for (int i = 0; i < hs; i++) {
            __bf16 *vv = r->layers[layer].value_cache[h / index].cache;
            __bf16 *yy = y + h * hs; // (1, hs)
            /* find v for the current head */
            yy[i] += dot(att, &vv[i * p->max_position_embeddings], (pos + 1 + 31) / 32 * 32);
        }
    }

    /* attn_output = attn_output * torch.sigmoid(gate) */
    for (int i = 0; i < attn_sz; i++) {
        x[i] = y[i] * sigmoid(r->gate[i]);
    }

    /* attn_output = self.o_proj(attn_output) */
    matmul(y, x, ow, attn_sz, p->hidden_size);
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

// __bf16 mixed_qkv_raw[4][8192] = {};
__bf16 mixed_qkv[4][8192] = {};

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

// #pragma GCC push_options
// #pragma GCC optimize("O0")

void recurrent_gated_delta_rule(float *g, float *beta, int layer, int pos, const struct Transformer *xfmr) {
    const struct Config *c = &xfmr->config;
    struct Runtime *r = &xfmr->runtime;
    const struct Mmapping *m = &xfmr->mmapping;

    int chunk = pos / 64;
    // int chunk = 0; /* TODO: hack */
    int offset = pos % 64;

    /* r->layers[layer].query */
    /* float *v_cache = r->layers[layer].v_cache[chunk].attn[h][offset]; */
    /* float *k_cache = r->layers[layer].k_cache[chunk].attn[h][offset]; */
    /* g */
    /* beta */

    float g_exp[32] = {};
    for (int i = 0; i < 32; ++i) {
        g_exp[i] = expf(g[i]);
    }

    /* last_recurrent_state = last_recurrent_state * g_t */
    for (int h = 0; h < 32; ++h) {
        float *recurrent = (float *)r->layers[layer].last_recurrent_state->decay[h]; /* 128x128 */
        for (int j = 0; j < 128 * 128; ++j) {
            recurrent[j] *= g_exp[h];
        }
    }

    struct RecurrentState kv_mem_tmp = {};
    for (int h = 0; h < 32; ++h) {
        float (*recurrent)[128] = r->layers[layer].last_recurrent_state->decay[h]; /* 128x128 */
        float *k_t = r->layers[layer].k_cache[chunk].attn[h][offset];
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                kv_mem_tmp.decay[h][i][j] = k_t[i] * recurrent[i][j];
            }
        }
    }

    float kv_mem[32][128] = {};
    for (int h = 0; h < 32; ++h) {
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                kv_mem[h][j] += kv_mem_tmp.decay[h][i][j];
            }
        }
    }

    float delta[32][128] = {};
    struct ProjectionChunk *v_cache = &r->layers[layer].v_cache[chunk];
    for (int h = 0; h < 32; ++h) {
        for (int j = 0; j < 128; ++j) {
            delta[h][j] = v_cache->attn[h][offset][j] - kv_mem[h][j];
        }
    }

    for (int h = 0; h < 32; ++h) {
        for (int j = 0; j < 128; ++j) {
            delta[h][j] *= beta[h];
        }
    }

    float tmp[32][128][128] = {};
    for (int h = 0; h < 32; ++h) {
        float *k_t = r->layers[layer].k_cache[chunk].attn[h][offset];
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                tmp[h][i][j] = k_t[i] * delta[h][j];
            }
        }
    }

    for (int h = 0; h < 32; ++h) {
        float (*recurrent)[128] = r->layers[layer].last_recurrent_state->decay[h]; /* 128x128 */
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                recurrent[i][j] += tmp[h][i][j];
            }
        }
    }

    float (*q)[128] = (float (*)[128])r->layers[layer].query;                                  /* 32x128 */
    float (*recurrent)[128][128] = (float (*)[128][128])r->layers[layer].last_recurrent_state; /* 32x128x128 */
    float core_tmp[32][128][128] = {};

    for (int h = 0; h < 32; ++h) {
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                core_tmp[h][i][j] = recurrent[h][i][j] * q[h][i];
            }
        }
    }

    float (*core)[64][128] = (float (*)[64][128]) & r->layers[layer].core_attn_out[chunk]; /* 32x64x128 */
    for (int h = 0; h < 32; ++h) {
        for (int i = 0; i < 128; ++i) {
            for (int j = 0; j < 128; ++j) {
                core[h][offset][j] += core_tmp[h][i][j];
            }
        }
    }
}

// #pragma GCC pop_options

void linear_attention(__bf16 xout[64][2048], __bf16 x[64][2048], const struct Transformer *xfmr, const int layer,
                      const int n, __bf16 sin[64][64], __bf16 cos[64][64], int prefill) {
    const struct Config *c = &xfmr->config;
    struct Runtime *r = &xfmr->runtime;
    const struct Mmapping *m = &xfmr->mmapping;
    // int pp = pos % 4;

    __bf16 (*qkvz)[64][12288] = (__bf16 (*)[64][12288])r->qkvz;
    __bf16 (*ba)[64][64] = (__bf16 (*)[64][64])r->ba;

    /* qkvz and ba projection */
    for (int i = 0; i < n; ++i) {
        matmul((*qkvz)[i], x[i], m->layers[layer].linear_attn_in_proj_qkvz_w, c->hidden_size, 12288);
    }

    for (int i = 0; i < n; ++i) {
        matmul((*ba)[i], x[i], m->layers[layer].linear_attn_in_proj_ba_w, c->hidden_size, 64);
    }

    /* Convert from interleaved to concatenated format */

    __bf16 (*raw)[8192] = r->layers[layer].mixed_qkv_raw;

    for (int k = 0; k < n; ++k) {

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

#if 0
    /* conv1d over qkv - 8192 dimensions */
    struct conv1d_w *w = (struct conv1d_w *)m->layers[layer].linear_attn_conv1d_w;
    for (int i = 0; i < 8192; i++) {
        __bf16 tmp = 0;
        for (int k = 0; k < 4; ++k) {
            __bf16 c1 = mixed_qkv_raw[(pp + 1 + k) % 4][i];
            __bf16 c2 = w[i].w[k];
            tmp += c1 * c2;
        }
        mixed_qkv[pp][i] = tmp;
    }

    /* silu on all 8192 elements */
    silu_array(mixed_qkv[pp], mixed_qkv[pp], 8192);

    struct projected_ba *ba = (struct projected_ba *)r->ba;
    __bf16 beta_bf16[32] = {};
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 2; ++j) {
            beta_bf16[i * 2 + j] = sigmoid(ba->ba[i].b[j]);
        }
    }

    float a[32] = {};
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 2; ++j) {
            a[i * 2 + j] = ba->ba[i].a[j];
        }
    }

    for (int i = 0; i < 32; ++i) {
        a[i] = a[i] + (float)m->layers[layer].linear_attn_dt_b[i];
    }

    for (int i = 0; i < 32; ++i) {
        a[i] = softplus(a[i]);
    }

    float nalogexp[32] = {};
    for (int i = 0; i < 32; ++i) {
        nalogexp[i] = -expf((float)m->layers[layer].linear_attn_a_log[i]);
    }

    float g[32] = {};
    for (int i = 0; i < 32; ++i) {
        g[i] = nalogexp[i] * a[i];
    }

    float beta[32] = {};
    for (int i = 0; i < 32; ++i) {
        beta[i] = beta_bf16[i];
    }

    /* v shaped as 32 heads x 128 dim */
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 128; j++) {
            qkvz->head[i / 2].v[(i % 2 * 128) + j] *= beta[i];
        }
    }

    __bf16 *query, *key, *value;
    query = mixed_qkv[pp] + 0;
    key = mixed_qkv[pp] + 2048;
    value = mixed_qkv[pp] + 4096;

    /* query is normalized and scaled */
    for (int i = 0; i < 2048; i += 128) {
        l2norm_forward(query + i, query + i, 128);
    }

    /* key is normalized not scaled */
    for (int i = 0; i < 2048; i += 128) {
        l2norm_forward(key + i, key + i, 128);
    }

    /*
     * Expand q and k to 32 heads
     * Convert qkv floats
     */

    float q[4096] = {}, k[4096] = {}, v[4096] = {};
    for (int i = 0; i < 4096; ++i) {
        v[i] = value[i];
    }

    float scale = 1.0f / sqrtf(128);
    for (int h = 0; h < 16; ++h) {
        for (int i = 0; i < 128; ++i) {
            int k1 = (h * 2 + 0) * 128 + i;
            int k2 = (h * 2 + 1) * 128 + i;
            k[k1] = k[k2] = key[h * 128 + i];
            /* scale query */
            q[k1] = q[k2] = scale * (float)query[h * 128 + i];
        }
    }

    int chunk = pos / 64;
    int offset = pos % 64;

    for (int h = 0; h < 32; ++h) {
        struct Projection *qq = r->layers[layer].query;
        memcpy((float *)qq->attn[h], q + h * 128, sizeof(float) * 128);
        volatile int dummy = 0;
    }

    /* insert into key cache, head by head */
    for (int h = 0; h < 32; ++h) {
        float *k_cache = r->layers[layer].k_cache[chunk].attn[h][offset];
        float *key = k + h * 128;
        memcpy(k_cache, key, sizeof(float) * 128);
    }

    for (int h = 0; h < 32; ++h) {
        float *v_cache = r->layers[layer].v_cache[chunk].attn[h][offset];
        float *value = v + h * 128;
        memcpy(v_cache, value, sizeof(float) * 128);
    }

    /*
     * ==========================================================
     */

    /*
     * v_beta = value * beta.unsqueeze(-1)
     * k_beta = key * beta.unsqueeze(-1)
     *
     * k_beta and v_beta are both 32 x 128
     */

    for (int h = 0; h < 32; ++h) {
        float *v = r->layers[layer].v_cache[chunk].attn[h][offset];
        float *k = r->layers[layer].k_cache[chunk].attn[h][offset];
        float *vb = r->layers[layer].v_beta_cache[chunk].attn[h][offset];
        float *kb = r->layers[layer].k_beta_cache[chunk].attn[h][offset];
        for (int j = 0; j < 128; ++j) {
            vb[j] = v[j] * beta[h];
            kb[j] = k[j] * beta[h];
        }
    }

    struct ProjectionChunk *v_beta = r->layers[layer].v_beta_cache;
    struct ProjectionChunk *k_beta = r->layers[layer].k_beta_cache;

    /*
     * g = g.cumsum(dim=-1)
     * g is cumulative sum of decay factors
     */
    for (int h = 0; h < 32; ++h) {
        r->layers[layer].g->decay[h][pos] = g[h];
        if (pos) {
            r->layers[layer].g->decay[h][pos] += r->layers[layer].g->decay[h][pos - 1];
        }

        /* fill in the rest with the last value */
        for (int j = pos + 1; j < 64; ++j) {
            r->layers[layer].g->decay[h][j] = r->layers[layer].g->decay[h][pos];
        }
    }

    if (prefill) {
        /* chunk gated delta rule */

        /*
         * decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
         *
         * The transforrmers code generates decay for all pairs of indices, then
         * apply a trianglar mask to remove the invalid ones (eg. current only
         * decays into the past). The python code creates the matrix for all
         * tokens, for decoding we only need one row.
         *
         * While g can be re-used, decay_mask needs to be re-calculated as current
         * moves forward.
         */
        for (int h = 0; h < 32; ++h) {
            float *decay_mask = r->layers[layer].decay_mask[chunk].attn[h][pos];
            float (*g)[64] = r->layers[layer].g->decay;
            for (int i = 0; i <= pos; ++i) {
                decay_mask[i] = expf(g[h][pos] - g[h][i]);
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

        for (int h = 0; h < 32; ++h) {
            float *attn = (float *)r->layers[layer].attn_cache[chunk].attn[h][offset];
            float *k_beta = (float *)r->layers[layer].k_beta_cache[chunk].attn[h][offset];
            float *k_cache = (float *)r->layers[layer].k_cache[chunk].attn[h];

            /*
             * k_beta attending to all keys
             * deliberately stop before offset because upper right is masked out
             */
            matmul_f32(attn, k_beta, k_cache, 128, offset);

            /* apply decay mask */
            for (int i = 0; i < offset; ++i) {
                attn[i] = -(attn[i] * r->layers[layer].decay_mask[chunk].attn[h][pos][i]);
            }

            float (*attn2)[64] = r->layers[layer].attn_cache[chunk].attn[h];
            float tmp[offset][offset];
            memset(tmp, 0, sizeof(tmp));
            for (int i = 0; i < offset; ++i) {
                /* upper right is cleared */
                for (int j = 0; j < i; ++j) {
                    tmp[i][j] = attn2[i][j];
                }
            }

            /* row.unsqueeze(-1) * sub */
            for (int i = 0; i < offset; ++i) {
                for (int j = 0; j < offset - 1; ++j) {
                    tmp[i][j] *= attn2[offset][i];
                }
            }

            /* vertical summation */
            for (int j = 0; j < offset; j++) {
                float sum = 0.0f;
                for (int i = 0; i < offset; i++) {
                    sum += tmp[i][j];
                }
                attn2[offset][j] += sum;
            }

            /* attn = attn + torch.eye */
            attn[offset] = 1.0f;
        }

        /*
         * value = attn @ v_beta
         * 64x128 = 64x64 @ 64x128
         */
        for (int h = 0; h < 32; ++h) {
            float *attn = (float *)r->layers[layer].attn_cache[chunk].attn[h][offset];
            float *v_beta = (float *)r->layers[layer].v_beta_cache[chunk].attn[h];
            float *value = (float *)r->layers[layer].value[chunk].attn[h][offset];

            /*
             * attn attending to all v_beta
             * 1x64 @ 64x128 = 1x128
             */

            float tmp[128][64] = {};
            transpose((float *)tmp, v_beta, 64, 128);
            matmul_f32(value, attn, (float *)tmp, 64, 128);
        }

        /* k_beta * g.exp() */
        for (int h = 0; h < 32; ++h) {
            float g = r->layers[layer].g[chunk].decay[h][pos];
            float g_exp = expf(g);
            float *k_beta = (float *)r->layers[layer].k_beta_cache[chunk].attn[h][offset];
            float *k_beta_exp = (float *)r->layers[layer].k_beta_exp_cache[chunk].attn[h][offset];
            for (int j = 0; j < 128; ++j) {
                k_beta_exp[j] = k_beta[j] * g_exp;
            }
        }

        /* k_cumdecay = attn @ (k_beta * g.exp()) */
        for (int h = 0; h < 32; ++h) {
            float *attn = (float *)r->layers[layer].attn_cache[chunk].attn[h][offset];
            float *k_beta = (float *)r->layers[layer].k_beta_exp_cache[chunk].attn[h];
            float *k_cumdecay = (float *)r->layers[layer].k_cumdecay[chunk].attn[h][offset];

            /*
             * attn is 1x64
             * (k_beta * g.exp()) is 64x128
             * k_cumdecay is 1x128
             */

            float tmp[128][64] = {};
            transpose((float *)tmp, k_beta, 64, 128);
            matmul_f32(k_cumdecay, attn, (float *)tmp, 64, 128);
        }

        // int chunk = pos / 64;
        // int offset = pos % 64;
        int chunks = ((pos + 1) + 63) / 64;

        // struct RecurrentState emptyRecurrentState = {};
        // *r->layers[layer].last_recurrent_state = emptyRecurrentState;

        struct ProjectionChunk project_chunk_zeroes = {};
        r->layers[layer].core_attn_out[chunk] = project_chunk_zeroes;

        /* attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0) */
        float attention[32][64] = {}; /* attention for current token only */
        for (int h = 0; h < 32; ++h) {
            float *q = r->layers[layer].query->attn[h];                  /* 1x1x128 */
            float *k = (float *)r->layers[layer].k_cache[chunk].attn[h]; /* 1x64x128 */
            float *attn = attention[h];                                  /* 1x64 */
            matmul_f32(attn, q, k, 128, 64);
            mul_f32(attn, attn, r->layers[layer].decay_mask[chunk].attn[h][offset], 64);
        }

        /* TODO: v_prime and v_new requires previous tokens because
         * attention calculation requires them later */

        /* v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state */
        for (int h = 0; h < 32; ++h) {
            float *k_cumdecay = (float *)r->layers[layer].k_cumdecay[chunk].attn[h][offset]; /* 1x128 */
            float *recurrent = (float *)r->layers[layer].last_recurrent_state->decay[h];     /* 128x128 */
            float *v_prime = (float *)r->layers[layer].v_prime[chunk].attn[h][offset];       /* 1x128 */
            float tmp[128][128] = {};
            transpose((float *)tmp, recurrent, 128, 128);
            matmul_f32(v_prime, k_cumdecay, (float *)tmp, 128, 128);
        }

        /* v_new = v_i - v_prime */
        for (int h = 0; h < 32; ++h) {
            float *v_i = (float *)r->layers[layer].value[chunk].attn[h][offset];
            float *v_prime = (float *)r->layers[layer].v_prime[chunk].attn[h][offset]; /* 1x128 */
            float *v_new = (float *)r->layers[layer].v_new[chunk].attn[h][offset];     /* 1x128 */
            subtract_f32(v_new, v_i, v_prime, 128);
        }

        /*
         * attn_inter = (q_i * g[:, :, i, :, None].exp()) @last_recurrent_state
         *
         * q_i -> 32x1x128
         * g.exp -> 32
         */

        float attn_inter[32][128] = {};
        for (int h = 0; h < 32; ++h) {
            float *q = r->layers[layer].query->attn[h]; /* 1x1x128 */
            float g = r->layers[layer].g->decay[h][pos];
            for (int j = 0; j < 128; ++j) {
                attn_inter[h][j] *= expf(g);
            }

            float tmp[128][128] = {};
            float *recurrent = (float *)r->layers[layer].last_recurrent_state->decay[h]; /* 128x128 */
            transpose((float *)tmp, recurrent, 128, 128);
            matmul_f32(attn_inter[h], attn_inter[h], (float *)tmp, 128, 1);
        }

        /* core_attn_out[:, :, i] = attn_inter + attn @ v_new */
        for (int h = 0; h < 32; ++h) {
            float *attn = attention[h]; /* 1x64 */
            float v_new_transposed[128][64] = {};
            float tmp[128] = {};
            float *v_new = (float *)r->layers[layer].v_new[chunk].attn[h]; /* 64x128 */
            transpose((float *)v_new_transposed, v_new, 64, 128);
            matmul_f32(tmp, attn, (float *)v_new_transposed, 64, 128); /* 1x64 @ 64x128 = 1x128*/

            float *core_attn_out = (float *)r->layers[layer].core_attn_out[chunk].attn[h][offset];
            for (int j = 0; j < 128; ++j) {
                core_attn_out[j] = attn_inter[h][j] + tmp[j] /* since attn is 1x1 */;
            }
        }

        /*
         * last_recurrent_state = (
         *   last_recurrent_state * g[:, :, i, -1, None, None].exp()
         *   + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
         */

        /* (g[:, :, i, -1, None] - g[:, :, i]) */
        struct DecayChunk g_copy = *r->layers[layer].g;
        for (int h = 0; h < 32; ++h) {
            float last = g_copy.decay[h][63];
            for (int j = 0; j < 64; ++j) {
                g_copy.decay[h][j] = last - g_copy.decay[h][j];
            }
        }

        /* (g[:, :, i, -1, None] - g[:, :, i]).exp() */
        for (int h = 0; h < 32; ++h) {
            for (int j = 0; j < 64; ++j) {
                g_copy.decay[h][j] = expf(g_copy.decay[h][j]);
            }
        }

        /* k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp() */
        struct ProjectionChunk k_i = r->layers[layer].k_cache[chunk]; /* 32x64x128 */
        for (int h = 0; h < 32; ++h) {
            for (int i = 0; i < 64; ++i) {
                for (int j = 0; j < 128; ++j) {
                    k_i.attn[h][i][j] *= g_copy.decay[h][i];
                }
            }
        }

        struct RecurrentState *last_recurrent_state = r->layers[layer].last_recurrent_state;
        for (int h = 0; h < 32; ++h) {
            float tmp[128][64] = {};
            transpose((float *)tmp, (float *)k_i.attn[h], 64, 128); /* 128x64 */
            for (int i = 0; i < 128; ++i) {
                tmp[i];                                                        /* 1x64 */
                float *v_new = (float *)r->layers[layer].v_new[chunk].attn[h]; /* 64x128 */
                float tmp2[128][64] = {};
                transpose((float *)tmp2, v_new, 64, 128);
                matmul_f32((float *)last_recurrent_state->decay[h][i], (float *)tmp[i], (float *)tmp2, 64,
                           128); /* 1x128 */
            }
        }
    } else {
        /* recurrent gated delta rule */
        recurrent_gated_delta_rule(g, beta, layer, pos, xfmr);
    }

    /* last_recurrent_state * g[:, :, i, -1, None, None].exp() + ...  */

    /* convert back to bf16 */
    __bf16 tmp[32 * 128] = {};
    for (int h = 0; h < 32; ++h) {
        float *core = (float *)r->layers[layer].core_attn_out[chunk].attn[h][offset];
        for (int j = 0; j < 128; ++j) {
            tmp[h * 128 + j] = (__bf16)core[j];
        }
    }

    __bf16 zz[32 * 128] = {};
    for (int h = 0; h < 16; ++h) {
        memcpy(&zz[h * 256], qkvz->head[h].z, 256 * sizeof(__bf16));
    }

    rmsnorm_gated(xout, tmp, zz, m->layers[layer].linear_attn_norm, 32, 128);
    memcpy(tmp, xout, 32 * 128 * sizeof(__bf16));

    matmul(xout, tmp, m->layers[layer].linear_attn_out_proj_w, 4096, c->hidden_size);
#endif
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
        r->layers[i].v_prime = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));
        r->layers[i].v_new = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));

        r->layers[i].last_recurrent_state = (struct RecurrentState *)calloc(1, sizeof(struct RecurrentState));

        r->layers[i].core_attn_out = (struct ProjectionChunk *)calloc(chunks, sizeof(struct ProjectionChunk));

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

    __bf16 (*skip)[64][2048] = aligned_malloc(64, 64 * sizeof(__bf16) * c->hidden_size);
    __bf16 (*emb)[64][2048] = aligned_malloc(64, 64 * sizeof(__bf16) * c->hidden_size);
    __bf16 (*emb2)[64][2048] = aligned_malloc(64, 64 * sizeof(__bf16) * c->hidden_size);
    __bf16 (*emb3)[64][2048] = aligned_malloc(64, 64 * sizeof(__bf16) * c->hidden_size);
    __bf16 (*emb4)[64][2048] = aligned_malloc(64, 64 * sizeof(__bf16) * c->hidden_size);

    __bf16 (*mlp_embeddings)[64][4096] = aligned_malloc(64, 64 * sizeof(__bf16) * c->intermediate_size);
    __bf16 (*mlp_embeddings2)[64][4096] = aligned_malloc(64, 64 * sizeof(__bf16) * c->intermediate_size);
    __bf16 (*mlp_embeddings3)[64][4096] = aligned_malloc(64, 64 * sizeof(__bf16) * c->intermediate_size);
    __bf16 (*mlp_embeddings4)[64][4096] = aligned_malloc(64, 64 * sizeof(__bf16) * c->intermediate_size);
    __bf16 (*expert_output)[64][4096] = aligned_malloc(64, 64 * sizeof(__bf16) * c->hidden_size * c->num_experts);

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
            memcpy((*emb)[i], m->embeddings + tokens[pos + i] * c->hidden_size, c->hidden_size * sizeof(__bf16));
        }

        for (int i = 0; i < n; ++i) {
            rope_forward(c, &c->d.rope, pos + i, cos[i], sin[i]);
        }

        for (int k = 0; k < x->config.n_layers; k++) {

            /* save skip */
            for (int i = 0; i < n; ++i) {
                memcpy(skip, (*emb)[i], c->hidden_size * sizeof(__bf16));
            }

            for (int i = 0; i < n; ++i) {
                layernorm((*emb2)[i], (*emb)[i], m->layers[k].input_layernorm, x);
            }

            if (m->layers[k].q_proj_w) {
#if 0
                self_attention(embeddings, embeddings2, x, k, pos, sin, cos);
#endif
            } else {
                /* core_attn_out is 32x128, and we allocated 16x256 ... */
                linear_attention(*emb, *emb2, x, k, n, sin, cos, prefill);
            }
#if 0
            /* residual */
            add(embeddings2, embeddings, skip, c->hidden_size);

            /* save skip */
            memcpy(skip, embeddings2, c->hidden_size * sizeof(__bf16));

            layernorm(embeddings, embeddings2, m->layers[k].post_attn_layernorm, x);

            /* gate projection to find experts */
            matmul(mlp_embeddings, embeddings, m->layers[k].gate, c->hidden_size, c->num_experts);

            struct ExpertRank ranks[c->num_experts];
            for (size_t i = 0; i < c->num_experts; ++i) {
                ranks[i].index = i;
                ranks[i].score = (float)mlp_embeddings[i];
            }

            qsort(ranks, c->num_experts, sizeof(struct ExpertRank), compare_expert_rank_desc);

            __bf16 experts[c->num_experts_per_token][c->hidden_size] __attribute__((aligned(64))) = {};

            for (size_t kk = 0; kk < c->num_experts_per_token; ++kk) {

                int ex = ranks[kk].index;

                /* gate projection */
                matmul(mlp_embeddings, embeddings, m->layers[k].experts[ex].gate_proj, c->hidden_size,
                       c->moe_intermediate_size);
                silu_array(mlp_embeddings2, mlp_embeddings, c->moe_intermediate_size);

                /* up projection */
                matmul(mlp_embeddings3, embeddings, m->layers[k].experts[ex].up_proj, c->hidden_size,
                       c->moe_intermediate_size);

                /* hidden */
                mul(mlp_embeddings, mlp_embeddings2, mlp_embeddings3, c->moe_intermediate_size);

                /* down */
                matmul(experts[kk], mlp_embeddings, m->layers[k].experts[ex].down_proj, c->moe_intermediate_size,
                       c->hidden_size);

                volatile int dummy = 0;
            }

            float top_scores[c->num_experts_per_token] = {};
            for (size_t i = 0; i < c->num_experts_per_token; ++i) {
                top_scores[i] = ranks[i].score;
            }
            softmax(top_scores, c->num_experts_per_token);

            __bf16 combined[c->hidden_size] = {};
            for (size_t k = 0; k < c->num_experts_per_token; ++k) {
                for (size_t kk = 0; kk < c->hidden_size; ++kk) {
                    combined[kk] += (__bf16)(top_scores[k] * (float)experts[k][kk]);
                }
            }

            /* shared_expert_output = self.shared_expert(hidden_states) */

            /* gate projection + silu */
            matmul(mlp_embeddings, embeddings, m->layers[k].shared_expert->gate_proj, c->hidden_size,
                   c->moe_intermediate_size);
            silu_array(mlp_embeddings2, mlp_embeddings, c->moe_intermediate_size);

            /* up projectoin */
            matmul(mlp_embeddings, embeddings, m->layers[k].shared_expert->up_proj, c->hidden_size,
                   c->moe_intermediate_size);

            /* product */
            mul(mlp_embeddings3, mlp_embeddings2, mlp_embeddings, c->moe_intermediate_size);

            /* down projection  */
            matmul(embeddings2, mlp_embeddings3, m->layers[k].shared_expert->down_proj, c->moe_intermediate_size,
                   c->hidden_size);

            /*
             * embeddings -> hidden
             * embeddings2 -> shared_expert_output
             */

            /* shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output */

            /* gate projection */
            __bf16 gate = 0;
            matmul(&gate, embeddings, m->layers[k].shared_expert->gate, 2048, 1);
            gate = sigmoid(gate);

            mul_scalar(embeddings, embeddings2, gate, c->hidden_size);

            /* final_hidden_states = final_hidden_states + shared_expert_output */
            add(embeddings2, combined, embeddings, c->hidden_size);

            add(embeddings, embeddings2, skip, c->hidden_size);

            volatile int dummy = 0;
#endif
        }

#if 0
        layernorm(embeddings2, embeddings, m->final_layernorm, x);

        __bf16 logits[c->vocab_size] = {};

        const __bf16 *weights = !m->lm_head ? m->embeddings : m->lm_head;
        matmul(logits, embeddings2, weights, c->hidden_size, c->vocab_size);

        // Generate prediction: pick argmax from logits
        int predict = 0;
        float max = (float)logits[0];
        for (int i = 1; i < c->vocab_size; i++) {
            if ((float)logits[i] > max) {
                max = (float)logits[i];
                predict = i;
            }
        }
        if (tokens[pos + 1] == 0) {
            tokens[pos + 1] = predict;
            prefill = 0;

            if (tokens[pos + 1] == stop_tokens[0] || tokens[pos + 1] == stop_tokens[1] ||
                tokens[pos + 1] == stop_tokens[2]) {
                // terminate if we see stop token
                break;
            }
        }

        printf("%s", x->runtime.lookup[tokens[pos + 1]]);
        fflush(stdout);
#endif

        ++pos;
    }

    clock_t end_time = clock(); // End timing
    double elapsed_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    double tokens_per_sec = (double)pos / elapsed_sec;
    printf("\nTokens processed: %d\n", pos);
    printf("Elapsed time: %.3f seconds\n", elapsed_sec);
    printf("Tokens per second: %.2f\n", tokens_per_sec);
}
