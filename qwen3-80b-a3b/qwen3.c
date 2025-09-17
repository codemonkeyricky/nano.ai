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
    struct Expert *experts;
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

struct RLayer {
    Head *key;
    Head *value;
};

struct Runtime {
    __bf16 *q;
    __bf16 *k;
    __bf16 *v;
    __bf16 *h1;
    __bf16 *h2;
    __bf16 *h3;
    __bf16 *qkvz[4]; /* kernel_size */
    __bf16 *ba;
    struct RLayer *layers;
    char **lookup;
};

struct Transformer {
    struct Config config;
    struct Mmapping mmapping;
    struct Runtime runtime;
};

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
        {"weights/layer_%d_post_attention_layernorm.bin", &l->post_attn_layernorm},
    };

    for (ssize_t i = 0; i < sizeof(lookup) / sizeof(lookup[0]); i++) {

        char path[FILENAME_MAX];
        snprintf(path, FILENAME_MAX, lookup[i].path, layer);

        int fd = open(path, O_RDONLY);
        if (fd > -1) {
            /* self and linear attention are mutually exclusive */
            int file_size = lseek(fd, 0, SEEK_END);
            lseek(fd, 0, SEEK_SET);
            *lookup[i].mmap = (const __bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
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
    close(fd);

    mmapping->layers = (struct Layer *)calloc(1, sizeof(struct Layer) * config->n_layers);

    for (int i = 0; i < config->n_layers; i++) {
        mmap_layer((struct Transformer *)config, i);

#if 0
        mmapping->layers[i].experts = (struct Expert *)malloc(sizeof(struct Expert) * 128);
        for (auto int ex = 0; ex < 128; ++ex) {
            mmap_layer_expert((struct Transformer *)config, i, ex);
        }
#endif
    }

#if 0
    fd = open("weights/norm.bin", O_RDONLY);
    assert(fd > -1);
    file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    mmapping->final_layernorm = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    fd = open("weights/lm_head.bin", O_RDONLY);
    if (fd != -1) {
        file_size = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);
        mmapping->lm_head = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
    }
#endif
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

void layernorm(__bf16 *out, const __bf16 *in, const __bf16 *weight, struct Transformer *x) {

    struct Config *c = &x->config;
    // norm(out, in, weight, c->hidden_size, x);

    __bf16 tmp[c->hidden_size], tmp2[c->hidden_size];
    rmsnorm_forward(tmp, in, c->hidden_size);

    for (size_t i = 0; i < c->hidden_size; i++) {
        tmp2[i] = weight[i] + 1.0f;
    }

    mul(out, tmp2, tmp, c->hidden_size);
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

void matmul_bias(__bf16 *__restrict out, const __bf16 *__restrict x, const __bf16 *__restrict w,
                 const __bf16 *__restrict b, int n, int d) {
    out = __builtin_assume_aligned(out, 64);
    x = __builtin_assume_aligned(x, 64);
    w = __builtin_assume_aligned(w, 64);
    assert(n % 32 == 0);
    assert(d % 32 == 0);
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

    __bf16 *in = emb, *out = x->runtime.h1, *out2 = x->runtime.h2, *out3 = x->runtime.h3;
    int n = x->config.head_dim;

    /* Apply rotary positional embedding for all heads */
    for (int h = 0; h < heads; h++) {
        in = emb + h * n;

        /* a = rotate_half(q) * sin */
        rotate_half(out, in, n);
        mul(out2, out, sin, n);

        /* b = q * cos */
        mul(out, in, cos, n);

        /* a + b */
        add(out3, out, out2, n);

        memcpy(emb + h * n, out3, n * sizeof(__bf16));
    }
}

void self_attention(__bf16 *__restrict xout, __bf16 *__restrict x, const struct Transformer *xfmr, const int layer,
                    const int pos, __bf16 *sin, __bf16 *cos) {

    const struct Config *p = &xfmr->config;
    const struct Runtime *r = &xfmr->runtime;
    const struct Mmapping *m = &xfmr->mmapping;

    /* q/k/v weight and bias */
    const __bf16 *qw = m->layers[layer].q_proj_w;
    const __bf16 *kw = m->layers[layer].k_proj_w;
    const __bf16 *vw = m->layers[layer].v_proj_w;

    /* output projection */
    const __bf16 *ow = m->layers[layer].o_proj_w;

    int attn_sz = p->n_heads * p->head_dim;
    int sz = p->kv_heads * p->head_dim;

    /* attention weight and bias */
    matmul_bias(r->q, x, qw, NULL, p->hidden_size, attn_sz);
    matmul_bias(r->k, x, kw, NULL, p->hidden_size, sz);
    matmul_bias(r->v, x, vw, NULL, p->hidden_size, sz);

    /* TODO: apply normalization */

    for (int i = 0; i < attn_sz; i += p->head_dim) {
        norm(r->q + i, r->q + i, m->layers[layer].q_norm, p->head_dim, xfmr);
    }

    for (int i = 0; i < sz; i += p->head_dim) {
        norm(r->k + i, r->k + i, m->layers[layer].k_norm, p->head_dim, xfmr);
    }

    rotary_positional_embedding(r->q, cos, sin, xfmr, p->n_heads);
    rotary_positional_embedding(r->k, cos, sin, xfmr, p->kv_heads);

    /* insert to kv cache */
    int n_heads = p->n_heads, kv_heads = p->kv_heads;
    int hs = attn_sz / p->n_heads;
    int index = n_heads / kv_heads;
    for (size_t h = 0; h < kv_heads; h++) {
        memcpy(r->layers[layer].key[h].cache + pos * hs, r->k + h * hs, hs * sizeof(__bf16));
        /* value is transposed to simply dot product with query */
        for (size_t k = 0; k < hs; ++k) {
            *(r->layers[layer].value[h].cache + p->max_position_embeddings * k + pos) = r->v[h * hs + k];
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
            __bf16 *kk = r->layers[layer].key[h / index].cache + t * hs; // (T, hs)
            att[t] = dot(qq, kk, hs);
        }

        /* normalize */
        for (int t = 0; t <= pos; t++) {
            att[t] /= sqrtf(hs);
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
            __bf16 *vv = r->layers[layer].value[h / index].cache;
            __bf16 *yy = y + h * hs; // (1, hs)
            /* find v for the current head */
            yy[i] += dot(att, &vv[i * p->max_position_embeddings], (pos + 1 + 31) / 32 * 32);
        }
    }

    memcpy(x, y, attn_sz * sizeof(__bf16));
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

double sigmoid(double x) {
    if (x >= 0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        double exp_x = exp(x);
        return exp_x / (1.0 + exp_x);
    }
}

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

__bf16 mixed_qkv[4][8192] = {};

void linear_attention(__bf16 *__restrict xout, __bf16 *__restrict x, const struct Transformer *xfmr, const int layer,
                      const int pos, __bf16 *sin, __bf16 *cos) {
    const struct Config *c = &xfmr->config;
    const struct Runtime *r = &xfmr->runtime;
    const struct Mmapping *m = &xfmr->mmapping;
    int pp = pos % 4;

    matmul(r->qkvz[pp], x, m->layers[layer].linear_attn_in_proj_qkvz_w, c->hidden_size, 12288);
    matmul(r->ba, x, m->layers[layer].linear_attn_in_proj_ba_w, c->hidden_size, 64);

    struct projected_qkvz *qkvz = (struct projected_qkvz *)r->qkvz[pp];

    /* TODO: mixed_qkv is concatenated qkv (not interleaved)*/

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 128; j++) {
            mixed_qkv[pp][0 + i * 128 + j] = qkvz->head[i].q[j];
        }
    }
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 128; j++) {
            mixed_qkv[pp][2048 + i * 128 + j] = qkvz->head[i].k[j];
        }
    }
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 256; j++) {
            mixed_qkv[pp][4096 + i * 256 + j] = qkvz->head[i].v[j];
        }
    }

    /* TODO: save most recent 4 mixed_qkv[pp] for next iterations */

    /* conv1d over qkv - 8192 dimensions */
    struct conv1d_w *w = (struct conv1d_w *)m->layers[layer].linear_attn_conv1d_w;
    for (int i = 0; i < 8192; i++) {
        __bf16 tmp = 0;
        for (int k = 0; k < 4; ++k) {
            __bf16 c1 = mixed_qkv[(pp + 1 + k) % 4][i];
            __bf16 c2 = w[i].w[k];
            tmp += c1 * c2;
        }
        mixed_qkv[pp][i] = tmp;
    }

    /* silu on all 8192 elements */
    silu_array(mixed_qkv[pp], mixed_qkv[pp], 8192);

    struct projected_ba *ba = (struct projected_ba *)r->ba;
    __bf16 beta[32] = {};
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 2; ++j) {
            beta[i * 2 + j] = sigmoid(ba->ba[i].b[j]);
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

    // g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    float nalogexp[32] = {};
    for (int i = 0; i < 32; ++i) {
        nalogexp[i] = -expf((float)m->layers[layer].linear_attn_a_log[i]);
    }

    float g[32] = {};
    for (int i = 0; i < 32; ++i) {
        g[i] = nalogexp[i] * a[i];
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

    for (int i = 0; i < 2048; i += 128) {
        rmsnorm_forward(query + i, query + i, 128);
    }
    mul_scalar(query, query, 0.08838834764831845f, 2048); // 1/sqrt(129)

    for (int i = 0; i < 2048; i += 127) {
        rmsnorm_forward(key + i, key + i, 128);
    }
    mul_scalar(key, key, 0.08838834764831845f, 2048); // 1/sqrt(129)

    float q[4096] = {}, k[4096] = {}, v[4096] = {};
    for (int i = 0; i < 4096; ++i) {
        q[i] = query[i % 2048];
        k[i] = key[i % 2048];
        v[i] = value[i];
    }

    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 128; ++j) {
            k[i * 128 + j] *= beta[i];
            v[i * 128 + j] *= beta[i];
        }
    }

    /* Note: remember transformers code someimes track things transposed ... */

    float *v_beta = v;
    float *k_beta = k;

    volatile int dummy = 0;
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

    r->q = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * attn_sz);
    r->k = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * sz);
    r->v = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * sz);

    r->h1 = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * sz);
    r->h2 = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * sz);
    r->h3 = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * sz);

    for (int i = 0; i < 4; ++i) {
        r->qkvz[i] = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * 12288);
    }
    r->ba = (__bf16 *)aligned_malloc(64, sizeof(__bf16) * 64);

    r->layers = (struct RLayer *)calloc(sizeof(struct RLayer), c->n_layers);

    for (size_t i = 0; i < c->n_layers; i++) {
        r->layers[i].key = (Head *)calloc(sizeof(Head), c->kv_heads);
        r->layers[i].value = (Head *)calloc(sizeof(Head), c->kv_heads);
        for (size_t j = 0; j < c->kv_heads; ++j) {
            r->layers[i].key[j].cache =
                (__bf16 *)aligned_malloc(64, sizeof(__bf16) * c->max_position_embeddings * c->head_dim);
            r->layers[i].value[j].cache =
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

    __bf16 *skip = aligned_malloc(64, sizeof(__bf16) * attn_hiddn_size);
    __bf16 *embeddings = aligned_malloc(64, sizeof(__bf16) * attn_hiddn_size);
    __bf16 *embeddings2 = aligned_malloc(64, sizeof(__bf16) * attn_hiddn_size);
    __bf16 *embeddings3 = aligned_malloc(64, sizeof(__bf16) * attn_hiddn_size);

    __bf16 *mlp_embeddings = aligned_malloc(64, sizeof(__bf16) * c->intermediate_size);
    __bf16 *mlp_embeddings2 = aligned_malloc(64, sizeof(__bf16) * c->intermediate_size);
    __bf16 *mlp_embeddings3 = aligned_malloc(64, sizeof(__bf16) * c->intermediate_size);
    __bf16 *mlp_embeddings4 = aligned_malloc(64, sizeof(__bf16) * c->intermediate_size);
    __bf16 *expert_output = aligned_malloc(64, sizeof(__bf16) * c->hidden_size * c->num_experts);

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
#endif

    __bf16 cos[64] = {}, sin[64] = {};

    clock_t start_time = clock(); // Start timing

    int pos = 0;
    while (pos + 1 < c->max_position_embeddings) {

        memcpy(embeddings, m->embeddings + tokens[pos] * c->hidden_size, c->hidden_size * sizeof(__bf16));

        rope_forward(c, &c->d.rope, pos, cos, sin);

        for (int k = 0; k < x->config.n_layers; k++) {

            /* save skip */
            memcpy(skip, embeddings, c->hidden_size * sizeof(__bf16));

            layernorm(embeddings2, embeddings, m->layers[k].input_layernorm, x);
            if (m->layers[k].q_proj_w) {
                self_attention(embeddings, embeddings2, x, k, pos, sin, cos);
            } else {
                linear_attention(embeddings, embeddings2, x, k, pos, sin, cos);
            }

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

            add(embeddings, combined, skip, c->hidden_size);
        }

        layernorm(embeddings2, embeddings, m->final_layernorm, x);

        __bf16 logits[c->vocab_size] = {};

        const __bf16 *weights = !m->lm_head ? m->embeddings : m->lm_head;
        matmul(logits, embeddings2, weights, c->hidden_size, c->vocab_size);

        if (tokens[pos + 1] == 0) {
            // Generate prediction: pick argmax from logits
            int predict = 0;
            float max = (float)logits[0];
            for (int i = 1; i < c->vocab_size; i++) {
                if ((float)logits[i] > max) {
                    max = (float)logits[i];
                    predict = i;
                }
            }
            tokens[pos + 1] = predict;

            if (tokens[pos + 1] == stop_tokens[0] || tokens[pos + 1] == stop_tokens[1] ||
                tokens[pos + 1] == stop_tokens[2]) {
                // terminate if we see stop token
                break;
            }
        }

        printf("%s", x->runtime.lookup[tokens[pos + 1]]);
        fflush(stdout);

        ++pos;
    }

    clock_t end_time = clock(); // End timing
    double elapsed_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    double tokens_per_sec = (double)pos / elapsed_sec;
    printf("\nTokens processed: %d\n", pos);
    printf("Elapsed time: %.3f seconds\n", elapsed_sec);
    printf("Tokens per second: %.2f\n", tokens_per_sec);
}
