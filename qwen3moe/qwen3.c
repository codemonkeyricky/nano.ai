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
    const __bf16 *q_proj_w;
    const __bf16 *q_norm;
    const __bf16 *k_proj_w;
    const __bf16 *k_norm;
    const __bf16 *v_proj_w;
    const __bf16 *o_proj_w;
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
    struct RLayer *layers;
    char **lookup;
};

struct Transformer {
    struct Config config;
    struct Mmapping mmapping;
    struct Runtime runtime;
};

void rope_init(struct Config *c) {

    const float t = 5000000.0;
    const int d = c->head_dim; // c->hidden_size / c->n_heads;

    // Calculate inverse frequencies
    for (int i = 0; i < d / 2; i++) {
        float r = (float)(i * 2); // r = 0, 2, 4, ..., d-2
        float exponent = r / (float)d;
        c->d.rope.inv_freq[i] = 1.0f / powf(t, exponent);
    }
}

void rope_forward(struct Config *c, struct RotaryPosEmb *rope, int p, __bf16 *cos, __bf16 *sin) {
    const int d = c->head_dim;
    for (int f = 0; f < d / 2; f++) {
        float freq = (float)p * rope->inv_freq[f];
        cos[f] = cos[d / 2 + f] = cosf(freq);
        sin[f] = sin[d / 2 + f] = sinf(freq);
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
        {"weights/layer_%d_q_proj_w.bin", &l->q_proj_w},
        {"weights/layer_%d_q_norm.bin", &l->q_norm},
        {"weights/layer_%d_k_proj_w.bin", &l->k_proj_w},
        {"weights/layer_%d_k_norm.bin", &l->k_norm},
        {"weights/layer_%d_v_proj_w.bin", &l->v_proj_w},
        {"weights/layer_%d_o_proj_w.bin", &l->o_proj_w},
        {"weights/layer_%d_post_attention_layernorm.bin", &l->post_attn_layernorm},
    };

    for (ssize_t i = 0; i < sizeof(lookup) / sizeof(lookup[0]); i++) {

        char path[FILENAME_MAX];
        snprintf(path, FILENAME_MAX, lookup[i].path, layer);

        int fd = open(path, O_RDONLY);
        assert(fd > -1);
        int file_size = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);
        *lookup[i].mmap = (const __bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
    }
}

void mmap_init(struct Config *config, struct Mmapping *mmapping) {
    int fd = open("weights/embeddings.bin", O_RDONLY);
    assert(fd > -1);
    int file_size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    mmapping->embeddings = (__bf16 *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);

    mmapping->layers = (struct Layer *)malloc(sizeof(struct Layer) * config->n_layers);

    for (int i = 0; i < config->n_layers; i++) {
        mmap_layer((struct Transformer *)config, i);

        mmapping->layers[i].experts = (struct Expert *)malloc(sizeof(struct Expert) * 128);
        for (auto int ex = 0; ex < 128; ++ex) {
            mmap_layer_expert((struct Transformer *)config, i, ex);
        }
    }

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
}

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

    config->d.rope.inv_freq = (float *)malloc(sizeof(float) * (config->head_dim) / 2);

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

void layernorm(__bf16 *out, const __bf16 *in, const __bf16 *weight, struct Transformer *x) {

    struct Config *c = &x->config;
    norm(out, in, weight, c->hidden_size, x);
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
#pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        out[i] = dot(x, &w[i * n], n) + (b ? (float)b[i] : 0);
    }
}

void matmul(__bf16 *__restrict out, const __bf16 *__restrict x, const __bf16 *__restrict w, int n, int d) {
    matmul_bias(out, x, w, 0, n, d);
}

void mul(__bf16 *out, const __bf16 *a, const __bf16 *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = (__bf16)((float)a[i] * (float)b[i]);
    }
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

void *aligned_malloc(size_t alignment, size_t size) {

    if ((alignment & (alignment - 1)) != 0) {
        return NULL;
    }

    if (alignment < sizeof(void *))
        alignment = sizeof(void *);

    void *raw = malloc(size + alignment + sizeof(void *) - 1);
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

void silu_array(__bf16 *output, const __bf16 *input, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i] / (1.0f + expf(-input[i]));
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

#if 1
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

    __bf16 cos[c->head_dim] = {}, sin[c->head_dim] = {};

    clock_t start_time = clock(); // Start timing

    int pos = 0;
    while (pos + 1 < c->max_position_embeddings) {

        memcpy(embeddings, m->embeddings + tokens[pos] * c->hidden_size, c->hidden_size * sizeof(__bf16));

        rope_forward(c, &c->d.rope, pos, cos, sin);

        for (int k = 0; k < x->config.n_layers; k++) {

            /* save skip */
            memcpy(skip, embeddings, c->hidden_size * sizeof(__bf16));

            layernorm(embeddings2, embeddings, m->layers[k].input_layernorm, x);
            self_attention(embeddings, embeddings2, x, k, pos, sin, cos);

            /* residual */
            add(embeddings2, embeddings, skip, c->hidden_size);

            /* save skip */
            memcpy(skip, embeddings2, c->hidden_size * sizeof(__bf16));

            layernorm(embeddings, embeddings2, m->layers[k].post_attn_layernorm, x);

            /* gate projection to find experts */
            matmul(mlp_embeddings, embeddings, m->layers[k].gate, c->hidden_size, 128);

            struct ExpertRank ranks[128];
            for (size_t i = 0; i < 128; ++i) {
                ranks[i].index = i;
                ranks[i].score = (float)mlp_embeddings[i];
            }

            qsort(ranks, 128, sizeof(struct ExpertRank), compare_expert_rank_desc);

            __bf16 experts[8][c->hidden_size] = {};

            for (size_t kk = 0; kk < 8; ++kk) {

                int ex = ranks[kk].index;

                /* gate projection */
                matmul(mlp_embeddings, embeddings, m->layers[k].experts[ex].gate_proj, c->hidden_size, c->moe_intermediate_size);
                silu_array(mlp_embeddings2, mlp_embeddings, c->moe_intermediate_size);

                /* up projection */
                matmul(mlp_embeddings3, embeddings, m->layers[k].experts[ex].up_proj, c->hidden_size, c->moe_intermediate_size);

                /* hidden */
                mul(mlp_embeddings, mlp_embeddings2, mlp_embeddings3, c->moe_intermediate_size);

                /* down */
                matmul(experts[kk], mlp_embeddings, m->layers[k].experts[ex].down_proj, c->moe_intermediate_size, c->hidden_size);
            }

            float top_scores[8] = {};
            for (size_t i = 0; i < 8; ++i) {
                top_scores[i] = ranks[i].score;
            }
            softmax(top_scores, 8);

            __bf16 combined[c->hidden_size] = {};
            for (size_t k = 0; k < 8; ++k) {
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
