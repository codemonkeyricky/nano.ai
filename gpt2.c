#include <assert.h>
#include <fcntl.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

typedef struct {
    int dim;        // xfmr dimension
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len;    // max sequence length
} Config;

typedef struct {
    const float *ln_1_w;
    const float *ln_1_b;
    const float *attn_c_w;
    const float *attn_c_b;
    const float *c_attn_proj_w;
    const float *c_attn_proj_b;
    const float *wq;
    const float *wk;
    const float *wv;
    const float *ln_2_w;
    const float *ln_2_b;
} Attention;

typedef struct {
    const float *c_fc_w;
    const float *c_fc_b;
    const float *c_proj_w;
    const float *c_proj_b;
} MLP;

typedef struct {
    Attention att;
    MLP mlp;
} Hidden;

typedef struct {
    const float *wte;
    const float *wpe;
    Hidden *h; /* hidden layers */
    const float *ln_f_w;
    const float *ln_f_b;
    const float *lm_head;
} Weights;

typedef struct {
    float *cache;
} Head;

typedef struct {
    Head *key;   // [h, T*hs]
    Head *value; // [h, T*hs]
} Layer;

typedef struct {
    float *emb;
    float *emb2;
    float *emb3;
    float *attn;
    float *mlp;
    Layer *layers;
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    Weights w;     // the weights of the model
    Layer *layers;
    RunState state; // buffers for the "wave" of activations in the forward pass
} Transformer;

typedef struct {
    int index;
    float logit;
} TokenLogit;

TokenLogit *token_logits;
float *probs;

static Transformer xfmr = {};
char **lookup = NULL; // Lookup array for token strings

void matadd(float *z, const float *x, const float *y, int n) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}

void matmul(float *__restrict xout_in, const float *__restrict x_in, const float *__restrict w_in, int n, int d) {

    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    // memset(xout, 0, d * sizeof(float32)); // clear output buffer
    float *xout = (float *)xout_in;
    float *x = (float *)x_in;
    float *w = (float *)w_in;
    for (int i = 0; i < d; i++) {
        xout[i] = 0;
        for (int j = 0; j < n; j++) {
            xout[i] += w[i * n + j] * x[j];
        }
    }
}

void matmul_bias(float *out, const float *x, const float *w, const float *b, int n, int d) {
    matmul(out, x, w, n, d);
    for (size_t i = 0; i < d; i++) {
        out[i] += b[i];
    }
}

void layer_norm(float *output, const float *input, Transformer *xfmr, const float *w, const float *b, float eps) {

    const ssize_t normalized_shape = xfmr->config.dim; // assuming dim is the normalized shape

    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < normalized_shape; i++) {
        mean += input[i];
    }
    mean /= normalized_shape;

    // Compute variance
    float var = 0.0f;
    for (int i = 0; i < normalized_shape; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= normalized_shape;

    // Normalize, scale, and shift
    for (int i = 0; i < normalized_shape; i++) {
        float norm = (input[i] - mean) / sqrtf(var + eps);
        norm *= w[i];
        norm += b[i];
        output[i] = norm;
    }
}

void self_attention(float *__restrict xout, float *__restrict x, const Transformer *xfmr, const int layer,
                    const int pos) {

    const Config *p = &xfmr->config;
    RunState *s = &xfmr->state;
    const Weights *w = &xfmr->w;
    const float *ww = w->h[layer].att.attn_c_w;
    const float *bb = w->h[layer].att.attn_c_b;
    float *attn = xfmr->state.attn;

    /* attention weight and bias */
    matmul_bias(attn, x, ww, bb, p->dim, p->dim * 3);

    /* split attention into q, k, v */
    const float *q = attn;
    const float *k = attn + p->dim;     // key
    const float *v = attn + p->dim * 2; // value

    /* Append current key/value to the cache */
    size_t hs = p->dim / p->n_heads;
    for (size_t h = 0; h < p->n_heads; h++) {
        memcpy(s->layers[layer].key[h].cache + pos * hs, k + h * hs, hs * sizeof(float));
        memcpy(s->layers[layer].value[h].cache + pos * hs, v + h * hs, hs * sizeof(float));
    }

    float *y = xout;
    memset(y, 0, p->dim * sizeof(float)); // clear output buffer

    /* Calculate attention score */
    float att[pos + 1] = {};
    for (int h = 0; h < p->n_heads; h++) {

        /* find the query head */
        const float *qq = q + h * hs; // (1, hs)
        for (int t = 0; t <= pos; t++) {
            float *kk = s->layers[layer].key[h].cache + t * hs; // (T, hs)
            float score = 0.0f;
            for (int i = 0; i < hs; i++) {
                score += qq[i] * kk[i];
            }
            att[t] = score;
        }

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
            float *vv = s->layers[layer].value[h].cache;
            float *yy = y + h * hs; // (1, hs)
            for (int t = 0; t <= pos; t++) {
                /* find v for the current head */
                yy[i] += att[t] * vv[t * hs + i];
            }
        }
    }

    memcpy(x, y, p->dim * sizeof(float));

    ww = w->h[layer].att.c_attn_proj_w; // weight for the projection
    bb = w->h[layer].att.c_attn_proj_b; // bias for the projection
    matmul_bias(y, x, ww, bb, p->dim, p->dim);
}

void mlp(float *__restrict xout, float *__restrict x, Transformer *xfmr, int layer) {

    Config *p = &xfmr->config;
    Weights *w = &xfmr->w;
    RunState *s = &xfmr->state;

    float *tmp = xfmr->state.mlp;

    /* projection */
    matmul_bias(tmp, x, w->h[layer].mlp.c_fc_w, w->h[layer].mlp.c_fc_b, p->dim, p->dim * 4);

    // GELU activation: y = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    for (size_t i = 0; i < p->dim * 4; i++) {
        tmp[i] = 0.5f * tmp[i] * (1.0f + erff(tmp[i] / sqrtf(2.0f)));
    }

    /* projection */
    matmul_bias(xout, tmp, w->h[layer].mlp.c_proj_w, w->h[layer].mlp.c_proj_b, p->dim * 4, p->dim);
}

void forward(float *logits, Transformer *xfmr, int token, int pos) {

    Config *c = &xfmr->config;
    Weights *w = &xfmr->w;
    RunState *s = &xfmr->state;

    float *e1, *e2, *e3;

    e1 = xfmr->state.emb;  // token embedding
    e2 = xfmr->state.emb2; // positional embedding
    e3 = xfmr->state.emb3; // output of the layer norm
    memcpy(e1, w->wte + token * c->dim, c->dim * sizeof(float));
    memcpy(e2, w->wpe + pos * c->dim, c->dim * sizeof(float));
    matadd(e3, e1, e2, c->dim);

    float *tmp1 = e3, *tmp2 = e1, *skip = e2;

    for (size_t layer = 0; layer < c->n_layers; layer++) {

        memcpy(skip, tmp1, c->dim * sizeof(float));

        layer_norm(tmp2, tmp1, xfmr, xfmr->w.h[layer].att.ln_1_w, xfmr->w.h[layer].att.ln_1_b, 1e-5);
        self_attention(tmp1, tmp2, xfmr, layer, pos);

        /* skip + (self_attention(layer_norm(x)) */
        matadd(tmp2, skip, tmp1, c->dim);

        memcpy(skip, tmp2, c->dim * sizeof(float));

        layer_norm(tmp1, tmp2, xfmr, xfmr->w.h[layer].att.ln_2_w, xfmr->w.h[layer].att.ln_2_b, 1e-5);
        mlp(tmp2, tmp1, xfmr, layer);

        /* skip + (mlp(layer_norm(x)) */
        matadd(tmp1, skip, tmp2, c->dim);
    }

    layer_norm(tmp2, tmp1, xfmr, xfmr->w.ln_f_w, xfmr->w.ln_f_b, 1e-5);

    matmul(logits, tmp2, xfmr->w.lm_head, c->dim, xfmr->config.vocab_size);
}

void mmap_head(Transformer *xfmr, const char *loc, int layer) {

    Hidden *head = &xfmr->w.h[layer];

    struct mmap_lookup {
        const char *path;
        const float **mmap;
    };

    struct mmap_lookup lookup[] = {
        {"%s/h.%d.ln_1_weight.bin", &head->att.ln_1_w},
        {"%s/h.%d.ln_1_bias.bin", &head->att.ln_1_b},
        {"%s/h.%d.ln_2_weight.bin", &head->att.ln_2_w},
        {"%s/h.%d.ln_2_bias.bin", &head->att.ln_2_b},
        {"%s/h.%d.attn_c_weight.bin", &head->att.attn_c_w},
        {"%s/h.%d.attn_c_bias.bin", &head->att.attn_c_b},
        {"%s/h.%d.attn_c_proj_weight.bin", &head->att.c_attn_proj_w},
        {"%s/h.%d.attn_c_proj_bias.bin", &head->att.c_attn_proj_b},
        {"%s/h.%d.mlp.c_fc_weight.bin", &head->mlp.c_fc_w},
        {"%s/h.%d.mlp.c_fc_bias.bin", &head->mlp.c_fc_b},
        {"%s/h.%d.mlp.c_proj_weight.bin", &head->mlp.c_proj_w},
        {"%s/h.%d.mlp.c_proj_bias.bin", &head->mlp.c_proj_b},
    };

    for (ssize_t i = 0; i < sizeof(lookup) / sizeof(lookup[0]); i++) {

        char path[FILENAME_MAX];
        snprintf(path, FILENAME_MAX, lookup[i].path, loc, layer);
        lookup[i].path = path;

        int fd = open(lookup[i].path, O_RDONLY);
        assert(fd > -1);
        int file_size = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);
        *lookup[i].mmap = (const float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
    }
}

void mmap_layers(Transformer *xfmr, const char *path) {

    for (ssize_t i = 0; i < xfmr->config.n_layers; i++) {
        mmap_head(xfmr, path, i);
    }
}

void mmap_singles(Transformer *xfmr, const char *path) {

    struct mmap_lookup {
        const char *path;
        const float **mmap;
    };

    struct mmap_lookup lookup[] = {
        {"%s/wte_weight.bin", &xfmr->w.wte},         {"%s/wpe_weight.bin", &xfmr->w.wpe},
        {"%s/ln_f_weight.bin", &xfmr->w.ln_f_w},     {"%s/ln_f_bias.bin", &xfmr->w.ln_f_b},
        {"%s/lm_head_weight.bin", &xfmr->w.lm_head},
    };

    for (ssize_t i = 0; i < sizeof(lookup) / sizeof(lookup[0]); i++) {

        char to_mmap_path[FILENAME_MAX];
        snprintf(to_mmap_path, FILENAME_MAX, lookup[i].path, path);

        int fd = open(to_mmap_path, O_RDONLY);
        assert(fd > -1);
        off_t file_size = lseek(fd, 0, SEEK_END);
        lseek(fd, 0, SEEK_SET);
        *lookup[i].mmap = (const float *)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
    }
}

void mmap_weights(Transformer *xfmr, const char *path) {

    mmap_singles(xfmr, path);

    xfmr->w.h = (Hidden *)calloc(1, sizeof(Hidden) * xfmr->config.n_layers);
    mmap_layers(xfmr, path);
}

void load_config(Transformer *xfmr, const char *config_path) {

    char path[FILENAME_MAX];
    snprintf(path, FILENAME_MAX, "%s/config.bin", config_path);

    FILE *f = fopen(path, "rb");
    assert(f != NULL);

    int vals[5];
    size_t n = fread(vals, sizeof(int), 5, f);
    assert(n == 5);
    fclose(f);

    xfmr->config.vocab_size = vals[0];
    xfmr->config.seq_len = vals[1];
    xfmr->config.dim = vals[2];
    xfmr->config.n_layers = vals[3];
    xfmr->config.n_heads = vals[4];
}

void set_config(Transformer *xfmr, const char *config_path) {

    Config *c = &xfmr->config;

    load_config(xfmr, config_path);
}

void allocate_runtime(Transformer *xfmr) {

    Config *c = &xfmr->config;
    RunState *s = &xfmr->state;

    s->layers = (Layer *)calloc(c->n_layers, sizeof(Layer));
    for (size_t i = 0; i < c->n_layers; i++) {
        s->layers[i].key = (Head *)calloc(sizeof(Head), c->n_heads);
        s->layers[i].value = (Head *)calloc(sizeof(Head), c->n_heads);
        for (size_t j = 0; j < c->n_heads; ++j) {
            s->layers[i].key[j].cache = (float *)calloc(sizeof(float), c->seq_len * (c->dim / c->n_heads));
            s->layers[i].value[j].cache = (float *)calloc(sizeof(float), c->seq_len * (c->dim / c->n_heads));
        }
    }

    s->emb = (float *)calloc(c->dim, sizeof(float));
    s->emb2 = (float *)calloc(c->dim, sizeof(float));
    s->emb3 = (float *)calloc(c->dim, sizeof(float));
    s->attn = (float *)calloc(c->dim * 3, sizeof(float));
    s->mlp = (float *)calloc(c->dim * 4, sizeof(float));
}

void free_runtime(Transformer *xfmr) {

    RunState *s = &xfmr->state;

    free(xfmr->w.h);

    for (size_t i = 0; i < xfmr->config.n_layers; i++) {
        for (size_t j = 0; j < xfmr->config.n_heads; j++) {
            free(s->layers[i].key[j].cache);
            free(s->layers[i].value[j].cache);
        }
        free(s->layers[i].key);
        free(s->layers[i].value);
    }
    free(s->layers);

    free(s->emb);
    free(s->emb2);
    free(s->emb3);
    free(s->attn);
    free(s->mlp);
}

void token_init(const char *tokenizer_path) {

    char path[FILENAME_MAX];
    snprintf(path, FILENAME_MAX, "%s/tokenizer.bin", tokenizer_path);

    FILE *f = fopen(path, "rb");
    assert(f != NULL);

    unsigned int vocab_size;
    fread(&vocab_size, sizeof(unsigned int), 1, f);

    lookup = malloc(vocab_size * sizeof(char *));
    assert(lookup);

    for (unsigned int i = 0; i < vocab_size; i++) {
        unsigned int len;
        fread(&len, sizeof(unsigned int), 1, f);

        lookup[i] = malloc(len + 1);
        assert(lookup[i]);

        fread(lookup[i], 1, len, f);
        lookup[i][len] = 0;
    }
    fclose(f);
}

void init() {

    set_config(&xfmr, "gpt2_weights");

    allocate_runtime(&xfmr);

    mmap_weights(&xfmr, "gpt2_weights");

    token_init("gpt2_weights");
}

void generate(float *logits, int token, int pos) { forward(logits, &xfmr, token, pos); }

void deinit() { free_runtime(&xfmr); }

// Comparison function for sorting in descending order
int compare_logits(const void *a, const void *b) {
    TokenLogit *ta = (TokenLogit *)a;
    TokenLogit *tb = (TokenLogit *)b;
    if (tb->logit > ta->logit)
        return 1;
    if (tb->logit < ta->logit)
        return -1;
    return 0;
}

// Samples a token using Top-k sampling
int sample_top_k(float *logits, int vocab_size, int top_k) {

    for (int i = 0; i < vocab_size; i++) {
        token_logits[i].index = i;
        token_logits[i].logit = logits[i];
    }

    // Sort by logit (descending)
    qsort(token_logits, vocab_size, sizeof(TokenLogit), compare_logits);

    // Keep only top-k tokens
    if (top_k > vocab_size)
        top_k = vocab_size;

    // Compute softmax probabilities
    float max_logit = token_logits[0].logit;
    float sum_exp = 0.0f;

    for (int i = 0; i < top_k; i++) {
        probs[i] = expf(token_logits[i].logit - max_logit);
        sum_exp += probs[i];
    }

    // Normalize and sample
    float r = (float)rand() / (float)RAND_MAX;
    float cum_prob = 0.0f;
    int sampled_token = token_logits[0].index;

    for (int i = 0; i < top_k; i++) {
        probs[i] /= sum_exp;
        cum_prob += probs[i];
        if (r <= cum_prob) {
            sampled_token = token_logits[i].index;
            break;
        }
    }

    return sampled_token;
}

void token_deinit() {
    if (lookup) {
        const int vocab_size = xfmr.config.vocab_size;
        for (int i = 0; i < vocab_size; i++) {
            free(lookup[i]);
        }
        free(lookup);
        lookup = NULL;
    }
    if (token_logits)
        free(token_logits);
    if (probs)
        free(probs);
}

int main() {

    srand(time(NULL)); // Seed randomness

    init();

    int k = 0, token = 198;                        // Starting token (e.g., '\n')
    const int top_k = 40;                          // Typical value for Top-k
    const int vocab_size = xfmr.config.vocab_size; // GPT-2 vocab size

    token_logits = malloc(vocab_size * sizeof(TokenLogit));
    probs = malloc(top_k * sizeof(float));

    float *logits = malloc(vocab_size * sizeof(float));
    while (k < 500) {
        generate(logits, token, k++);

        // Top-k sampling
        int next_token = sample_top_k(logits, vocab_size, top_k);

        printf("%s", lookup[token]);
        fflush(stdout);

        token = next_token;
    }
    free(logits);

    deinit();

    token_deinit();
    return 0;
}