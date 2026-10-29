// Graphs the XDNA backend rewrites, computed on the NPU and on the CPU backend
// and compared node by node.
//
//   test-xdna-graph
//
// A prefill layer's residual sum into its normalized projection, ADD,
// RMS_NORM, MUL, MUL_MAT: the backend lays the norm out as the projection's A
// at the MUL and can compute the ADD in the same pass, skipping the ADD where
// it stands. With a second reader of the sum ordered before the norm, ADD,
// SQR, RMS_NORM, MUL, MUL_MAT, that reader must still see the sum - it saw
// whatever the sum's memory held before.
//
// A one-token projection computed again with new activations, on the fused
// route (the decode GEMV) and the per-op one (GGML_XDNA_FUSED_LAYER=0, the
// prefill GEMM): a later graph must not reuse the A an earlier one laid out
// for the same tensor.
//
// A prefill FLASH_ATTN_EXT in the shapes llama builds - the 8 query heads of
// Qwen3.5 over a causal cache, and a bidirectional or banded attention of 4
// query heads with packed sequences, K and V cast to f16 - computed on the
// array and on the CPU backend. The array answers to ~2e-4 NMSE and the host
// path bit for bit, so the two are told apart: a mask the design cannot follow,
// or a call too small (the size rule, tried either side of both its bounds),
// has to come out equal.
//
// BF16 and F16 weight projections, which no quantized route carries, through
// the bf16 GEMM: an M that fits the decode block, one that takes several
// prefill blocks, one whose last block is partial and one whose last block
// holds fewer rows than a sweep of its kernel (it runs a shorter stream); a K
// of one run, of several and one that ends in a partial run; and, in one
// graph, projections of different K and N one after the other (the cores hang
// when consecutive runs differ in their K-loop count).
//
// Every intermediate is cleared before each backend runs, so a node read
// before it was written shows as zeros instead of as the other backend's
// result. Skips (exit 0) when no XDNA device is registered.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

static ggml_backend_dev_t xdna_device(void) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "XDNA") == 0) {
            return dev;
        }
    }
    return nullptr;
}

static std::vector<float> get_f32(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

// normalized mean squared error, as test-backend-ops measures it
static double nmse(const std::vector<float> & ref, const std::vector<float> & got) {
    double err = 0, sq = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        if (!std::isfinite(got[i])) {
            return INFINITY;
        }
        err += ((double) ref[i] - got[i]) * ((double) ref[i] - got[i]);
        sq += (double) ref[i] * ref[i];
    }
    return err / sq;
}

struct norm_graph {
    ggml_context *        ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    ggml_cgraph *         gf  = nullptr;
    std::vector<ggml_tensor *> inputs;
    std::vector<ggml_tensor *> checked;  // compared, in graph order
};

// K x M activations, an N x K Q4_K projection; side_reader adds SQR(sum)
// through a second projection, built first so it is ordered before the norm
static norm_graph build(ggml_backend_t cpu, bool side_reader, int64_t K, int64_t N, int64_t M) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);

    ggml_tensor * a  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * b  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, M);
    ggml_tensor * w  = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, K);
    ggml_tensor * p  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, N);
    ggml_tensor * p2 = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, N);
    g.inputs         = { a, b, w, p, p2 };

    ggml_tensor * sum = ggml_add(g.ctx, a, b);
    ggml_set_name(sum, "sum");
    ggml_tensor * side = nullptr;
    ggml_tensor * sqr  = nullptr;
    if (side_reader) {
        sqr = ggml_sqr(g.ctx, sum);
        ggml_set_name(sqr, "sqr");
        side = ggml_mul_mat(g.ctx, p2, sqr);
        ggml_set_name(side, "side");
    }
    ggml_tensor * norm = ggml_mul(g.ctx, ggml_rms_norm(g.ctx, sum, 1e-6f), w);
    ggml_set_name(norm, "norm");
    ggml_tensor * out = ggml_mul_mat(g.ctx, p, norm);
    ggml_set_name(out, "proj");
    if (side) {
        out = ggml_add(g.ctx, side, out);
        ggml_set_name(out, "out");
    }

    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.checked = { sum };
    if (side_reader) {
        g.checked.push_back(sqr);
        g.checked.push_back(side);
    }
    g.checked.push_back(out);

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, cpu);
    return g;
}

// the activations from `seed`, the weights always the same: a model's weights
// do not change under the address they were packed from
static void set_inputs(norm_graph & g, uint32_t seed) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (size_t i = 0; i < g.inputs.size(); i++) {
        ggml_tensor *      t = g.inputs[i];
        std::mt19937       rng((t->type == GGML_TYPE_F32 ? seed : 1234) * 16 + (uint32_t) i);
        std::vector<float> f(ggml_nelements(t));
        for (float & x : f) {
            x = u(rng);
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<uint8_t> q(ggml_nbytes(t));
            ggml_quantize_chunk(t->type, f.data(), q.data(), 0, t->ne[1], t->ne[0], nullptr);
            ggml_backend_tensor_set(t, q.data(), 0, q.size());
        }
    }
}

// clear everything, then the inputs, and compute on `be`
static bool run(norm_graph & g, ggml_backend_t be, std::vector<std::vector<float>> & res, uint32_t seed = 1234) {
    ggml_backend_buffer_clear(g.buf, 0);
    set_inputs(g, seed);
    if (ggml_backend_graph_compute(be, g.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    res.clear();
    for (const ggml_tensor * t : g.checked) {
        res.push_back(get_f32(t));
    }
    return true;
}

// one Q4_K projection of a single token, the same graph object every run
static norm_graph build_one_token(ggml_backend_t cpu, int64_t K, int64_t N) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    ggml_tensor * x     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * p     = ggml_new_tensor_2d(g.ctx, GGML_TYPE_Q4_K, K, N);
    g.inputs            = { x, p };
    ggml_tensor * out   = ggml_mul_mat(g.ctx, p, x);
    ggml_set_name(out, "proj");
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    g.checked = { out };
    g.buf     = ggml_backend_alloc_ctx_tensors(g.ctx, cpu);
    return g;
}

enum fa_mask { FA_CAUSAL, FA_FULL, FA_BAND, FA_HOLES };

// GGML_XDNA_FA_MM: the host, the array from the size rule's bound, the array whatever the size
enum fa_route { FA_ROUTE_HOST = 0, FA_ROUTE_BY_SIZE = 1, FA_ROUTE_ARRAY = 2 };

struct fa_case {
    const char * name;
    fa_route     route;
    int          heads;
    int          n_tokens;
    int          n_ctx;       // keys of earlier context before the tokens (causal)
    int          pad;         // the keys round up to a multiple of it, hidden
    int          seqs;        // equal sequences packed in the ubatch
    fa_mask      mask;
    int          window;      // FA_BAND: keys on each side
    float        scale;
    bool         cache_view;  // K and V as views of a cache, not a cast
    bool         on_array;    // expected to run on the array
};

static const int FA_KV_HEADS = 2;
static const int FA_HEAD_DIM = 256;

// The graph the model builds around one attention: a projection of n_tokens
// beside it, so the backend sees a prefill graph and starts a new mask epoch.
static norm_graph build_fa(ggml_backend_t cpu, const fa_case & c) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    const int64_t n_kv  = ((int64_t) c.n_ctx + c.n_tokens + c.pad - 1) / c.pad * c.pad;
    const int64_t D     = FA_HEAD_DIM;

    ggml_tensor * qr = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F32, D, c.heads, c.n_tokens);
    ggml_tensor * q  = ggml_permute(g.ctx, qr, 0, 2, 1, 3);
    ggml_tensor * kr, *vr, *k, *v;
    if (c.cache_view) {
        kr = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D, FA_KV_HEADS, n_kv);
        vr = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D, FA_KV_HEADS, n_kv);
        k  = ggml_permute(g.ctx, kr, 0, 2, 1, 3);
        v  = ggml_permute(g.ctx, vr, 0, 2, 1, 3);
    } else {
        kr = k = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D, n_kv, FA_KV_HEADS);
        vr = v = ggml_new_tensor_3d(g.ctx, GGML_TYPE_F16, D, n_kv, FA_KV_HEADS);
    }
    ggml_tensor * m  = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, n_kv, c.n_tokens);
    ggml_tensor * dw = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 64, 64);
    ggml_tensor * dx = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, 64, c.n_tokens);
    g.inputs         = { qr, kr, vr, m, dw, dx };

    ggml_tensor * out = ggml_flash_attn_ext(g.ctx, q, k, v, m, c.scale, 0.0f, 0.0f);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    ggml_set_name(out, "attn");
    g.gf = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.gf, out);
    ggml_build_forward_expand(g.gf, ggml_mul_mat(g.ctx, dw, dx));
    g.checked = { out };
    g.buf     = ggml_backend_alloc_ctx_tensors(g.ctx, cpu);
    return g;
}

// which keys a token sees: the same sequence, and the mask's own rule
static bool fa_visible(const fa_case & c, int t, int j) {
    const int len = c.n_tokens / c.seqs;
    const int s_t = std::min(t / len, c.seqs - 1);
    // keys before n_ctx are the first sequence's earlier positions
    const bool ctx_key = j < c.n_ctx;
    const int  jt      = j - c.n_ctx;
    if (!ctx_key && jt >= c.n_tokens) {
        return false;
    }
    const int s_j  = ctx_key ? 0 : std::min(jt / len, c.seqs - 1);
    const int p_t  = t - s_t * len + (s_t == 0 ? c.n_ctx : 0);
    const int p_j  = ctx_key ? j : jt - s_j * len + (s_j == 0 ? c.n_ctx : 0);
    if (s_t != s_j) {
        return false;
    }
    switch (c.mask) {
        case FA_CAUSAL: return p_j <= p_t;
        case FA_FULL:   return true;
        case FA_BAND:   return std::abs(p_j - p_t) <= c.window;
        case FA_HOLES:  return j % 2 == 0;
    }
    return false;
}

static void set_fa_inputs(norm_graph & g, const fa_case & c) {
    std::mt19937 rng(77);
    // q and k are small, as a model that normalises them leaves them: a score of a few
    std::normal_distribution<float> nd(0.0f, c.scale < 0.5f ? 1.0f : 0.25f);
    for (size_t i = 0; i < 3; i++) {
        ggml_tensor *      t = g.inputs[i];
        std::vector<float> f(ggml_nelements(t));
        for (float & x : f) {
            x = nd(rng);
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
        } else {
            std::vector<ggml_fp16_t> h(f.size());
            ggml_fp32_to_fp16_row(f.data(), h.data(), (int64_t) f.size());
            ggml_backend_tensor_set(t, h.data(), 0, ggml_nbytes(t));
        }
    }
    ggml_tensor *            m = g.inputs[3];
    std::vector<ggml_fp16_t> mk((size_t) ggml_nelements(m), ggml_fp32_to_fp16(-INFINITY));
    for (int t = 0; t < c.n_tokens; t++) {
        for (int64_t j = 0; j < m->ne[0]; j++) {
            if (fa_visible(c, t, (int) j)) {
                mk[(size_t) t * m->ne[0] + j] = ggml_fp32_to_fp16(0.0f);
            }
        }
    }
    ggml_backend_tensor_set(m, mk.data(), 0, ggml_nbytes(m));
    for (size_t i = 4; i < g.inputs.size(); i++) {
        std::vector<float> f(ggml_nelements(g.inputs[i]), 0.5f);
        ggml_backend_tensor_set(g.inputs[i], f.data(), 0, ggml_nbytes(g.inputs[i]));
    }
}

static void set_fa_route(fa_route route) {
    setenv("GGML_XDNA_FA_MM", std::to_string((int) route).c_str(), 1);
}

static bool run_fa(norm_graph & g, const fa_case & c, ggml_backend_t be, std::vector<float> & res) {
    ggml_backend_buffer_clear(g.buf, 0);
    set_fa_inputs(g, c);
    if (ggml_backend_graph_compute(be, g.gf) != GGML_STATUS_SUCCESS) {
        return false;
    }
    res = get_f32(g.checked[0]);
    return true;
}

struct gemm_shape {
    int64_t K;
    int64_t N;
};

// The projections of `shapes` on M activation rows, in this order and with no
// other op between them. The activations are one tensor per distinct K.
static norm_graph build_gemms(ggml_backend_t cpu, enum ggml_type wtype, int64_t M, const std::vector<gemm_shape> & shapes) {
    norm_graph g;
    ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
    g.ctx               = ggml_init(ip);
    g.gf                = ggml_new_graph(g.ctx);

    std::map<int64_t, ggml_tensor *> acts;
    for (const gemm_shape & sh : shapes) {
        ggml_tensor *& a = acts[sh.K];
        if (!a) {
            a = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F32, sh.K, M);
            g.inputs.push_back(a);
        }
        ggml_tensor * w = ggml_new_tensor_2d(g.ctx, wtype, sh.K, sh.N);
        g.inputs.push_back(w);
        ggml_tensor * out = ggml_mul_mat(g.ctx, w, a);
        ggml_set_name(out, (std::string("K") + std::to_string(sh.K) + " N" + std::to_string(sh.N)).c_str());
        ggml_build_forward_expand(g.gf, out);
        g.checked.push_back(out);
    }
    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, cpu);
    return g;
}

struct gemm_case {
    const char *            name;
    enum ggml_type          wtype;
    int64_t                 M;
    std::vector<gemm_shape> shapes;
};

int main(void) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = xdna_device();
    if (!dev) {
        std::printf("test-xdna-graph: no XDNA device, skipping\n");
        return 0;
    }
    ggml_backend_t npu = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!npu || !cpu) {
        std::printf("test-xdna-graph: backend init failed\n");
        return 1;
    }

    // the projection's activations are rounded to bf16 on the NPU and to
    // Q8_K on the CPU, which leaves them 1-2e-4 apart
    const double tol    = 1e-3;
    bool         passed = true;
    for (bool side_reader : { false, true }) {
        norm_graph                      g = build(cpu, side_reader, 1024, 1024, 128);
        std::vector<std::vector<float>> want, got;
        if (!run(g, cpu, want) || !run(g, npu, got)) {
            std::printf("test-xdna-graph: compute failed\n");
            return 1;
        }
        for (size_t i = 0; i < g.checked.size(); i++) {
            const double e  = nmse(want[i], got[i]);
            const bool   ok = e < tol;
            passed          = passed && ok;
            std::printf("%-26s %-5s nmse %.3g %s\n", side_reader ? "ADD SQR RMS_NORM MUL_MAT" : "ADD RMS_NORM MUL_MAT",
                        ggml_get_name(g.checked[i]), e, ok ? "ok" : "FAIL");
        }
        ggml_backend_buffer_free(g.buf);
        ggml_free(g.ctx);
    }

    for (const char * route : { "fused", "per-op" }) {
        // the NPU computes the graph again with new activations; the CPU's
        // answer for each set is computed right before it
        setenv("GGML_XDNA_FUSED_LAYER", std::strcmp(route, "fused") == 0 ? "1" : "0", 1);
        norm_graph g = build_one_token(cpu, 1024, 1024);
        for (uint32_t seed : { 1u, 2u, 3u }) {
            std::vector<std::vector<float>> want, got;
            if (!run(g, cpu, want, seed) || !run(g, npu, got, seed)) {
                std::printf("test-xdna-graph: compute failed\n");
                return 1;
            }
            const double e  = nmse(want[0], got[0]);
            const bool   ok = e < tol;
            passed          = passed && ok;
            std::printf("MUL_MAT one token, %-6s run %u nmse %.3g %s\n", route, seed, e, ok ? "ok" : "FAIL");
        }
        ggml_backend_buffer_free(g.buf);
        ggml_free(g.ctx);
    }
    unsetenv("GGML_XDNA_FUSED_LAYER");

    // the cases that are not about the size rule force the array (or the host), whatever the size; the
    // by-size cases sit either side of its two bounds: n_tokens x n_kv x n_head, then the visible keys x n_head
    const fa_case fa_cases[] = {
        // name                                       route               heads tokens   ctx  pad seqs mask       window scale   cache  array
        { "causal 8 heads, cache view",               FA_ROUTE_ARRAY,        8, 1536,    0, 256,  1, FA_CAUSAL,   0, 0.0625f, true,  true },
        { "causal with 1100 keys of context",         FA_ROUTE_ARRAY,        8,  600, 1100, 256,  1, FA_CAUSAL,   0, 0.0625f, true,  true },
        { "bidirectional, 3 sequences",               FA_ROUTE_ARRAY,        4,  777,    0,   1,  3, FA_FULL,     0, 1.0f,    false, true },
        { "band 512, 2 sequences",                    FA_ROUTE_ARRAY,        4, 1300,    0,   1,  2, FA_BAND,   512, 1.0f,    false, true },
        { "band 40, one sequence",                    FA_ROUTE_ARRAY,        4,  900,    0,   1,  1, FA_BAND,    40, 1.0f,    false, true },
        { "40 sequences of 16 tokens",                FA_ROUTE_ARRAY,        4,  640,    0,   1, 40, FA_FULL,     0, 1.0f,    false, true },
        { "1 head per KV head group, causal",         FA_ROUTE_ARRAY,        2,  700,    0,   1,  1, FA_CAUSAL,   0, 0.0625f, false, true },
        { "below 64 tokens: the host",                FA_ROUTE_ARRAY,        4,   40,    0,   1,  1, FA_FULL,     0, 1.0f,    false, false },
        { "a mask with holes: the host",              FA_ROUTE_ARRAY,        4,  700,    0,   1,  1, FA_HOLES,    0, 1.0f,    false, false },
        { "FA_MM=0: the host at any size",            FA_ROUTE_HOST,         4, 1600,    0,   1,  1, FA_FULL,     0, 1.0f,    false, false },
        { "by size: under the shape bound",           FA_ROUTE_BY_SIZE,      4,  700,    0,   1,  1, FA_FULL,     0, 1.0f,    false, false },
        { "by size: over it, few keys visible",       FA_ROUTE_BY_SIZE,      4, 1600,    0,   1,  1, FA_BAND,    40, 1.0f,    false, false },
        { "by size: over both bounds",                FA_ROUTE_BY_SIZE,      4, 1600,    0,   1,  1, FA_FULL,     0, 1.0f,    false, true },
        { "by size: 8 heads, 1100 keys of context",   FA_ROUTE_BY_SIZE,      8,  600, 1100, 256,  1, FA_CAUSAL,   0, 0.0625f, true,  false },
        { "by size: 8 heads, 1500 keys of context",   FA_ROUTE_BY_SIZE,      8,  600, 1500, 256,  1, FA_CAUSAL,   0, 0.0625f, true,  true },
    };
    for (const fa_case & c : fa_cases) {
        set_fa_route(c.route);
        norm_graph               g = build_fa(cpu, c);
        std::vector<float>       want, got;
        if (!run_fa(g, c, cpu, want) || !run_fa(g, c, npu, got)) {
            std::printf("test-xdna-graph: compute failed\n");
            return 1;
        }
        const double e  = nmse(want, got);
        const bool   ok = c.on_array ? e > 0.0 && e < 1e-3 : e == 0.0;
        passed          = passed && ok;
        std::printf("FLASH_ATTN_EXT %-42s nmse %.3g %s\n", c.name, e, ok ? "ok" : "FAIL");
        ggml_backend_buffer_free(g.buf);
        ggml_free(g.ctx);
    }
    unsetenv("GGML_XDNA_FA_MM");

    const gemm_case gemm_cases[] = {
        { "decode block",     GGML_TYPE_BF16, 15,  { { 512, 1024 }, { 1024, 512 } } },
        { "two blocks",       GGML_TYPE_BF16, 300, { { 512, 1024 }, { 512, 512 } } },
        { "partial block",    GGML_TYPE_BF16, 912, { { 512, 2048 }, { 512, 512 } } },
        { "short tail",       GGML_TYPE_BF16, 1100, { { 512, 512 }, { 1024, 512 } } },
        { "mixed K and N",    GGML_TYPE_BF16, 912, { { 512, 1024 }, { 1024, 512 }, { 2048, 512 }, { 576, 1024 }, { 512, 512 } } },
        { "f16 weights",      GGML_TYPE_F16,  130, { { 1536, 512 }, { 512, 2048 } } },
    };
    for (const gemm_case & c : gemm_cases) {
        norm_graph                      g = build_gemms(cpu, c.wtype, c.M, c.shapes);
        std::vector<std::vector<float>> want, got;
        if (!run(g, cpu, want) || !run(g, npu, got)) {
            std::printf("test-xdna-graph: compute failed\n");
            return 1;
        }
        for (size_t i = 0; i < g.checked.size(); i++) {
            const double e  = nmse(want[i], got[i]);
            const bool   ok = e < tol;
            passed          = passed && ok;
            std::printf("MUL_MAT %-4s M%-4lld %-14s %-12s nmse %.3g %s\n", ggml_type_name(c.wtype), (long long) c.M, c.name,
                        ggml_get_name(g.checked[i]), e, ok ? "ok" : "FAIL");
        }
        ggml_backend_buffer_free(g.buf);
        ggml_free(g.ctx);
    }

    ggml_backend_free(npu);
    ggml_backend_free(cpu);
    std::printf("test-xdna-graph: %s\n", passed ? "passed" : "FAILED");
    return passed ? 0 : 1;
}
