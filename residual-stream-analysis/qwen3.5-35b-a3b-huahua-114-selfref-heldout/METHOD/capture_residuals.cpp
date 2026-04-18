// capture_residuals.cpp — fork of capture_activations.cpp.
// Forked at scripts/reference/capture_activations.cpp in the active repo.
//
// Purpose: dump, per prompt and per token, the tensor the MoE router is reading at
// layers 13 / 14 / 15, alongside the router's logits at those same layers.
//
// Target architecture: qwen35moe (Qwen 3.5 MoE hybrid) — NOT qwen3moe. Qwen 3.5 is a
// hybrid architecture with Gated Delta Net recurrent layers interleaved with full
// attention layers, and shared experts alongside routed experts. In its graph builder
// (src/models/qwen35moe.cpp:56-60) the pre-router norm is:
//   ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, ...);
//   cb(attn_post_norm, "attn_post_norm", il);      // <-- this is the tap
//   cur = build_layer_ffn(attn_post_norm, il);      // passes it to build_moe_ffn -> build_lora_mm(gate_inp, cur)
// So the residual tensor the router reads is named `attn_post_norm-<il>`, NOT `ffn_norm-<il>`.
// (qwen3moe uses `ffn_norm`; qwen35moe uses `attn_post_norm`. Different model, different name.)
//
// In HuggingFace/transformer convention, `attn_post_norm` is the `post_attention_layernorm`
// output (the second norm in the block, applied after attention and before the FFN/MoE branch).
// It is NOT `input_layernorm`, which is the norm applied before attention.
//
// Tap mechanism: named via the standard llm_graph_context::cb() -> graph_get_cb() ->
// ggml_format_name chain at graph-build time. No llama.cpp source patch required.
//
// Precision: all tensor data is written as fp32 little-endian (NumPy dtype `<f4`). Source
// tensors may be f16/bf16/f32; get_f32() upcasts to fp32 before buffering. Downstream scripts
// must load with `numpy.load(...).astype(np.float32)` equivalent — expect no dtype surprises.
//
// Output per prompt (under <output-dir>/<prompt_id>/):
//   router/   ffn_moe_logits-13.npy,   ffn_moe_logits-14.npy,   ffn_moe_logits-15.npy   (n_rows x 256)
//   residual/ attn_post_norm-13.npy,   attn_post_norm-14.npy,   attn_post_norm-15.npy   (n_rows x n_embd)
//   metadata.txt, prompt_tokens.json, generated_tokens.json, generated_text.txt
// File names match the graph tensor name directly — no rename at save time. Enforced:
// n_rows == len(prompt_tokens) + len(generated_tokens) for every tensor; exactly 6 tensors per
// successful prompt in default mode (3 router + 3 residual). Anything else fails the capture
// and the partial output directory is removed.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "sampling.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <set>
#include <cstdlib>

namespace fs = std::filesystem;

// ================================================================
// Target layers — hard-coded for this pipeline. See CLAUDE.md "Why layer 14 (not 26)".
// ================================================================
static const std::set<int> TARGET_LAYERS = {13, 14, 15};
static constexpr int EXPECTED_TENSORS_PER_PROMPT = 6; // 3 router + 3 residual

// ================================================================
// NumPy .npy writer (v1.0, float32, C-contiguous) — unchanged from baseline
// ================================================================
static bool write_npy(const std::string & path, const float * data,
                      const std::vector<int64_t> & shape) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return false;

    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); i++) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size() || shape.size() == 1) shape_str += ",";
    }
    shape_str += ")";

    std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shape_str + ", }";
    int total = 10 + (int) header.size() + 1;
    int pad = 64 - (total % 64);
    if (pad == 64) pad = 0;
    header += std::string(pad, ' ');
    header += '\n';

    uint16_t header_len = (uint16_t) header.size();
    f.write("\x93NUMPY", 6);
    uint8_t ver[2] = {1, 0};
    f.write((char *) ver, 2);
    f.write((char *) &header_len, 2);
    f.write(header.data(), header_len);

    size_t n_elements = 1;
    for (auto s : shape) n_elements *= s;
    f.write((const char *) data, n_elements * sizeof(float));
    return f.good();
}

static std::string json_escape(const std::string & in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (unsigned char c : in) {
        switch (c) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char) c);
                }
                break;
        }
    }
    return out;
}

// ================================================================
// Tensor name classification — layer-gated. `ffn_norm-<il>` and `ffn_moe_logits-<il>` are named
// by graph_get_cb() in src/llama-context.cpp; match them and filter to TARGET_LAYERS only.
// ================================================================
enum tensor_kind { TK_NONE = 0, TK_ROUTER, TK_RESIDUAL };

static int parse_layer_suffix(const char * name, const char * prefix) {
    size_t plen = std::strlen(prefix);
    if (std::strncmp(name, prefix, plen) != 0) return -1;
    if (name[plen] != '-') return -1;
    int il = -1;
    // match the rest as a signed integer; reject trailing garbage
    char * end = nullptr;
    long v = std::strtol(name + plen + 1, &end, 10);
    if (end == nullptr || *end != '\0') return -1;
    il = (int) v;
    return il;
}

static tensor_kind classify_tensor(const char * name) {
    int il = parse_layer_suffix(name, "ffn_moe_logits");
    if (il >= 0 && TARGET_LAYERS.count(il)) return TK_ROUTER;

    // Qwen 3.5 MoE emits the pre-router norm output as `attn_post_norm-<il>`, not `ffn_norm-<il>`.
    // See file-level comment for the source location and why this differs from qwen3moe.
    il = parse_layer_suffix(name, "attn_post_norm");
    if (il >= 0 && TARGET_LAYERS.count(il)) return TK_RESIDUAL;

    return TK_NONE;
}

static bool is_router_tensor_any(const char * name) {
    // for expert-bias application: matches router logits at any layer (not just target),
    // because --expert-bias should still perturb the network identically regardless of layer filter.
    return std::strncmp(name, "ffn_moe_logits", 14) == 0;
}

// ================================================================
// Tensor accumulator — buffers data across decode calls
// ================================================================
struct tensor_record {
    std::string name;
    std::vector<float> data;
    int64_t dim;
    int64_t n_tokens;
    tensor_kind kind;
};

// ================================================================
// Capture state — shared between main loop and callback
// ================================================================
struct capture_state {
    std::map<std::string, tensor_record> tensors;
    std::set<std::string> discovered_tensor_names;
    std::set<std::string> logged_shape_mismatch;
    std::string output_dir;
    std::string current_prompt_id;
    bool prompt_has_error = false;
    bool list_tensors_mode = false;
    bool routing_only = false;
    bool verbose = false;
    int tensors_seen = 0;
    int tensors_captured = 0;
    std::vector<uint8_t> xfer_buf;
    std::map<int, float> expert_biases;
};

// ================================================================
// Float extraction (handles F32 / F16 / BF16) — unchanged from baseline
// ================================================================
static float get_f32(const uint8_t * data, ggml_type type, const size_t * nb,
                     int64_t i0, int64_t i1) {
    size_t off = (size_t) (i1 * nb[1] + i0 * nb[0]);
    switch (type) {
        case GGML_TYPE_F32:  return *(const float *) &data[off];
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(*(const ggml_fp16_t *) &data[off]);
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(*(const ggml_bf16_t *) &data[off]);
        default:             return 0.0f;
    }
}

static void set_f32(uint8_t * data, ggml_type type, const size_t * nb,
                    int64_t i0, int64_t i1, float value) {
    size_t off = (size_t) (i1 * nb[1] + i0 * nb[0]);
    switch (type) {
        case GGML_TYPE_F32:
            *(float *) &data[off] = value;
            break;
        case GGML_TYPE_F16:
            *(ggml_fp16_t *) &data[off] = ggml_fp32_to_fp16(value);
            break;
        case GGML_TYPE_BF16:
            *(ggml_bf16_t *) &data[off] = ggml_fp32_to_bf16(value);
            break;
        default:
            break;
    }
}

static bool parse_expert_bias_spec(const std::string & spec, std::map<int, float> & out) {
    out.clear();
    if (spec.empty()) {
        return true;
    }

    size_t start = 0;
    while (start < spec.size()) {
        size_t end = spec.find(',', start);
        std::string item = spec.substr(start, end == std::string::npos ? std::string::npos : end - start);
        size_t colon = item.find(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 >= item.size()) {
            return false;
        }

        // NOTE: the baseline at scripts/reference/capture_activations.cpp has undefined behavior here —
        // it calls .c_str() on `item.substr(...)` temporaries, storing the returned end-pointer into
        // expert_tail/bias_tail, then dereferences those pointers AFTER the temporaries have been
        // destroyed at end-of-expression. We materialize the substrings into named strings so their
        // buffers outlive the strtol/strtof calls and the tail-pointer checks.
        std::string expert_s = item.substr(0, colon);
        std::string bias_s   = item.substr(colon + 1);
        char * expert_tail = nullptr;
        char * bias_tail = nullptr;
        long expert = std::strtol(expert_s.c_str(), &expert_tail, 10);
        float bias = std::strtof(bias_s.c_str(), &bias_tail);
        if (!expert_tail || *expert_tail != '\0' || !bias_tail || *bias_tail != '\0') {
            return false;
        }
        if (expert < 0) {
            return false;
        }
        out[(int) expert] = bias;

        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }

    return true;
}

static void apply_expert_biases(std::vector<float> & f32, int64_t n_tokens, int64_t dim,
                                const std::map<int, float> & expert_biases) {
    if (expert_biases.empty()) {
        return;
    }

    for (const auto & [expert, bias] : expert_biases) {
        if (expert < 0 || expert >= dim) {
            continue;
        }
        for (int64_t tok = 0; tok < n_tokens; tok++) {
            f32[tok * dim + expert] += bias;
        }
    }
}

// ================================================================
// Eval callback — fires for every tensor in the compute graph
// ================================================================
static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;

    if (ask) {
        if (st->list_tensors_mode) return true;
        // Baseline parity for --expert-bias: the perturbation must reach every router layer, not only
        // the target-layer subset, otherwise the model's downstream computation diverges from what the
        // baseline would produce under the same spec. Request the tensor in write mode for *any*
        // layer's router logits whenever biasing is active; the write phase applies the bias and then
        // skips storage via classify_tensor() == TK_NONE for non-target layers.
        if (!st->expert_biases.empty() && is_router_tensor_any(t->name)) return true;
        if (st->routing_only) {
            return classify_tensor(t->name) == TK_ROUTER;
        }
        return classify_tensor(t->name) != TK_NONE;
    }

    st->tensors_seen++;

    if (st->list_tensors_mode) {
        st->discovered_tensor_names.insert(t->name);
        int nd = ggml_n_dims(t);
        LOG("[TENSOR] %-50s  type=%-6s  dims=[", t->name, ggml_type_name(t->type));
        for (int d = 0; d < nd; d++) LOG("%lld%s", (long long) t->ne[d], d < nd - 1 ? ", " : "");
        tensor_kind k = classify_tensor(t->name);
        LOG("]  %s%s\n",
            k == TK_ROUTER ? "ROUTER" : "",
            k == TK_RESIDUAL ? "RESIDUAL" : "");
        return true;
    }

    if (ggml_is_quantized(t->type)) return true;

    const bool on_host = ggml_backend_buffer_is_host(t->buffer);
    size_t nbytes = ggml_nbytes(t);
    if (!on_host) {
        st->xfer_buf.resize(nbytes);
        ggml_backend_tensor_get(t, st->xfer_buf.data(), 0, nbytes);
    }
    uint8_t * src = on_host ? (uint8_t *) t->data : st->xfer_buf.data();

    int64_t dim = t->ne[0];
    int64_t n_tokens = (ggml_n_dims(t) > 1) ? t->ne[1] : 1;

    size_t chunk = (size_t) (n_tokens * dim);
    std::vector<float> f32(chunk);
    for (int64_t tok = 0; tok < n_tokens; tok++) {
        for (int64_t d = 0; d < dim; d++) {
            f32[tok * dim + d] = get_f32(src, t->type, t->nb, d, tok);
        }
    }

    // Expert-bias injection: perturbs router logits in-place before top-k selection.
    // Applied to any layer's router (matches baseline behavior), not only the target layers —
    // otherwise --expert-bias would silently diverge from the reference run's semantics.
    if (is_router_tensor_any(t->name) && !st->expert_biases.empty()) {
        apply_expert_biases(f32, n_tokens, dim, st->expert_biases);
        for (int64_t tok = 0; tok < n_tokens; tok++) {
            for (int64_t d = 0; d < dim; d++) {
                set_f32(src, t->type, t->nb, d, tok, f32[tok * dim + d]);
            }
        }
        if (!on_host) {
            ggml_backend_tensor_set(t, src, 0, nbytes);
        }
    }

    tensor_kind kind = classify_tensor(t->name);
    if (kind == TK_NONE) {
        // Either we said `false` in the `ask` phase and somehow got called anyway (shouldn't happen),
        // or this is a router at a non-target layer that we needed to bias-perturb but not store.
        return true;
    }

    std::string name(t->name);
    auto it = st->tensors.find(name);
    if (it != st->tensors.end()) {
        auto & rec = it->second;
        if (rec.dim != dim) {
            if (st->logged_shape_mismatch.insert(name).second) {
                LOG_ERR("  tensor dim mismatch for %s: expected %lld got %lld\n",
                        name.c_str(), (long long) rec.dim, (long long) dim);
            }
            st->prompt_has_error = true;
            return true;
        }
        rec.data.insert(rec.data.end(), f32.begin(), f32.end());
        rec.n_tokens += n_tokens;
    } else {
        tensor_record rec;
        rec.name = name;
        rec.data = std::move(f32);
        rec.dim = dim;
        rec.n_tokens = n_tokens;
        rec.kind = kind;
        st->tensors[name] = std::move(rec);
    }

    st->tensors_captured++;
    return true;
}

// ================================================================
// Save all captured tensors for one prompt
// ================================================================
struct generated_token_info {
    int32_t token_id;
    std::string piece;
};

struct prompt_token_info {
    int32_t index;
    int32_t token_id;
    std::string piece;
    int32_t start_char;
    int32_t end_char;
};

static std::vector<prompt_token_info> build_prompt_token_info(
        const llama_vocab * vocab,
        const std::vector<llama_token> & tokens,
        const std::string & prompt_text) {
    std::vector<prompt_token_info> out;
    out.reserve(tokens.size());

    size_t cursor = 0;
    for (size_t i = 0; i < tokens.size(); i++) {
        char buf[512];
        int n = llama_token_to_piece(vocab, tokens[i], buf, sizeof(buf), 0, true);
        std::string piece(buf, n > 0 ? n : 0);

        int32_t start_char = -1;
        int32_t end_char = -1;

        if (!piece.empty()) {
            const bool matches_prompt =
                cursor + piece.size() <= prompt_text.size() &&
                prompt_text.compare(cursor, piece.size(), piece) == 0;
            if (matches_prompt) {
                start_char = (int32_t) cursor;
                cursor += piece.size();
                end_char = (int32_t) cursor;
            } else if (cursor > 0) {
                LOG_ERR("  prompt token span mismatch at token %zu ('%s') for %s\n",
                        i, piece.c_str(), prompt_text.substr(cursor, 64).c_str());
            }
        }

        out.push_back({
            (int32_t) i,
            (int32_t) tokens[i],
            piece,
            start_char,
            end_char,
        });
    }

    if (cursor != prompt_text.size()) {
        LOG_ERR("  prompt token spans consumed %zu/%zu chars; metadata boundaries may be unavailable\n",
                cursor, prompt_text.size());
    }

    return out;
}

static bool save_prompt(capture_state & st, const std::string & prompt_text,
                        const std::string & generated_text,
                        const std::vector<prompt_token_info> & prompt_tokens,
                        const std::vector<generated_token_info> & generated_tokens,
                        int n_tok_prompt, int n_tok_gen, double ms) {
    fs::path base = fs::path(st.output_dir);
    int n_router = 0, n_residual = 0;
    bool write_ok = true;
    const size_t n_in_state = st.tensors.size();

    for (auto & [name, rec] : st.tensors) {
        const char * subdir = (rec.kind == TK_ROUTER) ? "router" : "residual";
        fs::path dir = base / subdir;
        fs::create_directories(dir);

        // File name matches the graph tensor name directly (e.g. `attn_post_norm-14.npy`,
        // `ffn_moe_logits-14.npy`). No rename — the qwen35moe graph builder already uses a
        // name that's unambiguous about which norm is being dumped (post-attention, pre-MoE).
        std::string fname = name;
        for (char & c : fname) {
            if (c == '/' || c == ' ' || c == '.') c = '_';
        }

        std::vector<int64_t> shape = {rec.n_tokens, rec.dim};
        std::string out_path = (dir / (fname + ".npy")).string();
        if (write_npy(out_path, rec.data.data(), shape)) {
            if (rec.kind == TK_ROUTER) n_router++;
            else if (rec.kind == TK_RESIDUAL) n_residual++;
        } else {
            LOG_ERR("  write_npy FAILED for %s (disk full / permission / partial write?)\n",
                    out_path.c_str());
            write_ok = false;
        }
    }

    // Defensive count check: the number of tensors we wrote must equal the number we had in state
    // (every entry must have produced a file). This is redundant with the per-write failure flag
    // above — kept because a silent mismatch would otherwise only surface during downstream analysis.
    if ((size_t) (n_router + n_residual) != n_in_state) {
        LOG_ERR("  write count mismatch: %d router + %d residual written, %zu in state\n",
                n_router, n_residual, n_in_state);
        write_ok = false;
    }

    if (!write_ok) {
        return false;
    }

    {
        std::ofstream f((base / "metadata.txt").string());
        std::string esc = prompt_text;
        std::replace(esc.begin(), esc.end(), '\n', ' ');
        f << "prompt_id=" << st.current_prompt_id << "\n"
          << "prompt=" << esc << "\n"
          << "n_tokens_prompt=" << n_tok_prompt << "\n"
          << "n_tokens_generated=" << n_tok_gen << "\n"
          << "n_router_tensors=" << n_router << "\n"
          << "n_residual_tensors=" << n_residual << "\n"
          << "target_layers=13,14,15\n"
          << "elapsed_ms=" << (int) ms << "\n";
    }

    {
        std::ofstream f((base / "prompt_tokens.json").string());
        f << "[\n";
        for (size_t i = 0; i < prompt_tokens.size(); i++) {
            const auto & tok = prompt_tokens[i];
            f << "  {\"index\": " << tok.index
              << ", \"token_id\": " << tok.token_id
              << ", \"piece\": \"" << json_escape(tok.piece) << "\""
              << ", \"start_char\": ";
            if (tok.start_char >= 0) {
                f << tok.start_char;
            } else {
                f << "null";
            }
            f << ", \"end_char\": ";
            if (tok.end_char >= 0) {
                f << tok.end_char;
            } else {
                f << "null";
            }
            f << "}";
            if (i + 1 < prompt_tokens.size()) {
                f << ",";
            }
            f << "\n";
        }
        f << "]\n";
    }

    if (!generated_text.empty()) {
        std::ofstream f((base / "generated_text.txt").string());
        f << generated_text;
    }

    {
        std::ofstream f((base / "generated_tokens.json").string());
        f << "[\n";
        for (size_t i = 0; i < generated_tokens.size(); i++) {
            const auto & tok = generated_tokens[i];
            f << "  {\"step\": " << i
              << ", \"token_id\": " << tok.token_id
              << ", \"piece\": \"" << json_escape(tok.piece) << "\"}";
            if (i + 1 < generated_tokens.size()) {
                f << ",";
            }
            f << "\n";
        }
        f << "]\n";
    }

    LOG_INF("  saved %d router + %d residual tensors → %s\n",
            n_router, n_residual, st.output_dir.c_str());
    return true;
}

// ================================================================
// Parse tab-separated prompt file — unchanged from baseline
// ================================================================
struct prompt_entry {
    std::string id;
    std::string text;
};

static std::string unescape_prompt(const std::string & raw) {
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
        if (raw[i] == '\\' && i + 1 < raw.size()) {
            char next = raw[i + 1];
            if (next == 'n') { out.push_back('\n'); i++; continue; }
            if (next == 't') { out.push_back('\t'); i++; continue; }
            if (next == '\\') { out.push_back('\\'); i++; continue; }
        }
        out.push_back(raw[i]);
    }
    return out;
}

static std::vector<prompt_entry> parse_prompt_file(const std::string & path) {
    std::vector<prompt_entry> out;
    std::ifstream f(path);
    if (!f.is_open()) {
        LOG_ERR("failed to open prompt file: %s\n", path.c_str());
        return out;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        out.push_back({line.substr(0, tab), unescape_prompt(line.substr(tab + 1))});
    }
    return out;
}

static std::string sanitize_prompt_id(const std::string & raw, size_t fallback_idx) {
    if (raw.empty()) {
        return "prompt_" + std::to_string(fallback_idx + 1);
    }

    std::string out;
    out.reserve(raw.size());
    for (char c : raw) {
        const bool ok = (c >= 'a' && c <= 'z') ||
                        (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') ||
                        c == '_' || c == '-';
        out.push_back(ok ? c : '_');
    }

    if (out.empty()) {
        out = "prompt_" + std::to_string(fallback_idx + 1);
    }
    return out;
}

// ================================================================
// Per-prompt outcome + run manifest
// ================================================================
// Written as <output-dir>/capture_manifest.json after the main loop completes. Downstream scripts
// that iterate the output directory MUST NOT assume every subdir is a valid capture — instead they
// should consult this manifest to learn which prompt IDs succeeded. Exit code alone is not enough:
// the process may have exited 1 (partial failure) and a naive listing of raw/*/ would silently
// under-count.
struct prompt_result {
    std::string prompt_id;         // original from TSV (or "single")
    std::string safe_id;           // sanitized on-disk directory name
    bool succeeded;
    std::string failure_reason;    // empty if succeeded; short machine-readable code
    int n_tokens_prompt;           // 0 if failure occurred before tokenization
    int n_tokens_generated;        // 0 if failure occurred before generation
    double elapsed_ms;             // per-prompt wall time; 0 if t0 was never set
};

static void write_manifest(const std::vector<prompt_result> & results,
                           const std::string & output_dir,
                           bool list_tensors_run) {
    int n_succ = 0, n_fail = 0;
    for (const auto & r : results) {
        if (r.succeeded) n_succ++;
        else n_fail++;
    }

    fs::create_directories(output_dir);
    std::ofstream f((fs::path(output_dir) / "capture_manifest.json").string());
    if (!f.is_open()) {
        LOG_ERR("failed to write capture_manifest.json under %s\n", output_dir.c_str());
        return;
    }

    f << "{\n";
    f << "  \"run_summary\": {\n";
    f << "    \"n_prompts_processed\": " << results.size() << ",\n";
    f << "    \"n_succeeded\": " << n_succ << ",\n";
    f << "    \"n_failed\": "    << n_fail << ",\n";
    f << "    \"list_tensors_run\": " << (list_tensors_run ? "true" : "false") << "\n";
    f << "  },\n";
    f << "  \"prompts\": [\n";
    for (size_t i = 0; i < results.size(); i++) {
        const auto & r = results[i];
        f << "    {\n";
        f << "      \"prompt_id\": \""       << json_escape(r.prompt_id) << "\",\n";
        f << "      \"safe_id\": \""         << json_escape(r.safe_id)   << "\",\n";
        f << "      \"status\": \""          << (r.succeeded ? "succeeded" : "failed") << "\",\n";
        f << "      \"failure_reason\": ";
        if (r.succeeded) f << "null";
        else             f << "\"" << json_escape(r.failure_reason) << "\"";
        f << ",\n";
        f << "      \"n_tokens_prompt\": "    << r.n_tokens_prompt    << ",\n";
        f << "      \"n_tokens_generated\": " << r.n_tokens_generated << ",\n";
        f << "      \"elapsed_ms\": "         << (long long) r.elapsed_ms << "\n";
        f << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    f << "  ]\n";
    f << "}\n";
}

// ================================================================
// Pre-parse custom CLI flags, strip them from argv
// ================================================================
struct custom_args {
    std::string prompt_file;
    std::string output_dir = "./captures";
    bool list_tensors = false;
    bool routing_only = false;
    bool no_stream = false;
    std::string expert_bias_spec;
};

static custom_args strip_custom_args(int & argc, char **& argv) {
    custom_args ca;
    std::vector<char *> keep;
    keep.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--prompt-file" && i + 1 < argc) {
            ca.prompt_file = argv[++i];
        } else if ((a == "-o" || a == "--output-dir") && i + 1 < argc) {
            ca.output_dir = argv[++i];
        } else if (a == "--list-tensors") {
            ca.list_tensors = true;
        } else if (a == "--routing-only") {
            ca.routing_only = true;
        } else if (a == "--no-stream") {
            ca.no_stream = true;
        } else if (a == "--expert-bias" && i + 1 < argc) {
            ca.expert_bias_spec = argv[++i];
        } else {
            keep.push_back(argv[i]);
        }
    }

    argc = (int) keep.size();
    for (int i = 0; i < argc; i++) argv[i] = keep[i];
    return ca;
}

// ================================================================
// Main
// ================================================================
int main(int argc, char ** argv) {
    custom_args ca = strip_custom_args(argc, argv);

    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();

    capture_state state;
    state.list_tensors_mode = ca.list_tensors;
    state.routing_only = ca.routing_only;
    state.verbose = (params.verbosity > 0);
    if (!parse_expert_bias_spec(ca.expert_bias_spec, state.expert_biases)) {
        LOG_ERR("Failed to parse --expert-bias '%s' (expected expert:bias[,expert:bias...])\n",
                ca.expert_bias_spec.c_str());
        return 1;
    }

    params.cb_eval = capture_cb;
    params.cb_eval_user_data = &state;
    params.warmup = false;

    LOG_INF("=== Qwen3 MoE Residual + Router Capture (layers 13/14/15) ===\n");
    LOG_INF("Output dir   : %s\n", ca.output_dir.c_str());
    LOG_INF("n_predict    : %d\n", params.n_predict);
    LOG_INF("Text output  : %s\n", ca.no_stream ? "file only (--no-stream)" : "stdout + file");
    LOG_INF("Target layers: {13, 14, 15}\n");
    if (ca.list_tensors) LOG_INF("Mode         : --list-tensors (discovery)\n");
    if (ca.routing_only) LOG_INF("Mode         : --routing-only (router tensors only; skip residual)\n");
    if (!state.expert_biases.empty()) {
        LOG_INF("Expert bias  : %s\n", ca.expert_bias_spec.c_str());
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (!model || !ctx) { LOG_ERR("model/context init failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const int32_t eos_id = llama_vocab_eos(vocab);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    common_sampler * smpl = nullptr;

    if (params.n_predict > 0 && !ca.list_tensors) {
        smpl = common_sampler_init(model, params.sampling);
        if (!smpl) {
            LOG_ERR("failed to initialize sampler\n");
            return 1;
        }
        LOG_INF("Sampler      : %s\n", common_sampler_print(smpl).c_str());
    }

    LOG_INF("EOS token id : %d\n", eos_id);
    LOG_INF("EOG tokens in vocab:\n");
    for (int i = 0; i < n_vocab; i++) {
        if (llama_vocab_is_eog(vocab, i)) {
            char buf[256];
            int n = llama_token_to_piece(vocab, i, buf, sizeof(buf), 0, true);
            std::string tok_str(buf, n > 0 ? n : 0);
            LOG_INF("  token %d = '%s'%s\n", i, tok_str.c_str(),
                    i == eos_id ? " [PRIMARY EOS - will stop generation]" : " [EOG - will NOT stop generation]");
        }
    }

    LOG_INF("%s\n\n", common_params_get_system_info(params).c_str());

    std::vector<prompt_entry> prompts;
    if (!ca.prompt_file.empty()) {
        prompts = parse_prompt_file(ca.prompt_file);
        if (prompts.empty()) {
            LOG_ERR("No valid prompts loaded from --prompt-file %s\n", ca.prompt_file.c_str());
            return 1;
        }
    } else if (!params.prompt.empty()) {
        prompts.push_back({"single", params.prompt});
    } else {
        LOG_ERR("No prompts.  Use --prompt-file <tsv> or -p <text>\n");
        return 1;
    }
    LOG_INF("Loaded %zu prompts\n\n", prompts.size());

    int failed_prompts = 0;
    std::vector<prompt_result> results;
    results.reserve(prompts.size());

    std::set<std::string> used_prompt_dirs;
    for (size_t pi = 0; pi < prompts.size(); pi++) {
        const auto & pe = prompts[pi];
        const std::string safe_id_base = sanitize_prompt_id(pe.id, pi);
        std::string safe_id = safe_id_base;
        int suffix = 2;
        while (!used_prompt_dirs.insert(safe_id).second) {
            safe_id = safe_id_base + "_" + std::to_string(suffix++);
        }
        if (safe_id_base != pe.id || safe_id != safe_id_base) {
            LOG_INF("  sanitized prompt id '%s' -> '%s'\n", pe.id.c_str(), safe_id.c_str());
        }

        LOG_INF("[%zu/%zu] %s : %.70s%s\n",
                pi + 1, prompts.size(), pe.id.c_str(),
                pe.text.c_str(), pe.text.size() > 70 ? "..." : "");

        state.tensors.clear();
        state.discovered_tensor_names.clear();
        state.logged_shape_mismatch.clear();
        state.tensors_seen = 0;
        state.tensors_captured = 0;
        state.prompt_has_error = false;
        state.current_prompt_id = pe.id;
        state.output_dir = (fs::path(ca.output_dir) / safe_id).string();
        fs::create_directories(state.output_dir);

        llama_memory_clear(llama_get_memory(ctx), true);
        if (smpl) {
            common_sampler_reset(smpl);
        }

        std::vector<llama_token> tokens = common_tokenize(ctx, pe.text, add_bos);
        int n_prompt = (int) tokens.size();
        if (tokens.empty()) {
            LOG_ERR("  empty tokenization — skipping\n");
            fs::remove_all(state.output_dir);
            failed_prompts++;
            results.push_back({pe.id, safe_id, false, "empty_tokenization", 0, 0, 0.0});
            continue;
        }
        std::vector<prompt_token_info> prompt_tokens = build_prompt_token_info(vocab, tokens, pe.text);

        auto t0 = std::chrono::high_resolution_clock::now();

        const int prefill_batch = params.n_batch > 0 ? params.n_batch : 512;
        for (int pos = 0; pos < n_prompt; ) {
            int n_cur = std::min(prefill_batch, n_prompt - pos);
            if (llama_decode(ctx, llama_batch_get_one(tokens.data() + pos, n_cur))) {
                LOG_ERR("  prefill decode failed at token %d/%d — skipping\n", pos, n_prompt);
                state.prompt_has_error = true;
                break;
            }
            if (state.prompt_has_error) {
                break;
            }
            pos += n_cur;
        }

        if (state.prompt_has_error) {
            LOG_ERR("  prompt capture failed during prefill; removing partial output\n");
            auto t_err = std::chrono::high_resolution_clock::now();
            double ms_err = std::chrono::duration<double, std::milli>(t_err - t0).count();
            fs::remove_all(state.output_dir);
            failed_prompts++;
            results.push_back({pe.id, safe_id, false, "prefill_decode_failed", n_prompt, 0, ms_err});
            continue;
        }

        int n_gen = 0;
        std::string gen_text;
        std::vector<generated_token_info> generated_tokens;

        if (params.n_predict > 0 && !ca.list_tensors) {
            for (int g = 0; g < params.n_predict; g++) {
                llama_token next = common_sampler_sample(smpl, ctx, -1);
                if (next == eos_id) break;

                n_gen++;
                {
                    char buf[256];
                    int n = llama_token_to_piece(vocab, next, buf, sizeof(buf), 0, true);
                    std::string piece = n > 0 ? std::string(buf, n) : std::string();
                    generated_tokens.push_back({next, piece});
                    if (n > 0) {
                        gen_text.append(piece);
                        if (!ca.no_stream) {
                            fwrite(buf, 1, n, stdout);
                            fflush(stdout);
                        }
                    }
                }
                common_sampler_accept(smpl, next, true);

                if (llama_decode(ctx, llama_batch_get_one(&next, 1))) {
                    LOG_ERR("  gen decode failed at token %d\n", g);
                    state.prompt_has_error = true;
                    break;
                }
                if (state.prompt_has_error) {
                    LOG_ERR("  prompt capture failed during generation; aborting prompt\n");
                    break;
                }
            }
        }

        if (!ca.no_stream && n_gen > 0) {
            fprintf(stdout, "\n");
            fflush(stdout);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        LOG_INF("  tokens: %d prompt + %d gen | tensors: %d seen, %d captured, %zu unique | %.0f ms\n",
                n_prompt, n_gen,
                state.tensors_seen, state.tensors_captured, state.tensors.size(),
                ms);

        if (ca.list_tensors) {
            LOG_INF("\nDiscovered %zu unique tensor names.  Exiting.\n", state.discovered_tensor_names.size());
            // Discovery path: one prompt drove the forward pass, no tensors were stored, no npy
            // writes occurred. Record it as succeeded so the manifest is populated (the run_summary's
            // list_tensors_run: true flag disambiguates discovery from regular capture). This keeps
            // the "every prompt ID seen appears in the manifest" invariant and lets the exit-code
            // logic below return 0 instead of misclassifying discovery as "zero prompts processed".
            results.push_back({pe.id, safe_id, true, "", n_prompt, n_gen, ms});
            break;
        }

        if (state.prompt_has_error) {
            LOG_ERR("  prompt capture failed; removing partial output\n");
            fs::remove_all(state.output_dir);
            failed_prompts++;
            results.push_back({pe.id, safe_id, false, "gen_decode_failed", n_prompt, n_gen, ms});
            continue;
        }

        // Sanity check #1 — tensor count. We expect exactly 6 tensors per prompt in the default mode
        // (3 router + 3 residual), or 3 in --routing-only mode. If either hook didn't fire, fail loudly
        // instead of silently saving a partial capture — Step 1 of PLAN.md can't run on an incomplete tap.
        size_t expected = ca.routing_only ? 3u : (size_t) EXPECTED_TENSORS_PER_PROMPT;
        if (state.tensors.size() != expected) {
            LOG_ERR("  CAPTURE FAILED: expected %zu tensors, got %zu\n", expected, state.tensors.size());
            for (auto & [name, rec] : state.tensors) {
                LOG_ERR("    have: %s [%s, %lldx%lld]\n", name.c_str(),
                        rec.kind == TK_ROUTER ? "router" : "residual",
                        (long long) rec.n_tokens, (long long) rec.dim);
            }
            fs::remove_all(state.output_dir);
            failed_prompts++;
            results.push_back({pe.id, safe_id, false, "tensor_count_mismatch", n_prompt, n_gen, ms});
            continue;
        }

        // Sanity check #2 — row-count per tensor. Per CLAUDE.md "Row-count assertion": every
        // captured tensor must have n_tokens == n_prompt + n_gen. This catches L39-style off-by-one
        // capture bugs at capture time instead of at analysis time. Any row-count disagreement
        // fails the capture and removes partial output; the run proceeds to the next prompt.
        {
            const int64_t expected_rows = (int64_t) n_prompt + (int64_t) n_gen;
            bool row_mismatch = false;
            for (auto & [name, rec] : state.tensors) {
                if (rec.n_tokens != expected_rows) {
                    LOG_ERR("  CAPTURE FAILED: row-count mismatch for %s: got %lld, expected %lld (%d prompt + %d gen)\n",
                            name.c_str(),
                            (long long) rec.n_tokens,
                            (long long) expected_rows,
                            n_prompt, n_gen);
                    row_mismatch = true;
                }
            }
            if (row_mismatch) {
                fs::remove_all(state.output_dir);
                failed_prompts++;
                results.push_back({pe.id, safe_id, false, "row_count_mismatch", n_prompt, n_gen, ms});
                continue;
            }
        }

        if (!save_prompt(state, pe.text, gen_text, prompt_tokens, generated_tokens, n_prompt, n_gen, ms)) {
            LOG_ERR("  save_prompt failed; removing partial output\n");
            fs::remove_all(state.output_dir);
            failed_prompts++;
            results.push_back({pe.id, safe_id, false, "save_write_failed", n_prompt, n_gen, ms});
            continue;
        }
        results.push_back({pe.id, safe_id, true, "", n_prompt, n_gen, ms});
    }

    common_sampler_free(smpl);

    // Write the per-prompt manifest BEFORE exiting so wrappers inspecting a nonzero exit still
    // have a complete record of which prompts succeeded/failed and why.
    write_manifest(results, ca.output_dir, ca.list_tensors);

    LOG_INF("\n=== Capture Complete ===\n");
    LOG_INF("Prompts: %zu processed, %d failed  |  Output: %s\n",
            results.size(), failed_prompts, ca.output_dir.c_str());
    LOG_INF("Manifest: %s/capture_manifest.json\n", ca.output_dir.c_str());

    // Exit-code contract for wrapper scripts (strict: load-bearing capture for the verbalizer
    // pipeline, downstream steps must not treat a successful process exit as a complete capture set):
    //   0 — every processed prompt succeeded.
    //   1 — partial failure: at least one prompt succeeded AND at least one failed. Wrappers with
    //       `set -e` will abort. Wrappers that want to proceed on partial data MUST explicitly
    //       check rc == 1 and consume capture_manifest.json to learn which prompt IDs are present.
    //   2 — total failure: every processed prompt failed, OR zero prompts were processed.
    //       Nothing to analyze downstream.
    if (failed_prompts > 0) {
        LOG_ERR("WARNING: %d of %zu prompts failed; %zu succeeded. Consult %s/capture_manifest.json\n",
                failed_prompts, results.size(), results.size() - (size_t) failed_prompts,
                ca.output_dir.c_str());
    }

    llama_perf_context_print(ctx);
    llama_backend_free();

    if (results.empty())                                 return 2;  // no prompts processed
    if (failed_prompts == 0)                             return 0;  // all succeeded
    if ((size_t) failed_prompts >= results.size())       return 2;  // all failed
    return 1;                                                       // partial
}
