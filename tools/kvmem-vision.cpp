#include "llama-kvmem-diag.h"
#include "kvmem-vision.h"
#include "server-common.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <set>

kvmem_prompt::kvmem_prompt(const std::vector<llama_token> & input)
    : tokens(input), native_(std::make_shared<server_tokens>(input, true)) {}

kvmem_prompt::kvmem_prompt(std::shared_ptr<server_tokens> native) : native_(std::move(native)) {
    tokens.reserve(native_->size());
    for (size_t i = 0; i < native_->size(); ++i) tokens.push_back((*native_)[i]);
    for (const auto & range : media_ranges()) {
        position_offsets_[range.second] = native_->pos_next(range.second) - (llama_pos) range.second;
    }
}

bool kvmem_prompt::has_media() const {
    return std::find(tokens.begin(), tokens.end(), LLAMA_TOKEN_NULL) != tokens.end();
}

size_t kvmem_prompt::common_prefix(const kvmem_prompt & other) const {
    return native_->get_common_prefix(*other.native_);
}

llama_pos kvmem_prompt::model_pos(size_t row) const {
    const auto it = std::prev(position_offsets_.upper_bound(row));
    return (llama_pos) row + it->second;
}

const mtmd_input_chunk * kvmem_prompt::chunk(size_t row) const {
    return native_->find_chunk(row).get();
}

size_t kvmem_prompt::media_end(size_t row) const {
    return row + mtmd_input_chunk_get_n_tokens(chunk(row));
}

std::vector<std::pair<uint32_t, uint32_t>> kvmem_prompt::media_ranges() const {
    std::vector<std::pair<uint32_t, uint32_t>> result;
    for (size_t row = 0; row < tokens.size();) {
        if (tokens[row] != LLAMA_TOKEN_NULL) { ++row; continue; }
        const auto end = media_end(row);
        result.emplace_back(row, end);
        row = end;
    }
    return result;
}

std::shared_ptr<kvmem_prompt> kvmem_prompt::with_generated(const std::vector<llama_token> & gen) const {
    auto native = std::make_shared<server_tokens>(native_->clone());
    native->insert(gen);
    return std::make_shared<kvmem_prompt>(std::move(native));
}

common_chat_msg_spans kvmem_prompt::message_spans(const common_chat_msg_delimiters & delimiters) const {
    return native_->find_message_spans(delimiters);
}

std::vector<std::pair<uint32_t, std::string>> kvmem_prompt::media_identity() const {
    std::vector<std::pair<uint32_t, std::string>> ids;
    for (const auto & range : media_ranges()) ids.emplace_back(range.first, mtmd_input_chunk_get_id(chunk(range.first)));
    return ids;
}

std::shared_ptr<kvmem_prompt> kvmem_prompt::prefix(size_t rows) const {
    auto native = std::make_shared<server_tokens>(native_->clone());
    native->keep_first(rows);
    return std::make_shared<kvmem_prompt>(std::move(native));
}

std::shared_ptr<kvmem_prompt> kvmem_prompt::cache_index() const {
    auto native = std::make_shared<server_tokens>();
    native->has_mtmd = true;
    for (size_t row = 0; row < tokens.size();) {
        if (tokens[row] == LLAMA_TOKEN_NULL) {
            native->push_back_placeholder(chunk(row));
            row = media_end(row);
        } else {
            native->push_back(tokens[row++]);
        }
    }
    return std::make_shared<kvmem_prompt>(std::move(native));
}

size_t kvmem_prompt::index_bytes() const {
    // The private native token vector can retain growth capacity; bound it by
    // twice its length. Media entries here are placeholders in disk-cache mode.
    size_t bytes = sizeof(*this) + sizeof(server_tokens) + tokens.capacity()*sizeof(llama_token) +
        tokens.size()*sizeof(llama_token)*2 + position_offsets_.size()*64;
    for (const auto & range : media_ranges()) {
        size_t size = 0;
        if (mtmd_input_chunk_save(chunk(range.first), nullptr, 0, &size))
            throw std::runtime_error("cannot measure session media index");
        bytes += size + 256;
    }
    return bytes;
}

std::string kvmem_parse_media_messages(const std::string & body, bool allow_images, bool allow_video,
                                      std::vector<std::vector<uint8_t>> & files) {
    auto parsed = common_json::parse(body);
    server_chat_params params;
    params.allow_image = allow_images;
    params.allow_audio = false;
    params.allow_video = allow_video;
    oaicompat_chat_process_media(parsed, params, files);
    return parsed.dump();
}

kvmem_vision::kvmem_vision(llama_model * model, const std::string & path, bool gpu,
                           ggml_backend_dev_t device, int min_tokens, int max_tokens, int n_threads,
                           float video_fps) : video_fps_(video_fps) {
    auto params = mtmd_context_params_default();
    params.media_marker = get_media_marker();
    params.use_gpu = gpu;
    params.device = gpu ? device : nullptr;
    params.image_min_tokens = min_tokens;
    params.image_max_tokens = max_tokens;
    if (n_threads > 0) params.n_threads = n_threads;
    // Native lazy warmup uses the real image instead of a fixed 2116-token dummy.
    params.warmup = false;
    params.print_timings = true;
    ctx_ = mtmd_init_from_file(path.c_str(), model, params);
    if (!ctx_) throw std::runtime_error("failed to load mmproj: " + path);
    n_embd_ = llama_model_n_embd_inp(model);
    kvmem_diag("KVMEM_TRACE vision_load device=%s embedding_width=%d min_tokens=%d max_tokens=%d threads=%d\n",
            gpu ? (device ? ggml_backend_dev_name(device) : "GPU-auto") : "CPU",
            n_embd_, min_tokens, max_tokens, params.n_threads);
}

kvmem_vision::~kvmem_vision() { mtmd_free(ctx_); }

bool kvmem_vision::supports_video() const {
    return mtmd_helper_support_video(ctx_);
}

std::shared_ptr<kvmem_prompt> kvmem_vision::tokenize(const std::string & prompt,
                                                 const std::vector<std::vector<uint8_t>> & files) {
    std::lock_guard<std::mutex> lock(mu_);
    auto opt = mtmd_helper_init_opt_default();
    opt.video_params.fps_target = video_fps_;
    auto native = std::make_shared<server_tokens>(process_mtmd_prompt(ctx_, prompt, files, opt));
    return std::make_shared<kvmem_prompt>(std::move(native));
}

bool kvmem_prompt::media_ready(size_t begin) const {
    for (const auto & range : media_ranges()) {
        if (range.second > begin && !embeddings_.count(range.first)) return false;
    }
    return true;
}

void kvmem_vision::prepare(kvmem_prompt & prompt, size_t begin,
                           const std::function<bool()> & cancelled) {
    constexpr size_t limit = 128ull*1024*1024;
    std::unique_lock<std::timed_mutex> preparation(preparation_mu_, std::defer_lock);
    while (!preparation.try_lock_for(std::chrono::milliseconds(50))) {
        if (cancelled && cancelled()) throw std::runtime_error("media preparation cancelled");
    }
    std::unique_lock<std::mutex> lock(mu_);
    const auto ranges = prompt.media_ranges();
    size_t requested = 0;
    std::set<std::string> distinct;
    for (const auto & range : ranges) {
        if (range.second <= begin && !prompt.embeddings_.count(range.first)) continue;
        const auto * chunk = prompt.chunk(range.first);
        if (!distinct.insert(mtmd_input_chunk_get_id(chunk)).second) continue;
        const size_t tokens = mtmd_input_chunk_get_n_tokens(chunk);
        if (tokens > limit / (sizeof(float)*(size_t)n_embd_)) {
            throw std::invalid_argument("image embeddings exceed 128 MiB; reduce --image-max-tokens");
        }
        requested += tokens*(size_t)n_embd_*sizeof(float);
        if (requested > limit) throw std::invalid_argument("prepared media exceeds 128 MiB; reduce image count or tokens");
    }
    for (const auto & range : ranges) {
        if (range.second <= begin || prompt.embeddings_.count(range.first)) continue;
        if (cancelled && cancelled()) throw std::runtime_error("media preparation cancelled");
        const auto * chunk = prompt.chunk(range.first);
        const std::string id = mtmd_input_chunk_get_id(chunk);
        auto it = cache_.find(id);
        if (it == cache_.end()) {
            const size_t count = mtmd_input_chunk_get_n_tokens(chunk)*(size_t)n_embd_;
            const size_t bytes = count*sizeof(float);
            while (accounting_->bytes.load() + bytes > limit) {
                auto victim = cache_.end();
                for (auto candidate = cache_.begin(); candidate != cache_.end(); ++candidate) {
                    if (candidate->second.embd.use_count() == 1 &&
                            (victim == cache_.end() || candidate->second.used < victim->second.used)) victim = candidate;
                }
                if (victim != cache_.end()) {
                    cache_bytes_ -= victim->second.embd->size()*sizeof(float);
                    cache_.erase(victim);
                } else {
                    if (cancelled && cancelled()) throw std::runtime_error("media preparation cancelled");
                    accounting_->changed.wait_for(lock, std::chrono::milliseconds(50));
                }
            }
            const auto start = std::chrono::steady_clock::now();
            if (mtmd_encode_chunk(ctx_, chunk) != 0) throw std::runtime_error("vision encoder failed");
            ++prompt.encode_calls;
            prompt.encode_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            const float * data = mtmd_get_output_embd(ctx_);
            if (!data) throw std::runtime_error("vision encoder returned no embeddings");
            auto * values = new std::vector<float>(data, data + count);
            auto accounting = accounting_;
            accounting->bytes += bytes;
            std::shared_ptr<const std::vector<float>> embedding(values, [accounting, bytes](const std::vector<float> * value) {
                delete value;
                accounting->bytes -= bytes;
                accounting->changed.notify_all();
            });
            it = cache_.emplace(id, entry{std::move(embedding), 0}).first;
            cache_bytes_ += bytes;
        }
        it->second.used = ++clock_;
        prompt.embeddings_[range.first] = it->second.embd;
    }
    kvmem_diag("KVMEM_VISION_PREPARED encodes=%u encoder_ms=%.2f live_bytes=%zu\n",
                prompt.encode_calls, prompt.encode_ms, accounting_->bytes.load());
}

int kvmem_vision::decode(llama_context * ctx, const kvmem_prompt & prompt, size_t row, int n_batch,
                       const std::function<int(llama_batch)> & dispatch) {
    const auto * chunk = prompt.chunk(row);
    const auto ready = prompt.embeddings_.find(row);
    if (ready == prompt.embeddings_.end()) throw std::logic_error("media was not prepared before lane admission");
    struct callback_data {
        size_t row;
        const std::function<int(llama_batch)> * dispatch;
    } data {row, &dispatch};
    auto decode = [](llama_context *, llama_batch batch, void * opaque) -> int32_t {
        auto & data = *static_cast<callback_data *>(opaque);
        std::vector<llama_pos> logical(batch.n_tokens);
        for (int i = 0; i < batch.n_tokens; ++i) logical[i] = data.row + i;
        batch.logical_pos = logical.data();
        const int rc = (*data.dispatch)(batch);
        if (rc == 0) data.row += batch.n_tokens;
        return rc;
    };
    llama_pos next = prompt.model_pos(row);
    const int rc = mtmd_helper_decode_image_chunk_with_decoder(ctx_, ctx, chunk, const_cast<float *>(ready->second->data()),
            next, 0, n_batch, &next, nullptr, &data, decode);
    if (rc == 0 && next != prompt.model_pos(prompt.media_end(row))) throw std::runtime_error("inconsistent image position cursor");
    return rc;
}
