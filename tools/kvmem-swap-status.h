#pragma once

// Swap-status page: one pixel per host KV block, coloured by where the block
// currently lives. The model thread publishes a snapshot (throttled), HTTP
// threads only read it, so status polling never touches the model lock.

#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct kvmem_swap_block {
    uint32_t id = 0;
    uint32_t pos = 0;
    uint32_t n_tokens = 0;
    int32_t  tier = 0; // 0 GPU, 1 CPU, 2 SSD
    int32_t  gpu_slot = -1;
    int32_t  cpu_slot = -1;
    int32_t  nvme_slot = -1;
    bool     working = false;
    bool     in_flight = false;
    uint32_t remaps = 0;
};

struct kvmem_swap_snapshot {
    bool     valid = false; // KVMem memory is attached and was sampled at least once
    uint64_t serial = 0;
    std::chrono::steady_clock::time_point stamp {};
    uint32_t block_tokens = 0;
    uint32_t n_slots = 0;
    uint32_t free_slots = 0;
    uint32_t store_tokens = 0;
    uint32_t n_resident = 0;
    std::vector<kvmem_swap_block> blocks;
    // Logical token timeline the block positions index into. Immutable once
    // published, so readers detokenize without holding the model lock.
    std::shared_ptr<const std::vector<llama_token>> tokens;
};

class kvmem_swap_publisher {
public:
    static constexpr int64_t min_interval_ms = 200;

    // Model thread only. `force` bypasses the throttle. A null `tokens` keeps
    // the previous detokenization source.
    bool publish(std::vector<kvmem_swap_block> blocks, uint32_t block_tokens, uint32_t n_slots,
                 uint32_t free_slots, uint32_t store_tokens, uint32_t n_resident,
                 std::shared_ptr<const std::vector<llama_token>> tokens, bool force) {
        const auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(mu_);
        if (!force && data_.valid &&
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_).count() < min_interval_ms) {
            return false;
        }
        data_.valid = true;
        data_.serial++;
        data_.stamp = now;
        data_.block_tokens = block_tokens;
        data_.n_slots = n_slots;
        data_.free_slots = free_slots;
        data_.store_tokens = store_tokens;
        data_.n_resident = n_resident;
        data_.blocks = std::move(blocks);
        if (tokens) data_.tokens = std::move(tokens);
        last_ = now;
        return true;
    }

    kvmem_swap_snapshot get() const {
        std::lock_guard<std::mutex> lock(mu_);
        kvmem_swap_snapshot out;
        out.valid = data_.valid;
        out.serial = data_.serial;
        out.stamp = data_.stamp;
        out.block_tokens = data_.block_tokens;
        out.n_slots = data_.n_slots;
        out.free_slots = data_.free_slots;
        out.store_tokens = data_.store_tokens;
        out.n_resident = data_.n_resident;
        out.blocks = data_.blocks;
        out.tokens = data_.tokens;
        return out;
    }

private:
    mutable std::mutex mu_;
    kvmem_swap_snapshot data_;
    std::chrono::steady_clock::time_point last_ {};
};

inline std::string kvmem_swap_token_piece(const llama_vocab * vocab, llama_token id) {
    if (!vocab) return {};
    char buf[256];
    const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
    if (n <= 0) return {};
    return std::string(buf, (size_t) n);
}

// Compact row form: the page polls once per second, so 2048 blocks must not
// cost a few hundred KB of JSON. `fields` in the status document names the
// columns.
inline const std::vector<const char *> & kvmem_swap_block_fields() {
    static const std::vector<const char *> fields = {
        "id", "pos", "n", "tier", "gpu", "cpu", "nvme", "ws", "fly", "remap",
    };
    return fields;
}

inline nlohmann::json kvmem_swap_status_json(const kvmem_swap_snapshot & snap, double age_ms, int interval_ms) {
    nlohmann::json out;
    // A sampled snapshot with no GPU pool means the model runs without KVMem.
    out["enabled"] = snap.valid && snap.n_slots > 0;
    out["serial"] = snap.serial;
    out["age_ms"] = age_ms;
    out["interval_ms"] = interval_ms;
    out["block_tokens"] = snap.block_tokens;
    out["n_slots"] = snap.n_slots;
    out["free_slots"] = snap.free_slots;
    out["store_tokens"] = snap.store_tokens;
    out["n_blocks"] = snap.blocks.size();
    out["n_resident"] = snap.n_resident;
    out["has_tokens"] = snap.tokens != nullptr;
    out["fields"] = nlohmann::json::array();
    for (const char * name : kvmem_swap_block_fields()) out["fields"].push_back(name);
    out["blocks"] = nlohmann::json::array();
    for (const auto & b : snap.blocks) {
        out["blocks"].push_back(nlohmann::json::array({b.id, b.pos, b.n_tokens, b.tier, b.gpu_slot, b.cpu_slot,
                                                       b.nvme_slot, (int) b.working, (int) b.in_flight, b.remaps}));
    }
    return out;
}

inline nlohmann::json kvmem_swap_block_json(const kvmem_swap_snapshot & snap, const llama_vocab * vocab, int64_t id) {
    if (!snap.valid || id < 0 || (size_t) id >= snap.blocks.size()) return nlohmann::json::object();
    const auto & b = snap.blocks[(size_t) id];
    nlohmann::json out = {
        {"id", b.id}, {"pos", b.pos}, {"n", b.n_tokens}, {"tier", b.tier},
        {"gpu", b.gpu_slot}, {"cpu", b.cpu_slot}, {"nvme", b.nvme_slot},
        {"ws", b.working}, {"fly", b.in_flight}, {"remap", b.remaps},
        {"text_available", false},
    };
    if (!snap.tokens || b.n_tokens == 0) return out;
    const auto & tokens = *snap.tokens;
    if ((size_t) b.pos >= tokens.size()) return out;
    const size_t begin = b.pos;
    const size_t end = std::min<size_t>(tokens.size(), (size_t) b.pos + b.n_tokens);
    constexpr size_t text_limit = 4096;
    std::string text;
    nlohmann::json ids = nlohmann::json::array();
    for (size_t i = begin; i < end; ++i) {
        ids.push_back((int32_t) tokens[i]);
        text += kvmem_swap_token_piece(vocab, tokens[i]);
        if (text.size() >= text_limit) break;
    }
    out["token_begin"] = begin;
    out["token_end"] = end;
    out["token_ids"] = std::move(ids);
    out["text"] = text;
    out["text_truncated"] = text.size() >= text_limit;
    out["text_available"] = true;
    return out;
}

// Self-contained page: no build step and no dependency on the bundled chat UI,
// so it also works with --no-ui.
inline const char * kvmem_swap_page_html() {
    return R"kvmemswap(<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KVMem swap status</title>
<style>
:root { --bg:#0b0f14; --panel:#141b23; --line:#243040; --fg:#dbe4ee; --dim:#8fa2b7; }
* { box-sizing:border-box; }
html, body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.45 ui-monospace,Consolas,monospace; }
header { display:flex; align-items:center; gap:12px; flex-wrap:wrap; padding:10px 14px; background:var(--panel); border-bottom:1px solid var(--line); }
h1 { margin:0; font-size:14px; font-weight:600; }
a, button { color:var(--fg); background:#1d2836; border:1px solid var(--line); border-radius:4px; padding:3px 9px; font:inherit; text-decoration:none; cursor:pointer; }
a:hover, button:hover { background:#25344a; }
#stats { color:var(--dim); }
#stats b { color:var(--fg); font-weight:600; }
#banner { display:none; margin:10px 14px 0; padding:8px 10px; border-radius:4px; background:#3a2416; border:1px solid #6b4423; color:#ffd9b0; }
#legend { display:flex; gap:14px; flex-wrap:wrap; padding:10px 14px; color:var(--dim); }
.sw { display:inline-block; width:10px; height:10px; margin-right:5px; border-radius:2px; vertical-align:-1px; }
#stage { padding:0 14px; }
#grid { display:block; cursor:crosshair; }
#detail { margin:12px 14px 24px; padding:10px 12px; background:var(--panel); border:1px solid var(--line); border-radius:6px; }
#detail h2 { margin:0 0 6px; font-size:13px; }
#body { color:var(--dim); }
#body b { color:var(--fg); font-weight:600; }
#body .kv { margin-right:14px; white-space:nowrap; }
#text { display:none; margin-top:8px; padding:8px; max-height:240px; overflow:auto; white-space:pre-wrap; word-break:break-all; background:#05080c; border:1px solid var(--line); border-radius:4px; color:var(--fg); }
</style>
</head>
<body>
<header>
  <h1>KVMem swap status</h1>
  <span id="stats">loading...</span>
  <button id="pause" type="button">pause</button>
  <a href="/">chat UI</a>
</header>
<div id="banner"></div>
<div id="legend">
  <span><i class="sw" style="background:#22c55e"></i>GPU resident</span>
  <span><i class="sw" style="background:#ef4444"></i>swapped out to host RAM</span>
  <span><i class="sw" style="background:#a855f7"></i>NVMe tier</span>
  <span><i class="sw" style="background:#eab308"></i>copy in flight</span>
  <span>one pixel = one KV block; hover to highlight; click for the block detail</span>
</div>
<div id="stage"><canvas id="grid"></canvas></div>
<div id="detail"><h2 id="dhead">block detail</h2><div id="body">click a pixel.</div><pre id="text"></pre></div>
<script>
(function () {
  'use strict';
  var qs = new URLSearchParams(location.search);
  var key = qs.get('key') || '';
  var interval = parseInt(qs.get('interval') || '1000', 10);
  if (!isFinite(interval) || interval < 250) interval = 1000;
  if (interval > 60000) interval = 60000;

  var STATUS = '/kvmem/swap/status';
  var BLOCK = '/kvmem/swap/block';
  var CELL = 12, GAP = 2, STRIDE = CELL + GAP;
  var HIGHLIGHT = '#7dd3fc';
  var COLORS = { gpu: '#22c55e', cpu: '#ef4444', ssd: '#a855f7', fly: '#eab308' };
  var TIERS = ['GPU', 'host RAM', 'NVMe'];

  var stage = document.getElementById('stage');
  var grid = document.getElementById('grid');
  var statsEl = document.getElementById('stats');
  var bannerEl = document.getElementById('banner');
  var headEl = document.getElementById('dhead');
  var bodyEl = document.getElementById('body');
  var textEl = document.getElementById('text');
  var pauseBtn = document.getElementById('pause');

  var state = { blocks: [], enabled: false, serial: 0, n_slots: 0, free_slots: 0,
                store_tokens: 0, n_resident: 0 };
  var F = { id: 0, pos: 1, n: 2, tier: 3, gpu: 4, cpu: 5, nvme: 6, ws: 7, fly: 8, remap: 9 };
  var cols = 1, hover = -1, selected = -1, paused = false, inflight = false;

  function apiHeaders() {
    var h = { 'Cache-Control': 'no-cache' };
    if (key) h['Authorization'] = 'Bearer ' + key;
    return h;
  }
  function banner(msg) { bannerEl.style.display = 'block'; bannerEl.textContent = msg; }
  function clearBanner() { bannerEl.style.display = 'none'; }

  // green GPU resident, red swapped out to the host side, purple NVMe tier,
  // amber while an async copy owns the block.
  function colorOf(b) {
    if (b[F.fly]) return COLORS.fly;
    if (b[F.tier] === 0) return COLORS.gpu;
    if (b[F.tier] === 2) return COLORS.ssd;
    return COLORS.cpu;
  }

  function layout() {
    var cssW = Math.max(CELL, (stage.clientWidth || window.innerWidth) - 28);
    cols = Math.max(1, Math.floor(cssW / STRIDE));
    var rows = Math.max(1, Math.ceil(Math.max(1, state.blocks.length) / cols));
    var dpr = window.devicePixelRatio || 1;
    grid.style.width = (cols * STRIDE) + 'px';
    grid.style.height = (rows * STRIDE) + 'px';
    grid.width = Math.round(cols * STRIDE * dpr);
    grid.height = Math.round(rows * STRIDE * dpr);
    var ctx = grid.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    grid._ctx = ctx;
  }

  // Repaint one cell. Hover only repaints the two affected cells, so a large
  // pool does not cost a full-grid redraw per mouse move.
  function paintCell(i, outline) {
    var ctx = grid._ctx;
    if (!ctx || i < 0 || i >= state.blocks.length) return;
    var b = state.blocks[i];
    var x = (i % cols) * STRIDE, y = Math.floor(i / cols) * STRIDE;
    ctx.clearRect(x - 2, y - 2, CELL + 4, CELL + 4);
    ctx.fillStyle = colorOf(b);
    ctx.fillRect(x, y, CELL, CELL);
    var ring = outline || (i === selected ? '#ffffff' : (i === hover ? HIGHLIGHT : null));
    if (ring) {
      ctx.strokeStyle = ring;
      ctx.lineWidth = 2;
      ctx.strokeRect(x - 1, y - 1, CELL + 2, CELL + 2);
    }
  }

  function draw() {
    var ctx = grid._ctx;
    if (!ctx) return;
    ctx.clearRect(0, 0, grid.width, grid.height);
    for (var i = 0; i < state.blocks.length; i++) paintCell(i, null);
  }

  function hit(ev) {
    var rect = grid.getBoundingClientRect();
    if (!rect.width) return -1;
    var c = Math.floor((ev.clientX - rect.left) / STRIDE);
    var r = Math.floor((ev.clientY - rect.top) / STRIDE);
    if (c < 0 || c >= cols || r < 0) return -1;
    var i = r * cols + c;
    return i < state.blocks.length ? i : -1;
  }

  function renderStats() {
    var hoverTxt = '';
    if (hover >= 0 && hover < state.blocks.length) {
      var b = state.blocks[hover];
      hoverTxt = ' | hover #' + b[F.id] + ' ' + (TIERS[b[F.tier]] || b[F.tier]) +
                 ' pos=' + b[F.pos] + ' n=' + b[F.n] + (b[F.fly] ? ' in-flight' : '');
    }
    statsEl.textContent = 'serial ' + state.serial + ' | blocks ' + state.blocks.length +
      ' | GPU ' + state.n_resident + ' | free slots ' + state.free_slots + '/' + state.n_slots +
      ' | stored ' + state.store_tokens + ' tok' + hoverTxt;
  }

  function kv(label, value) {
    var span = document.createElement('span');
    span.className = 'kv';
    span.appendChild(document.createTextNode(label + ' '));
    var b = document.createElement('b');
    b.textContent = String(value);
    span.appendChild(b);
    return span;
  }

  function renderDetail(d) {
    headEl.textContent = 'block #' + d.id;
    bodyEl.textContent = '';
    textEl.style.display = 'none';
    textEl.textContent = '';
    bodyEl.appendChild(kv('tier', TIERS[d.tier] || d.tier));
    bodyEl.appendChild(kv('orig pos', d.pos + '..' + (d.pos + d.n - 1)));
    bodyEl.appendChild(kv('tokens', d.n));
    bodyEl.appendChild(kv('gpu slot', d.gpu));
    bodyEl.appendChild(kv('cpu slot', d.cpu));
    bodyEl.appendChild(kv('nvme slot', d.nvme));
    bodyEl.appendChild(kv('working set', d.ws ? 'yes' : 'no'));
    bodyEl.appendChild(kv('in flight', d.fly ? 'yes' : 'no'));
    bodyEl.appendChild(kv('remaps', d.remap));
    if (!d.text_available) {
      bodyEl.appendChild(kv('token text', 'none yet, run a request first'));
      return;
    }
    if (d.token_ids && d.token_ids.length) {
      var ids = d.token_ids.slice(0, 32).join(' ');
      if (d.token_ids.length > 32) ids += ' ...';
      bodyEl.appendChild(kv('ids', ids));
    }
    textEl.textContent = d.text || '';
    if (d.text_truncated) textEl.textContent += '\n[truncated]';
    textEl.style.display = 'block';
  }

  function showDetail(idx) {
    selected = idx;
    draw();
    headEl.textContent = 'block #' + idx;
    bodyEl.textContent = 'loading...';
    fetch(BLOCK + '?id=' + idx, { headers: apiHeaders(), cache: 'no-store' })
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(renderDetail)
      .catch(function (e) { bodyEl.textContent = 'detail failed: ' + e.message; });
  }

  // The status document names its columns, so keep the index map in sync.
  function remapFields(fields) {
    var names = ['id', 'pos', 'n', 'tier', 'gpu', 'cpu', 'nvme', 'ws', 'fly', 'remap'];
    for (var i = 0; i < names.length; i++) {
      var at = fields.indexOf(names[i]);
      if (at >= 0) F[names[i]] = at;
    }
  }

  function refresh() {
    if (inflight || paused) return;
    inflight = true;
    fetch(STATUS + '?interval=' + interval, { headers: apiHeaders(), cache: 'no-store' })
      .then(function (r) {
        if (r.status === 401) throw new Error('401 unauthorized: append ?key=YOUR_API_KEY to the page URL');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (j) {
        state.blocks = j.blocks || [];
        state.enabled = !!j.enabled;
        state.serial = j.serial || 0;
        state.n_slots = j.n_slots || 0;
        state.free_slots = j.free_slots || 0;
        state.store_tokens = j.store_tokens || 0;
        state.n_resident = j.n_resident || 0;
        if (j.fields && j.fields.length) remapFields(j.fields);
        if (!state.enabled) {
          banner('KVMem is not active on this server; start llama-kvmem-server with --kvmem.');
        } else if (!state.blocks.length) {
          banner('KVMem is active but has no blocks yet; send a chat request.');
        } else {
          clearBanner();
        }
        layout();
        draw();
        renderStats();
      })
      .catch(function (e) { banner(e.message); })
      .then(function () { inflight = false; });
  }

  grid.addEventListener('mousemove', function (ev) {
    var idx = hit(ev);
    if (idx === hover) return;
    var prev = hover;
    hover = idx;
    paintCell(prev, null);
    paintCell(hover, HIGHLIGHT);
    renderStats();
  });
  grid.addEventListener('mouseleave', function () {
    if (hover < 0) return;
    var prev = hover;
    hover = -1;
    paintCell(prev, null);
    renderStats();
  });
  grid.addEventListener('click', function (ev) {
    var idx = hit(ev);
    if (idx >= 0) showDetail(idx);
  });
  window.addEventListener('resize', function () { layout(); draw(); });
  pauseBtn.addEventListener('click', function () {
    paused = !paused;
    pauseBtn.textContent = paused ? 'resume' : 'pause';
  });

  layout();
  refresh();
  setInterval(refresh, interval);
})();
</script>
</body>
</html>
)kvmemswap";
}

