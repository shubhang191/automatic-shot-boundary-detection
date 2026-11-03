// sbdv.hpp - SBD-Verifier core, version 2.  C++17, header only, no third-party libraries.
//
// What this is: a small, dependency-free cut verifier.  A shot-boundary detector (any detector) proposes cuts; the verifier looks at the
// tiny 48x27 pictures around every proposed cut and says KEEP or DROP.  It can only remove cuts (or, for the OmniShotCut model with
// "rescue", add big picture jumps); it can never find a cut the detector did not propose.
//
// Two ways to run the SAME code:
//   * RAM mode     : all frames are in memory (uint8 [T][27][48][3]).  Needed by models that use whole-video numbers (the old v1-v3 /
//                    omni models).  run_pipeline().
//   * stream mode  : frames arrive one by one (e.g. straight from ffmpeg) and are kept in a small ring buffer, so memory does NOT grow
//                    with the length of the video.  Works for every model that only uses "local" numbers (model.streamable()), which
//                    includes the generic picture-only model.  class Streamer.
// Both modes call the same per-candidate feature code, so they give bit-identical features and decisions (tests/run_tests.py checks it).
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace sbdv {

#define SBDV_VERSION "2.0"
constexpr int FH = 27, FW = 48, FC = 3, FPIX = FH * FW, FSZ = FPIX * FC;   // one frame = 27 x 48 x 3 = 3888 bytes
constexpr int TW = 64;                                                       // TCN window (frames around a candidate)
constexpr int TCH = 6;                                                       // TCN input curves
constexpr int PAD_B = 61, PAD_F = 60;                                        // frames the local features read before a / after b

// ------------------------------------------------------------------------------------------------ feature names
// Models refer to features BY NAME, so the order here is free.  The last five are new in version 2 and only use frames near the candidate
// (so they work in stream mode): n60, loc_motion, d_adj_loc, gprev, gnext.
#define SBDV_FEATS(X)                                                                                                         \
  X(d_adj) X(d_ctx) X(spike) X(persist) X(hist_d) X(bright) X(ld) X(lf) X(p_at) X(p_near) X(gap) X(is_low) X(gap_prev)         \
  X(gap_next) X(density)                                                                                                       \
  X(mc_d) X(mc_ratio) X(edge_d) X(edge_rel) X(frac05) X(frac10) X(frac20) X(ctr_d) X(brd_d) X(d_wide) X(wide_ratio)           \
  X(pre_stab) X(post_stab) X(cut_vs_stab) X(n15) X(dup_dist) X(dup_p) X(nb_p_prev) X(nb_p_next) X(p_rank)                     \
  X(vid_motion) X(vid_motion_p90) X(vid_bright) X(vid_contrast) X(vid_len) X(vid_low_share) X(d_adj_rel) X(p_vs_vid)         \
  X(lab_hard_cut) X(lab_sudden_jump) X(lab_transition) X(lab_new_start) X(n_dropped) X(is_rescue)                              \
  X(as_p_at) X(as_p_near) X(as_agrees)                                                                                         \
  X(n60) X(loc_motion) X(d_adj_loc) X(gprev) X(gnext)
enum FeatId : int {
#define X(n) F_##n,
  SBDV_FEATS(X)
#undef X
      NF
};
static const char* const FEAT_NAME[NF] = {
#define X(n) #n,
    SBDV_FEATS(X)
#undef X
};
inline int feat_index(const std::string& s) {
  for (int i = 0; i < NF; i++) if (s == FEAT_NAME[i]) return i;
  return -1;
}

// ------------------------------------------------------------------------------------------------ small tables
struct Tables {
  float lut[256];       // u / 255 in float32, exactly like numpy's  frames.astype(float32) / 255.0
  uint8_t bin8[256];    // histogram bin of a pixel value (8 bins on [0,1], last bin closed)
  Tables() {
    for (int i = 0; i < 256; i++) { lut[i] = (float)i / 255.0f; int b = (8 * i) / 255; bin8[i] = (uint8_t)(b > 7 ? 7 : b); }
  }
};
inline const Tables& TB() { static Tables t; return t; }

inline double clipd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline int clampi(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
inline double rf(double x) { return (double)(float)x; }     // round to float32 like numpy does

inline double now_us() {
  using namespace std::chrono;
  return (double)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

// ------------------------------------------------------------------------------------------------ candidates
enum Src : int { SRC_BASE = 0, SRC_LOW = 1, SRC_DET = 2, SRC_RESCUE = 3, SRC_USER = 4 };
struct Cand {
  int a = 0, b = 0;            // a = last frame of the old shot, b = first frame of the new shot
  int src = SRC_USER;
  float p_at = 1.f, p_near = 1.f;
  int is_low = 0, is_rescue = 0, n_dropped = 0;
  int lab[4] = {0, 0, 0, 0};   // hard_cut, sudden_jump, transition, new_start  (OmniShotCut labels; zero for other detectors)
  int valid = 1;               // 0 = refers to frames outside the video (set when the video length is known)
};
typedef std::array<int, 2> Pair;

// ------------------------------------------------------------------------------------------------ model
struct Trees { std::vector<int32_t> feat, left, right, root; std::vector<double> thr, val; };
enum MType : int { M_LOGREG = 1, M_BOOST = 2, M_BAG = 3, M_TCN = 4 };
struct Member {
  int type = 0;
  std::vector<int> cols;                                 // canonical feature ids this member reads
  bool has_scale = false; std::vector<double> mean, scale, w; double b = 0;           // logistic regression
  double bias = 0; Trees tr;                                                          // boosted trees (sum + sigmoid) / bagged trees (mean)
  int n_ch = TCH, n_tab = 0;                                                          // TCN
  std::vector<float> mu, sd, c0w, c0b, c1w, c1b, c2w, c2b, tabw, tabb, h0w, h0b, h1w, h1b;
};

struct Need {
  bool f[NF] = {};
  bool pix = false, look = false, pic2 = false, crowd = false, loc = false, vid = false, win = false, sig = false, needT = false;
  void derive() {
    auto any = [&](std::initializer_list<int> l) { for (int i : l) if (f[i]) return true; return false; };
    vid = any({F_vid_motion, F_vid_motion_p90, F_vid_bright, F_vid_contrast, F_vid_len, F_vid_low_share, F_d_adj_rel, F_p_vs_vid});
    pix = vid || any({F_d_adj, F_d_ctx, F_spike, F_persist, F_hist_d, F_bright, F_d_adj_loc});
    look = any({F_ld, F_lf});
    pic2 = any({F_mc_d, F_mc_ratio, F_edge_d, F_edge_rel, F_frac05, F_frac10, F_frac20, F_ctr_d, F_brd_d, F_d_wide, F_wide_ratio,
                F_pre_stab, F_post_stab, F_cut_vs_stab});
    crowd = any({F_n15, F_dup_dist, F_dup_p, F_nb_p_prev, F_nb_p_next, F_p_rank});
    loc = any({F_loc_motion, F_d_adj_loc});
    sig = vid || win;
    needT = any({F_density, F_vid_len, F_gap_next});      // numbers that depend on the total length of the video
  }
};

struct Model {
  std::string name;
  int input_kind = 0;                 // 0 = per-frame probabilities, 1 = list of cuts / shots
  int rescue = 0;                     // 1 = also propose extra candidates at big picture jumps (the detector may have missed them)
  double tau = 0.5, tau_rescue = 0.5, base_thr = 0.296, low_thr = 0.10, r_ratio = 4.0, r_min = 0.05;
  int dedupe = 0, snap_w = 0;
  std::vector<Member> members;
  Need need;
  size_t file_bytes = 0;

  // true when every number the model reads comes from frames near the candidate: it can run frame by frame with constant memory
  bool streamable() const { return !need.sig && !need.needT && !rescue && snap_w == 0; }

  // ---- loading from the .sbdm file written by sbdv_export.py
  struct Rd {
    const uint8_t* p; size_t n, i = 0; bool bad = false;
    template <class T> T get() { T v{}; if (i + sizeof(T) > n) { bad = true; return v; } std::memcpy(&v, p + i, sizeof(T)); i += sizeof(T); return v; }
    std::string str() { uint32_t L = get<uint32_t>(); if (bad || i + L > n) { bad = true; return ""; } std::string s((const char*)p + i, L); i += L; return s; }
    template <class T> void vec(std::vector<T>& v, size_t cnt) {
      if (bad || cnt > (n - i) / sizeof(T)) { bad = true; return; }
      v.resize(cnt); if (cnt) std::memcpy(v.data(), p + i, cnt * sizeof(T)); i += cnt * sizeof(T);
    }
  };
  bool load_mem(const uint8_t* buf, size_t n, std::string& err) {
    Rd r{buf, n};
    char magic[8]; for (int k = 0; k < 8; k++) magic[k] = (char)r.get<uint8_t>();
    if (r.bad || std::memcmp(magic, "SBDVMDL\x01", 8) != 0) { err = "not an sbdv model file (bad magic)"; return false; }
    uint32_t ver = r.get<uint32_t>(); if (ver != 1) { err = "unsupported model version"; return false; }
    input_kind = (int)r.get<uint32_t>(); uint32_t flags = r.get<uint32_t>(); rescue = (flags & 1) ? 1 : 0;
    tau = r.get<double>(); tau_rescue = r.get<double>(); base_thr = r.get<double>(); low_thr = r.get<double>();
    r_ratio = r.get<double>(); r_min = r.get<double>(); dedupe = r.get<int32_t>(); snap_w = r.get<int32_t>();
    name = r.str(); uint32_t nm = r.get<uint32_t>();
    if (r.bad || nm == 0 || nm > 4096) { err = "corrupt header"; return false; }
    members.clear(); members.resize(nm);
    for (auto& m : members) {
      m.type = (int)r.get<uint32_t>(); uint32_t nc = r.get<uint32_t>();
      if (r.bad || nc > 256) { err = "corrupt member"; return false; }
      for (uint32_t k = 0; k < nc; k++) {
        std::string s = r.str(); int id = feat_index(s);
        if (id < 0) { err = "unknown feature name in model: " + s + " (model made by a newer sbdv?)"; return false; }
        m.cols.push_back(id);
      }
      if (m.type == M_LOGREG) {
        m.has_scale = r.get<uint32_t>() != 0;
        if (m.has_scale) { r.vec(m.mean, nc); r.vec(m.scale, nc); }
        r.vec(m.w, nc); m.b = r.get<double>();
      } else if (m.type == M_BOOST || m.type == M_BAG) {
        m.bias = r.get<double>(); uint32_t nt = r.get<uint32_t>(), nn = r.get<uint32_t>();
        r.vec(m.tr.feat, nn); r.vec(m.tr.thr, nn); r.vec(m.tr.left, nn); r.vec(m.tr.right, nn); r.vec(m.tr.val, nn); r.vec(m.tr.root, nt);
        if (!r.bad) {
          for (uint32_t k = 0; k < nn; k++) {
            if (m.tr.feat[k] >= (int)nc || m.tr.feat[k] < -1) { err = "bad tree feature index"; return false; }
            if (m.tr.feat[k] >= 0 && (m.tr.left[k] < 0 || m.tr.left[k] >= (int)nn || m.tr.right[k] < 0 || m.tr.right[k] >= (int)nn)) { err = "bad tree child index"; return false; }
          }
          for (uint32_t k = 0; k < nt; k++) if (m.tr.root[k] < 0 || m.tr.root[k] >= (int)nn) { err = "bad tree root"; return false; }
          if (nt == 0) { err = "model has no trees"; return false; }
        }
      } else if (m.type == M_TCN) {
        m.n_ch = (int)r.get<uint32_t>(); m.n_tab = (int)r.get<uint32_t>();
        if (m.n_ch != TCH || m.n_tab != (int)nc) { err = "TCN shape mismatch"; return false; }
        r.vec(m.mu, m.n_tab); r.vec(m.sd, m.n_tab);
        r.vec(m.c0w, 24 * TCH * 5); r.vec(m.c0b, 24); r.vec(m.c1w, 24 * 24 * 5); r.vec(m.c1b, 24); r.vec(m.c2w, 32 * 24 * 3); r.vec(m.c2b, 32);
        r.vec(m.tabw, (size_t)32 * m.n_tab); r.vec(m.tabb, m.n_tab ? 32 : 0);
        size_t hin = 64 + (m.n_tab ? 32 : 0);
        r.vec(m.h0w, 32 * hin); r.vec(m.h0b, 32); r.vec(m.h1w, 32); r.vec(m.h1b, 1);
        m.has_scale = false;
        if (m.n_tab && m.sd.size() != (size_t)m.n_tab) { err = "TCN normaliser missing"; return false; }
      } else { err = "unknown member type"; return false; }
      if (r.bad) { err = "model file truncated"; return false; }
    }
    file_bytes = n;
    need = Need();
    for (auto& m : members) { if (m.type == M_TCN) need.win = true; for (int c : m.cols) need.f[c] = true; }
    need.derive();
    return true;
  }
  bool load(const std::string& path, std::string& err) {
    // Integration portability patch: UTF-8 model paths on Windows, unchanged model math.
    std::ifstream f(std::filesystem::u8path(path), std::ios::binary);
    if (!f) { err = "cannot open model file: " + path; return false; }
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return load_mem(buf.data(), buf.size(), err);
  }
  // size of the model in numbers (what a "parameter count" means here: split thresholds + leaf values, weights, biases)
  size_t n_params() const {
    size_t s = 0;
    for (auto& m : members) {
      if (m.type == M_LOGREG) s += m.w.size() + 1;
      else if (m.type == M_BOOST || m.type == M_BAG) s += m.tr.val.size() + 1;
      else if (m.type == M_TCN) s += m.c0w.size() + m.c0b.size() + m.c1w.size() + m.c1b.size() + m.c2w.size() + m.c2b.size() + m.tabw.size() + m.tabb.size() + m.h0w.size() + m.h0b.size() + m.h1w.size() + m.h1b.size();
    }
    return s;
  }
  size_t n_trees() const { size_t s = 0; for (auto& m : members) if (m.type == M_BOOST || m.type == M_BAG) s += m.tr.root.size(); return s; }
  size_t n_nodes() const { size_t s = 0; for (auto& m : members) if (m.type == M_BOOST || m.type == M_BAG) s += m.tr.val.size(); return s; }
  size_t n_tcn() const { size_t s = 0; for (auto& m : members) if (m.type == M_TCN) s++; return s; }
};

// ------------------------------------------------------------------------------------------------ one video
// Frames live in a "store" of `cap` slots: frame t sits in slot t % cap.  RAM mode: cap = T, so slot = t.  Stream mode: cap is small and
// the oldest frame is overwritten.  T is the number of frames seen so far (the final length once the video has ended).
struct Video {
  const uint8_t* d = nullptr; long cap = 1; int T = 0;
  std::vector<float> p;                 // detector probabilities, padded / cut to T (kind 0 only)
  std::vector<float> d1, d2, b1, rel;   // per-frame signals of the WHOLE video (only for models that need them; RAM mode)
  bool sig = false;
  std::vector<float> d1c; std::vector<int> d1tag;      // lazily cached frame-to-frame change, indexed like the store
  double vm = 0, vp90 = 0, vbright = 0, vcontrast = 0;
  const uint8_t* at(int t) const {
#ifdef SBDV_DEBUG_RING
    if (t < 0 || t >= T || (long)t <= (long)T - 1 - cap) { std::fprintf(stderr, "ring violation: frame %d requested, have [%ld, %d]\n", t, std::max(0L, (long)T - cap), T - 1); std::abort(); }
#endif
    return d + (size_t)((long)t % cap) * FSZ;
  }

  static double mad(const uint8_t* A, const uint8_t* B) {                  // mean |A-B| on the 0..1 scale (exact integer sum of differences)
    uint32_t s = 0;
    for (int j = 0; j < FSZ; j++) s += (uint32_t)std::abs((int)A[j] - (int)B[j]);
    return (double)s / (255.0 * FSZ);
  }
  float get_d1(int t) {
    if (t <= 0) return 0.f;
    if (sig) return d1[t];
    if (d1c.empty()) { d1c.assign((size_t)cap, 0.f); d1tag.assign((size_t)cap, -1); }
    size_t s = (size_t)((long)t % cap);
    if (d1tag[s] != t) { d1c[s] = (float)mad(at(t), at(t - 1)); d1tag[s] = t; }
    return d1c[s];
  }
  // mean of frames [t0, t1) on the 0..1 scale
  void avgf(int t0, int t1, float* out) const {
    uint16_t acc[FSZ]; const uint8_t* f = at(t0);
    for (int i = 0; i < FSZ; i++) acc[i] = f[i];
    for (int t = t0 + 1; t < t1; t++) { f = at(t); for (int i = 0; i < FSZ; i++) acc[i] = (uint16_t)(acc[i] + f[i]); }
    const float den = 255.0f * (float)(t1 - t0);
    for (int i = 0; i < FSZ; i++) out[i] = (float)acc[i] / den;
  }
  void grey(int t, float* g) const {
    const uint8_t* f = at(t);
    for (int i = 0; i < FPIX; i++) g[i] = (float)((int)f[3 * i] + (int)f[3 * i + 1] + (int)f[3 * i + 2]) / 765.0f;
  }
  void grey_sum(int t, int16_t* g) const {                                   // r+g+b per pixel (0..765), exact
    const uint8_t* f = at(t);
    for (int i = 0; i < FPIX; i++) g[i] = (int16_t)((int)f[3 * i] + (int)f[3 * i + 1] + (int)f[3 * i + 2]);
  }
  static float median_f(std::vector<float>& v) {
    size_t n = v.size(); if (!n) return 0.f;
    size_t m = n / 2; std::nth_element(v.begin(), v.begin() + m, v.end()); float hi = v[m];
    if (n & 1) return hi;
    float lo = *std::max_element(v.begin(), v.begin() + m); return (lo + hi) * 0.5f;
  }

  // ---- per-frame signals of the whole video (frame_signals + the per-video numbers of sbd_verifier3.py).  RAM mode only.
  void compute_signals() {
    if (sig) return;
    d1.assign(T, 0.f); d2.assign(T, 0.f); b1.assign(T, 0.f); rel.assign(T, 0.f);
    std::vector<double> fm(T);
    double tot = 0, sq = 0;
    for (int t = 0; t < T; t++) {
      const uint8_t* f = at(t); uint32_t s = 0; uint64_t q = 0;
      for (int j = 0; j < FSZ; j++) { s += f[j]; q += (uint32_t)f[j] * f[j]; }
      fm[t] = (double)s / (255.0 * FSZ); tot += (double)s; sq += (double)q;
    }
    for (int t = 1; t < T; t++) { d1[t] = (float)mad(at(t), at(t - 1)); b1[t] = (float)std::fabs((double)(float)fm[t] - (double)(float)fm[t - 1]); }
    for (int t = 2; t < T; t++) d2[t] = (float)mad(at(t), at(t - 2));
    float med = 0.f;
    if (T > 1) { std::vector<float> v(d1.begin() + 1, d1.end()); med = median_f(v); }
    float den = (float)((double)med + 1e-3);
    for (int t = 0; t < T; t++) { float q = d1[t] / den; q = q < 0.f ? 0.f : (q > 10.f ? 10.f : q); rel[t] = q / 10.f; }
    sig = true;
    // per-video numbers
    vm = med; vp90 = 0;
    if (T > 1) {
      std::vector<float> v(d1.begin() + 1, d1.end()); size_t N = v.size(); double vi = 0.9 * (double)(N - 1);
      size_t lo = (size_t)std::floor(vi), hi = std::min(lo + 1, N - 1);
      std::nth_element(v.begin(), v.begin() + lo, v.end()); double a = v[lo];
      double bq = a; if (hi != lo) { bq = *std::min_element(v.begin() + lo + 1, v.end()); }
      double g = vi - (double)lo; vp90 = a + (bq - a) * g;
    }
    const double N = (double)T * FSZ, mean = tot / (255.0 * N); vbright = mean;
    double var = sq / (255.0 * 255.0 * N) - mean * mean; vcontrast = std::sqrt(var > 0 ? var : 0.0);
  }
};

// ------------------------------------------------------------------------------------------------ candidate generation
// kind 0: AutoShot-style.  Peaks above base_thr are the normal cuts, peaks between low_thr and base_thr are borderline candidates.
inline void probs_runs(const std::vector<float>& p, float thr, std::vector<Pair>& out) {
  int n = (int)p.size(), i = 0; out.clear();
  while (i < n) {
    if (p[i] > thr) { int j = i; while (j + 1 < n && p[j + 1] > thr) j++; out.push_back({i, j}); i = j + 1; }
    else i++;
  }
}
inline void sort_cands(std::vector<Cand>& C) {
  std::stable_sort(C.begin(), C.end(), [](const Cand& x, const Cand& y) { return x.a != y.a ? x.a < y.a : x.b < y.b; });
}
// n = number of probabilities (= number of frames of the video)
inline void make_candidates_probs(const std::vector<float>& P, double base_thr, double low_thr, std::vector<Cand>& C) {
  int n = (int)P.size(); std::vector<Pair> rb, rl; C.clear();
  float bt = (float)base_thr, lt = (float)low_thr;
  probs_runs(P, bt, rb); probs_runs(P, lt, rl);
  for (auto& r : rb) if (r[0] > 0 && r[1] < n - 1) { Cand c; c.a = r[0]; c.b = r[1] + 1; c.src = SRC_BASE; C.push_back(c); }
  for (auto& r : rl) {
    if (!(r[0] > 0 && r[1] < n - 1)) continue;
    float mx = -1e30f; for (int k = r[0]; k <= r[1]; k++) mx = std::max(mx, P[k]);
    if (mx <= bt) { Cand c; c.a = r[0]; c.b = r[1] + 1; c.src = SRC_LOW; c.is_low = 1; C.push_back(c); }
  }
  sort_cands(C);
  for (auto& c : C) {
    int aa = clampi(c.a, 0, n - 1), bb = clampi(c.b, 0, n - 1);
    float m1 = -1e30f; for (int k = aa; k < std::min(std::max(bb, aa + 1), n); k++) m1 = std::max(m1, P[k]);
    float m2 = -1e30f; for (int k = std::max(0, aa - 2); k < std::min(n, bb + 3); k++) m2 = std::max(m2, P[k]);
    c.p_at = m1; c.p_near = m2;
  }
}
// kind 1: a list of cuts from any detector (a = last frame before, b = first frame after, optional score).
// Cuts that make no sense (a < 0 or b <= a) are dropped and counted in n_bad; frames beyond the end of the video are caught later (valid = 0).
inline void make_candidates_cuts(const std::vector<Pair>& cuts, const std::vector<float>& scores, std::vector<Cand>& C, int* n_bad = nullptr) {
  C.clear(); if (n_bad) *n_bad = 0;
  for (size_t i = 0; i < cuts.size(); i++) {
    if (cuts[i][0] < 0 || cuts[i][1] <= cuts[i][0]) { if (n_bad) (*n_bad)++; continue; }
    Cand c; c.a = cuts[i][0]; c.b = cuts[i][1]; c.src = SRC_USER;
    c.p_at = c.p_near = (i < scores.size() ? scores[i] : 1.f);
    C.push_back(c);
  }
  sort_cands(C);
}
// OmniShotCut output -> boundaries, exactly like omni_boundaries() / to_gt_format() in the Python files
inline void lab_flags(const std::string& lab, int* f) {
  std::string s; for (char ch : lab) s += (char)std::tolower((unsigned char)ch);
  f[0] = s.find("hard") != std::string::npos; f[1] = s.find("sudden") != std::string::npos;
  f[2] = s.find("transition") != std::string::npos; f[3] = s.find("new") != std::string::npos;
}
inline void make_candidates_ranges(const std::vector<Pair>& ranges, const std::vector<std::string>& intra, const std::vector<std::string>& inter,
                                   std::vector<Cand>& C) {
  C.clear(); std::vector<int> keep;
  for (size_t k = 0; k < ranges.size(); k++) if (intra.empty() || intra[k] == "General") keep.push_back((int)k);
  for (size_t j = 1; j < keep.size(); j++) {
    Pair prev = {ranges[keep[j - 1]][0], ranges[keep[j - 1]][1] - 1}, cur = {ranges[keep[j]][0], ranges[keep[j]][1] - 1};
    Cand c; c.a = prev[1]; c.b = cur[0]; c.src = SRC_DET; c.p_at = c.p_near = 1.f; c.n_dropped = keep[j] - keep[j - 1] - 1;
    if (!inter.empty()) lab_flags(inter[keep[j]], c.lab);
    C.push_back(c);
  }
}
// rescue_candidates(): big picture jumps nobody reported (RAM mode only; needs the signals of the whole video)
inline void add_rescue(Video& V, std::vector<Cand>& C, double ratio, double min_d, int pad = 3, int cap = 40) {
  int T = V.T; if (T < 8) return;
  struct Pk { float r; int t; };
  std::vector<Pk> pk; std::vector<float> buf;
  const float min_df = (float)min_d, ratf = (float)ratio;
  for (int t = 1; t < T; t++) {
    float v = V.d1[t];
    if (v < min_df) continue;
    int lo = std::max(1, t - 2), hi = std::min(T, t + 3); float mx = -1e30f;
    for (int k = lo; k < hi; k++) mx = std::max(mx, V.d1[k]);
    if (v < mx) continue;
    buf.assign(V.d1.begin() + std::max(1, t - 15), V.d1.begin() + std::min(T, t + 16));
    float loc = Video::median_f(buf);
    float den = loc + 0.001f;
    if (v < ratf * den) continue;
    bool near = false;
    for (auto& e : C) if (t >= e.a - pad && t - 1 <= e.b + pad) { near = true; break; }
    if (near) continue;
    pk.push_back({v / den, t});
  }
  std::sort(pk.begin(), pk.end(), [](const Pk& x, const Pk& y) { return x.r != y.r ? x.r > y.r : x.t > y.t; });
  if ((int)pk.size() > cap) pk.resize(cap);
  std::sort(pk.begin(), pk.end(), [](const Pk& x, const Pk& y) { return x.t < y.t; });
  for (auto& q : pk) { Cand c; c.a = q.t - 1; c.b = q.t; c.src = SRC_RESCUE; c.is_rescue = 1; c.p_at = c.p_near = 1.f; C.push_back(c); }
  sort_cands(C);
}

// ------------------------------------------------------------------------------------------------ features
inline void hist24(const uint8_t* f, float* h) {      // 3 channels x 8 bins, normalised by all 3888 pixels (hist3 in sbd_verifier.py)
  const uint8_t* bn = TB().bin8; int cnt[24] = {0};
  for (int i = 0; i < FPIX; i++) { cnt[bn[f[3 * i]]]++; cnt[8 + bn[f[3 * i + 1]]]++; cnt[16 + bn[f[3 * i + 2]]]++; }
  for (int k = 0; k < 24; k++) h[k] = (float)cnt[k] / (float)FSZ;
}
inline double mean_abs(const float* A, const float* B, int n) {
  double acc = 0; for (int i = 0; i < n; i += 96) { float s = 0.f; int e = std::min(n, i + 96); for (int j = i; j < e; j++) s += std::fabs(A[j] - B[j]); acc += s; }
  return acc / n;
}
inline double stab_of(const Video& V, int t0, int t1) {   // _stab(): mean change of the grey picture between neighbouring frames in [t0,t1)
  if (t1 - t0 < 2) return 0.0;
  static thread_local std::vector<int16_t> g0, g1; g0.resize(FPIX); g1.resize(FPIX);
  V.grey_sum(t0, g0.data()); uint64_t acc = 0;
  for (int t = t0 + 1; t < t1; t++) {
    V.grey_sum(t, g1.data()); uint32_t s = 0;
    for (int i = 0; i < FPIX; i++) s += (uint32_t)std::abs((int)g1[i] - (int)g0[i]);
    acc += s; g0.swap(g1);
  }
  return (double)acc / (765.0 * (double)(t1 - t0 - 1) * FPIX);
}
inline void edge_map(const float* g, float* out) {         // sbd_verifier2._edges: 26 x 47
  int k = 0;
  for (int y = 0; y < 26; y++) for (int x = 0; x < 47; x++) {
    float gx = g[y * FW + x + 1] - g[y * FW + x], gy = g[(y + 1) * FW + x] - g[y * FW + x];
    out[k++] = std::sqrt(gx * gx + gy * gy);
  }
}

// Everything needed to compute the features of one candidate.  The candidate list (C) must be complete and sorted before it is built.
// one(k, x) writes the NF numbers of candidate k into x.  It only reads frames inside [a-PAD_B, b+PAD_F] (PAD_B/PAD_F only for the
// local numbers; the whole-video numbers of the old models need RAM mode).
struct FeatCtx {
  Video& V; const std::vector<Cand>& C; const Need& nd;
  int n = 0; std::vector<double> cen, cs; double med_p = 0, low_share = 0;
  std::vector<float> P, Q, P2, ga, gb, ea, eb, seg;
  FeatCtx(Video& v, const std::vector<Cand>& c, const Need& need) : V(v), C(c), nd(need) {
    n = (int)C.size(); cen.resize(n); for (int k = 0; k < n; k++) cen[k] = (C[k].a + C[k].b) / 2.0;
    cs = cen; std::sort(cs.begin(), cs.end());
    if (nd.vid && n > 0) {
      std::vector<double> pa(n); for (int k = 0; k < n; k++) pa[k] = C[k].p_at;
      std::sort(pa.begin(), pa.end());
      med_p = (n & 1) ? pa[n / 2] : 0.5 * (pa[n / 2 - 1] + pa[n / 2]);
      double s = 0; for (auto& c2 : C) s += c2.is_low; low_share = s / n;
    }
    P.resize(FSZ); Q.resize(FSZ); P2.resize(FPIX); ga.resize(FPIX); gb.resize(FPIX); ea.resize(26 * 47); eb.resize(26 * 47);
  }
  // number of other candidate centres within +-r, and the distance to the nearest other centre (all from the sorted centre list)
  int count_within(double c, double r) const {
    auto lo = std::lower_bound(cs.begin(), cs.end(), c - r), hi = std::upper_bound(cs.begin(), cs.end(), c + r);
    return (int)(hi - lo) - 1;
  }
  double nearest_other(double c) const {
    auto lo = std::lower_bound(cs.begin(), cs.end(), c), hi = std::upper_bound(cs.begin(), cs.end(), c);
    if (hi - lo >= 2) return 0.0;
    double best = 1e18;
    if (lo != cs.begin()) best = std::min(best, c - *(lo - 1));
    if (hi != cs.end()) best = std::min(best, *hi - c);
    return best;
  }

  void one(int k, float* x) {
    const int T = V.T; const float* L = TB().lut; const Cand& c = C[k];
    std::fill(x, x + NF, 0.f);
    const int a = clampi(c.a, 0, T - 1), b = clampi(c.b, 0, T - 1);
    const uint8_t *fa = V.at(a), *fb = V.at(b);
    // ---- the cheap numbers that come with the candidate itself
    x[F_p_at] = c.p_at; x[F_p_near] = c.p_near; x[F_gap] = (float)(c.b - c.a - 1); x[F_is_low] = (float)c.is_low;
    {
      const double cc = cen[k];
      auto it = std::lower_bound(cs.begin(), cs.end(), cc - 1e-9);                 // values < c-1e-9 are before 'it'
      double prev = (it == cs.begin()) ? 0.0 : *(it - 1);
      auto jt = std::upper_bound(cs.begin(), cs.end(), cc + 1e-9);                 // first value > c+1e-9
      const bool has_next = (jt != cs.end()); double nxt = has_next ? *jt : (double)T;
      x[F_gap_prev] = (float)std::log1p(cc - prev);
      if (nd.f[F_gap_next]) x[F_gap_next] = (float)std::log1p(nxt - cc);          // old number: depends on the video length
      if (nd.f[F_density]) x[F_density] = (float)((double)cs.size() / (double)std::max(T, 1) * 1000.0);
      // new, length-free versions: distance to the neighbouring candidates, capped at 1000 frames
      x[F_gprev] = (float)std::log1p(it == cs.begin() ? 1000.0 : std::min(cc - *(it - 1), 1000.0));
      x[F_gnext] = (float)std::log1p(has_next ? std::min(*jt - cc, 1000.0) : 1000.0);
      x[F_n60] = (float)count_within(cc, 60);
    }
    x[F_lab_hard_cut] = (float)c.lab[0]; x[F_lab_sudden_jump] = (float)c.lab[1]; x[F_lab_transition] = (float)c.lab[2]; x[F_lab_new_start] = (float)c.lab[3];
    x[F_n_dropped] = (float)c.n_dropped; x[F_is_rescue] = (float)c.is_rescue;
    double d_adj = 0;
    // ---- pix_feats (sbd_verifier.py)
    if (nd.pix) {
      d_adj = rf(Video::mad(fb, fa));
      V.avgf(std::max(0, a - 3), a + 1, P.data()); V.avgf(b, std::min(T, b + 4), Q.data());
      double d_ctx = rf(mean_abs(Q.data(), P.data(), FSZ));
      int lo = std::max(0, a - 15), hi = std::min(T - 1, b + 15);
      seg.clear(); if (hi > lo) for (int t = lo + 1; t <= hi; t++) seg.push_back(V.get_d1(t)); else seg.push_back(0.f);
      double med = Video::median_f(seg);
      double spike = rf(clipd(d_adj / (med + 1e-3), 0, 50));
      double persist = rf(clipd(Video::mad(V.at(std::min(T - 1, b + 2)), V.at(std::max(0, a - 2))) / (d_adj + 1e-3), 0, 20));
      float ha[24], hb[24]; hist24(fa, ha); hist24(fb, hb);
      double hs = 0; for (int q = 0; q < 24; q++) hs += std::fabs(ha[q] - hb[q]);
      double ma = 0, mb = 0; for (int i = 0; i < FSZ; i++) { ma += L[fa[i]]; mb += L[fb[i]]; }
      x[F_d_adj] = (float)d_adj; x[F_d_ctx] = (float)d_ctx; x[F_spike] = (float)spike; x[F_persist] = (float)persist;
      x[F_hist_d] = (float)(0.5 * hs); x[F_bright] = (float)std::fabs(mb / FSZ - ma / FSZ);
    }
    // ---- local motion level: median frame-to-frame change in a +-60 frame window (replaces the whole-video numbers)
    if (nd.loc) {
      int lo = std::max(0, a - 60), hi = std::min(T - 1, b + 60);
      seg.clear(); if (hi > lo) for (int t = lo + 1; t <= hi; t++) seg.push_back(V.get_d1(t)); else seg.push_back(0.f);
      double lm = (double)Video::median_f(seg);
      x[F_loc_motion] = (float)lm; x[F_d_adj_loc] = (float)clipd(d_adj / (lm + 1e-3), 0, 50);
    }
    // ---- look_diff, look_frac on the 0..255 scale (sbd_refine.py)
    if (nd.look) {
      int bb2 = std::max(c.b, c.a + 1), pre0 = std::max(0, c.a - 3 + 1), pre1 = c.a + 1, post0 = bb2, post1 = std::min(T, bb2 + 3);
      pre1 = std::min(pre1, T);
      if (pre1 <= pre0 || post1 <= post0 || post0 >= T) { x[F_ld] = 1.f; x[F_lf] = 1.f; }
      else {
        const float kp = (float)(pre1 - pre0), kq = (float)(post1 - post0);
        uint16_t s1[FSZ], s2[FSZ]; const uint8_t* f0 = V.at(pre0); for (int i = 0; i < FSZ; i++) s1[i] = f0[i];
        for (int t = pre0 + 1; t < pre1; t++) { const uint8_t* f = V.at(t); for (int i = 0; i < FSZ; i++) s1[i] = (uint16_t)(s1[i] + f[i]); }
        f0 = V.at(post0); for (int i = 0; i < FSZ; i++) s2[i] = f0[i];
        for (int t = post0 + 1; t < post1; t++) { const uint8_t* f = V.at(t); for (int i = 0; i < FSZ; i++) s2[i] = (uint16_t)(s2[i] + f[i]); }
        for (int i = 0; i < FSZ; i++) { P[i] = (float)s1[i] / kp; Q[i] = (float)s2[i] / kq; }
        double acc = 0; for (int i = 0; i < FSZ; i += 96) { float s = 0.f; int e = std::min(FSZ, i + 96); for (int j = i; j < e; j++) s += std::fabs(P[j] - Q[j]); acc += s; }
        x[F_ld] = (float)((acc / FSZ) / 255.0);
        int cntp = 0; const float thr = (float)0.12;
        for (int i = 0; i < FPIX; i++) {
          float m = std::fabs(P[3 * i] - Q[3 * i]); m = std::max(m, std::fabs(P[3 * i + 1] - Q[3 * i + 1])); m = std::max(m, std::fabs(P[3 * i + 2] - Q[3 * i + 2]));
          if (m / 255.0f > thr) cntp++;
        }
        x[F_lf] = (float)cntp / (float)FPIX;
      }
    }
    // ---- pic_feats2 (sbd_verifier2.py)
    if (nd.pic2) {
      std::vector<float>& diff = P2;                       // per-pixel change (27 x 48)
      for (int i = 0; i < FPIX; i++) {
        float s = std::fabs(L[fb[3 * i]] - L[fa[3 * i]]); s += std::fabs(L[fb[3 * i + 1]] - L[fa[3 * i + 1]]); s += std::fabs(L[fb[3 * i + 2]] - L[fa[3 * i + 2]]);
        diff[i] = s / 3.0f;
      }
      double dsum = 0; int f05 = 0, f10 = 0, f20 = 0;
      const float t05 = (float)0.05, t10 = (float)0.10, t20 = (float)0.20;
      for (int i = 0; i < FPIX; i++) { dsum += diff[i]; f05 += diff[i] > t05; f10 += diff[i] > t10; f20 += diff[i] > t20; }
      double d_adj2 = rf(dsum / FPIX);
      V.grey(a, ga.data()); V.grey(b, gb.data());
      // best match after sliding the picture up to 2 pixels
      double best = 1e9; const int s = 2;
      for (int dy = -s; dy <= s; dy++) for (int dx = -s; dx <= s; dx++) {
        double acc = 0;
        for (int y = s; y < FH - s; y++) {
          const float* ra = &ga[y * FW + s]; const float* rb = &gb[(y + dy) * FW + s + dx]; float rs = 0.f;
          for (int xx = 0; xx < FW - 2 * s; xx++) rs += std::fabs(ra[xx] - rb[xx]);
          acc += rs;
        }
        double dm = acc / ((FH - 2 * s) * (FW - 2 * s)); if (dm < best) best = dm;
      }
      double mc = rf(best);
      double full = mean_abs(gb.data(), ga.data(), FPIX);
      double mc_ratio = rf(clipd(mc / (full + 1e-3), 0, 2));
      edge_map(ga.data(), ea.data()); edge_map(gb.data(), eb.data());
      double edge_d = rf(mean_abs(eb.data(), ea.data(), 26 * 47));
      double ma = 0, mb = 0; for (int i = 0; i < 26 * 47; i++) { ma += ea[i]; mb += eb[i]; }
      ma /= 26 * 47; mb /= 26 * 47;
      double edge_rel = rf(clipd(edge_d / (ma + mb + 1e-3), 0, 5));
      double ctr = 0; int nc = 0; double tot = 0;
      for (int y = 0; y < FH; y++) for (int xx = 0; xx < FW; xx++) { double v = diff[y * FW + xx]; tot += v; if (y >= 7 && y < 20 && xx >= 12 && xx < 36) { ctr += v; nc++; } }
      double ctr_d = ctr / nc, brd_d = (tot - ctr) / (FPIX - nc);
      V.avgf(std::max(0, a - 5), std::max(1, a - 1), P.data());
      { int lo = std::min(T - 1, b + 3), hi = std::min(T, b + 7); V.avgf(lo, std::max(hi, lo + 1), Q.data()); }
      double d_wide = rf(mean_abs(Q.data(), P.data(), FSZ));
      double wide_ratio = rf(clipd(d_wide / (d_adj2 + 1e-3), 0, 20));
      double pre_stab = rf(stab_of(V, std::max(0, a - 8), a + 1)), post_stab = rf(stab_of(V, b, std::min(T, b + 9)));
      double cut_vs_stab = rf(clipd(d_adj2 / (pre_stab + post_stab + 1e-3), 0, 50));
      x[F_mc_d] = (float)mc; x[F_mc_ratio] = (float)mc_ratio; x[F_edge_d] = (float)edge_d; x[F_edge_rel] = (float)edge_rel;
      x[F_frac05] = (float)f05 / FPIX; x[F_frac10] = (float)f10 / FPIX; x[F_frac20] = (float)f20 / FPIX;
      x[F_ctr_d] = (float)ctr_d; x[F_brd_d] = (float)brd_d; x[F_d_wide] = (float)d_wide; x[F_wide_ratio] = (float)wide_ratio;
      x[F_pre_stab] = (float)pre_stab; x[F_post_stab] = (float)post_stab; x[F_cut_vs_stab] = (float)cut_vs_stab;
    }
    // ---- crowding (extra_feats in sbd_verifier2.py)
    if (nd.crowd) {
      const double cc = cen[k];
      x[F_n15] = (float)count_within(cc, 15);
      if (nd.f[F_dup_dist] || nd.f[F_dup_p]) {
        const double dj = n > 1 ? nearest_other(cc) : 1e18;
        x[F_dup_dist] = (n > 1) ? (float)std::min(dj, 30.0) : 30.f;
        if (nd.f[F_dup_p]) {                                  // needs WHICH candidate is nearest (first minimum, like argmin)
          int j = -1; double dm = 1e9;
          for (int q = 0; q < n; q++) { if (q == k) continue; double dq = std::fabs(cen[q] - cen[k]); if (dq < dm) { dm = dq; j = q; } }
          x[F_dup_p] = (j >= 0 && dm <= 30) ? C[j].p_at : 0.f;
        }
      }
      x[F_nb_p_prev] = k > 0 ? C[k - 1].p_at : 0.f; x[F_nb_p_next] = k + 1 < n ? C[k + 1].p_at : 0.f;
      if (nd.f[F_p_rank]) {
        double m = C[k].p_at;
        for (int q = 0; q < n; q++) if (q != k && std::fabs(cen[q] - cen[k]) <= 10) m = std::max(m, (double)C[q].p_at);
        x[F_p_rank] = (float)((double)C[k].p_at / (m + 1e-6));
      }
    }
    // ---- per-video features (whole-video numbers; RAM mode only)
    if (nd.vid) {
      x[F_vid_motion] = (float)V.vm; x[F_vid_motion_p90] = (float)V.vp90; x[F_vid_bright] = (float)V.vbright; x[F_vid_contrast] = (float)V.vcontrast;
      x[F_vid_len] = (float)std::log((double)std::max(T, 1)); x[F_vid_low_share] = (float)low_share;
      x[F_d_adj_rel] = (float)clipd(d_adj / (V.vm + 1e-3), 0, 50);
      x[F_p_vs_vid] = (float)clipd((double)C[k].p_at / (med_p + 1e-3), 0, 20);
    }
  }
};

// Feature table of a whole candidate list (RAM mode; used for training and by run_pipeline).  Invalid candidates keep a row of zeros.
inline void compute_features(Video& V, std::vector<Cand>& C, const Need& nd, std::vector<float>& X) {
  const int n = (int)C.size(); X.assign((size_t)n * NF, 0.f);
  if (!n) return;
  if (nd.sig) V.compute_signals();
  FeatCtx ctx(V, C, nd);
  for (int k = 0; k < n; k++) if (C[k].valid) ctx.one(k, &X[(size_t)k * NF]);
}

// ------------------------------------------------------------------------------------------------ models
inline double sigmoid(double z) { return 1.0 / (1.0 + std::exp(-z)); }
inline double eval_tree(const Trees& t, int root, const double* x) {
  int i = root;
  while (t.feat[i] >= 0) i = (x[t.feat[i]] <= t.thr[i]) ? t.left[i] : t.right[i];
  return t.val[i];
}
static void conv1d_relu(const float* in, int Cin, int Lw, const float* w, const float* bs, int Cout, int K, int dil, int pad, float* out) {
  for (int co = 0; co < Cout; co++) {
    float* o = out + (size_t)co * Lw;
    for (int t = 0; t < Lw; t++) o[t] = bs[co];
    for (int ci = 0; ci < Cin; ci++) {
      const float* xi = in + (size_t)ci * Lw;
      for (int k = 0; k < K; k++) {
        const float wk = w[((size_t)co * Cin + ci) * K + k]; const int off = k * dil - pad;
        const int t0 = std::max(0, -off), t1 = std::min(Lw, Lw - off);
        for (int t = t0; t < t1; t++) o[t] += wk * xi[t + off];
      }
    }
    for (int t = 0; t < Lw; t++) o[t] = o[t] > 0.f ? o[t] : 0.f;
  }
}
inline double eval_tcn(const Member& m, const float* win /*6 x 64*/, const double* xs /*selected raw features*/) {
  float h0[24 * TW], h1[24 * TW], h2[32 * TW];
  conv1d_relu(win, TCH, TW, m.c0w.data(), m.c0b.data(), 24, 5, 1, 2, h0);
  conv1d_relu(h0, 24, TW, m.c1w.data(), m.c1b.data(), 24, 5, 2, 4, h1);
  conv1d_relu(h1, 24, TW, m.c2w.data(), m.c2b.data(), 32, 3, 2, 2, h2);
  float z[96]; int zn = 64; const int c = TW / 2;
  for (int ch = 0; ch < 32; ch++) {
    const float* r = h2 + ch * TW; float mx = r[0]; for (int t = 1; t < TW; t++) mx = std::max(mx, r[t]);
    float s = 0.f; for (int t = c - 4; t < c + 4; t++) s += r[t];
    z[ch] = mx; z[32 + ch] = s / 8.0f;
  }
  if (m.n_tab) {
    float tn[256];
    for (int j = 0; j < m.n_tab; j++) {
      float v = (float)xs[j]; v = (v - m.mu[j]) / m.sd[j]; tn[j] = v < -8.f ? -8.f : (v > 8.f ? 8.f : v);
    }
    for (int o = 0; o < 32; o++) {
      float s = m.tabb[o]; for (int j = 0; j < m.n_tab; j++) s += m.tabw[(size_t)o * m.n_tab + j] * tn[j];
      z[64 + o] = s > 0.f ? s : 0.f;
    }
    zn = 96;
  }
  float hid[32];
  for (int o = 0; o < 32; o++) {
    float s = m.h0b[o]; for (int j = 0; j < zn; j++) s += m.h0w[(size_t)o * zn + j] * z[j];
    hid[o] = s > 0.f ? s : 0.f;
  }
  float out = m.h1b[0]; for (int j = 0; j < 32; j++) out += m.h1w[j] * hid[j];
  return sigmoid((double)out);
}

// the 6 curves around a candidate (window() in sbd_verifier3.py).  RAM mode only (needs the signals of the whole video).
inline void make_window(const Video& V, const Cand& c, float* out) {
  const int T = V.T, ctr = (c.a + c.b) / 2, lo = ctr - TW / 2, hi = lo + TW;
  std::memset(out, 0, sizeof(float) * TCH * TW);
  const int s = std::max(lo, 0), e = std::min(hi, T);
  auto clip3 = [](float v) { return v < 0.f ? 0.f : (v > 3.f ? 3.f : v); };
  for (int t = s; t < e; t++) {
    int j = t - lo;
    out[0 * TW + j] = V.p.empty() ? 0.f : V.p[t];
    out[1 * TW + j] = clip3(V.d1[t] * 5.f); out[2 * TW + j] = clip3(V.d2[t] * 5.f); out[3 * TW + j] = clip3(V.b1[t] * 10.f);
    out[4 * TW + j] = V.rel[t];
  }
  const int m0 = std::max(c.a, lo), m1 = std::min(c.b, hi - 1);
  if (m1 >= m0) for (int t = m0; t <= m1; t++) out[5 * TW + (t - lo)] = 1.0f;
}

// probability of KEEP for one candidate: the members are averaged (summed in order, then divided, like the Python code)
inline double predict_one(const Model& M, const float* row, const float* win) {
  double xs[256], acc = 0;
  for (auto& m : M.members) {
    for (size_t j = 0; j < m.cols.size(); j++) xs[j] = (double)row[m.cols[j]];
    double p = 0;
    if (m.type == M_LOGREG) {
      double z = m.b;
      for (size_t j = 0; j < m.cols.size(); j++) { double v = xs[j]; if (m.has_scale) v = (v - m.mean[j]) / m.scale[j]; z += m.w[j] * v; }
      p = sigmoid(z);
    } else if (m.type == M_BOOST) {
      double z = m.bias; for (int32_t r : m.tr.root) z += eval_tree(m.tr, r, xs);
      p = sigmoid(z);
    } else if (m.type == M_BAG) {
      double s = 0; for (int32_t r : m.tr.root) s += eval_tree(m.tr, r, xs);
      p = s / (double)m.tr.root.size();
    } else p = eval_tcn(m, win, xs);
    acc += p;
  }
  return acc / (double)M.members.size();
}

// ------------------------------------------------------------------------------------------------ decisions -> shots
inline void build_scenes(const std::vector<Pair>& ab, int n, std::vector<Pair>& out, std::vector<int>& used) {
  out.clear(); used.clear(); int start = 0;
  for (size_t k = 0; k < ab.size(); k++) {
    int a = ab[k][0], b = ab[k][1];
    if (a < start || b <= a || b > n - 1) continue;
    out.push_back({start, a}); start = b; used.push_back((int)k);
  }
  out.push_back({start, n - 1});
}
inline void dedupe_mask(const std::vector<double>& cen, const std::vector<double>& pk, std::vector<uint8_t>& kept, int dist) {
  if (dist <= 0) return;
  std::vector<int> ord(cen.size()); std::iota(ord.begin(), ord.end(), 0);
  std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return cen[x] < cen[y]; });
  int last = -1;
  for (int i : ord) {
    if (!kept[i]) continue;
    if (last >= 0 && cen[i] - cen[last] <= dist) {
      if (pk[i] > pk[last]) { kept[last] = 0; last = i; } else kept[i] = 0;
    } else last = i;
  }
}
inline void snap_scenes(Video& V, std::vector<Pair>& S, int w, double min_ratio = 1.5) {
  const int n = V.T; if (w <= 0 || n < 2 || S.size() < 2) return;
  std::vector<float> d(n, 0.f);
  for (int t = 1; t < n; t++) {
    const uint8_t *f = V.at(t), *g = V.at(t - 1); long s = 0;
    for (int i = 0; i < FSZ; i++) s += std::abs((int)f[i] - (int)g[i]);
    d[t] = (float)((double)s / (double)FSZ / 255.0);
  }
  for (size_t i = 1; i < S.size(); i++) {
    if (S[i][0] - S[i - 1][1] != 1) continue;
    int b = S[i][0], lo = std::max({S[i - 1][0] + 1, b - w, 1}), hi = std::min({S[i][1], b + w, n - 1});
    if (hi < lo) continue;
    int best = lo; for (int t = lo + 1; t <= hi; t++) if (d[t] > d[best]) best = t;
    if (best != b && (double)d[best] > min_ratio * std::max((double)d[b], 1e-6)) { S[i][0] = best; S[i - 1][1] = best - 1; }
  }
}

// ------------------------------------------------------------------------------------------------ the whole thing
struct Input {
  const float* probs = nullptr; int nprobs = 0;                                  // kind 0
  std::vector<Pair> cuts; std::vector<float> scores;                            // kind 1: (a,b[,score]) per cut
  std::vector<Pair> ranges; std::vector<std::string> intra, inter;              // kind 1 alternative: OmniShotCut ranges (half-open) with labels
};
struct Output {
  std::vector<Cand> cands; std::vector<double> p_keep; std::vector<uint8_t> kept;   // one entry per candidate (p_keep = -1 when invalid)
  std::vector<Pair> scenes;                                                     // final shots, inclusive [start, end]
  double t_us[6] = {0, 0, 0, 0, 0, 0};                                          // candidates, (unused), features, models, decisions, total
  std::vector<float> X;                                                         // feature table (only filled when keep_features)
  int n_bad = 0;                                                                // cuts dropped because they make no sense (a < 0, b <= a)
  int n_invalid = 0;                                                            // candidates that refer to frames beyond the end of the video
  int T = 0;                                                                    // number of frames processed
};

// decisions -> kept flags and final shots (shared by RAM mode and stream mode)
inline void decide(const Model& M, int T, Video* V_for_snap, Output& out) {
  const int n = (int)out.cands.size();
  std::vector<uint8_t> keep(n);
  for (int k = 0; k < n; k++) { const Cand& c = out.cands[k]; double th = c.is_rescue ? M.tau_rescue : M.tau; keep[k] = c.valid && out.p_keep[k] >= th; }
  if (M.dedupe > 0) { std::vector<double> cen(n); for (int k = 0; k < n; k++) cen[k] = (out.cands[k].a + out.cands[k].b) / 2.0; dedupe_mask(cen, out.p_keep, keep, M.dedupe); }
  std::vector<Pair> ab; std::vector<int> idx;
  for (int k = 0; k < n; k++) if (keep[k]) { ab.push_back({out.cands[k].a, out.cands[k].b}); idx.push_back(k); }
  std::vector<int> used; build_scenes(ab, T, out.scenes, used);
  out.kept.assign(n, 0); for (int u : used) out.kept[idx[u]] = 1;
  if (M.snap_w > 0 && V_for_snap && out.scenes.size() > 1) snap_scenes(*V_for_snap, out.scenes, M.snap_w);
}

inline bool run_pipeline(const Model& M, const uint8_t* frames, int T, const Input& in, Output& out, std::string& err, bool keep_features = false,
                         const Need* force_need = nullptr) {
  double t0 = now_us();
  if (T < 2) { err = "need at least 2 frames"; return false; }
  if (!frames) { err = "no frames"; return false; }
  Video V; V.d = frames; V.cap = T; V.T = T;
  Need nd = force_need ? *force_need : M.need; if (force_need) nd.derive();
  std::vector<Cand> C; out = Output(); out.T = T;
  const bool probs_mode = (M.input_kind == 0);
  if (probs_mode) {
    if (!in.probs || in.nprobs <= 0) { err = "this model needs per-frame probabilities from the detector (--probs)"; return false; }
    V.p.assign(T, 0.f); int m = std::min(T, in.nprobs); std::memcpy(V.p.data(), in.probs, sizeof(float) * m);
    make_candidates_probs(V.p, M.base_thr, M.low_thr, C);
  } else {
    if (!in.ranges.empty()) {
      make_candidates_ranges(in.ranges, in.intra, in.inter, C);
      for (auto& c : C) { c.a = clampi(c.a, 0, T - 1); c.b = clampi(c.b, 0, T - 1); }       // same as the Python code for OmniShotCut ranges
    } else make_candidates_cuts(in.cuts, in.scores, C, &out.n_bad);
    if (M.rescue) { V.compute_signals(); add_rescue(V, C, M.r_ratio, M.r_min); }
  }
  for (auto& c : C) if (c.a >= T || c.b > T - 1 || c.b <= c.a) { c.valid = 0; out.n_invalid++; }
  double t1 = now_us();
  std::vector<float>& X = out.X;
  compute_features(V, C, nd, X);
  double t2 = now_us();
  out.cands = C;
  if (C.empty()) { out.scenes.push_back({0, T - 1}); out.t_us[0] = t1 - t0; out.t_us[2] = t2 - t1; out.t_us[5] = now_us() - t0; return true; }
  const int n = (int)C.size();
  out.p_keep.assign(n, -1.0);
  std::vector<float> win;
  if (M.need.win) { V.compute_signals(); win.resize((size_t)TCH * TW); }
  for (int k = 0; k < n; k++) {
    if (!C[k].valid) continue;
    if (M.need.win) make_window(V, C[k], win.data());
    out.p_keep[k] = predict_one(M, &X[(size_t)k * NF], win.data());
  }
  double t3 = now_us();
  decide(M, T, &V, out);
  double t4 = now_us();
  out.t_us[0] = t1 - t0; out.t_us[1] = 0; out.t_us[2] = t2 - t1; out.t_us[3] = t3 - t2; out.t_us[4] = t4 - t3; out.t_us[5] = t4 - t0;
  if (!keep_features) { std::vector<float>().swap(out.X); }
  return true;
}

// ------------------------------------------------------------------------------------------------ stream mode
// Usage:   Streamer S; S.init(model, input, keep_features, err);
//          for each frame:  memcpy(S.slot(), frame, FSZ); S.commit();
//          S.finish(output, err);
// Memory = (largest candidate gap + 124) frames + the candidate list.  Needs a model with streamable() == true.
struct Streamer {
  const Model* M = nullptr; std::vector<Cand> C; Video V; std::vector<uint8_t> ring; long cap = 0;
  std::vector<int> order; size_t nxt = 0; std::unique_ptr<FeatCtx> ctx;
  std::vector<double> pk; std::vector<float> X, row; bool keep_features = false; int head = -1, n_bad = 0;
  double t_cand = 0, t_feat = 0, t_model = 0; Need nd; int nprobs_hint = 0;

  bool init(const Model& model, const Input& in, bool keep_feat, std::string& err) {
    M = &model; keep_features = keep_feat;
    if (!model.streamable()) { err = "this model needs whole-video numbers (old v1-v3/omni models): it cannot run in stream mode; use RAM mode"; return false; }
    double t0 = now_us(); nd = model.need;
    if (model.input_kind == 0) {
      if (!in.probs || in.nprobs <= 0) { err = "this model needs per-frame probabilities from the detector (--probs)"; return false; }
      V.p.assign(in.probs, in.probs + in.nprobs); nprobs_hint = in.nprobs;
      make_candidates_probs(V.p, model.base_thr, model.low_thr, C);
    } else {
      if (!in.ranges.empty()) {
        make_candidates_ranges(in.ranges, in.intra, in.inter, C);
        for (auto& c : C) { c.a = std::max(c.a, 0); c.b = std::max(c.b, 0); }
      } else make_candidates_cuts(in.cuts, in.scores, C, &n_bad);
    }
    int maxgap = 1;
    for (auto& c : C) { if (c.b <= c.a) { c.valid = 0; continue; } maxgap = std::max(maxgap, c.b - c.a); }
    cap = (long)maxgap + PAD_B + PAD_F + 3;
    if ((double)cap * FSZ > 1.5e9) { err = "a candidate cut spans more than ~100000 frames; cannot stream"; return false; }
    ring.assign((size_t)cap * FSZ, 0); V.d = ring.data(); V.cap = cap; V.T = 0;
    order.resize(C.size()); std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return C[x].b != C[y].b ? C[x].b < C[y].b : x < y; });
    ctx.reset(new FeatCtx(V, C, nd)); pk.assign(C.size(), -1.0); row.assign(NF, 0.f);
    if (keep_features) X.assign(C.size() * (size_t)NF, 0.f);
    t_cand = now_us() - t0; return true;
  }
  uint8_t* slot() { return ring.data() + (size_t)((long)(head + 1) % cap) * FSZ; }     // where the next frame goes
  void process(size_t k_ord) {
    const int k = order[k_ord]; if (!C[k].valid) return;
    double t0 = now_us(); float* x = keep_features ? &X[(size_t)k * NF] : row.data();
    ctx->one(k, x); double t1 = now_us();
    pk[k] = predict_one(*M, x, nullptr); t_feat += t1 - t0; t_model += now_us() - t1;
  }
  void commit() {                                    // the frame in slot() is complete
    head++; V.T = head + 1;
    while (nxt < order.size()) {
      const Cand& c = C[order[nxt]];
      if (c.valid && c.b + PAD_F > head) break;       // its look-ahead is not here yet (candidates are ordered by b)
      process(nxt); nxt++;
    }
  }
  bool finish(Output& out, std::string& err) {
    const int T = head + 1; double t0 = now_us();
    if (T < 2) { err = "need at least 2 frames"; return false; }
    V.T = T; out = Output(); out.T = T; out.n_bad = n_bad;
    for (auto& c : C) if (c.valid && (c.a >= T || c.b > T - 1)) { c.valid = 0; out.n_invalid++; }
    // candidates that became invalid may sit anywhere in `order`; the ones not processed yet are finished now with the final T
    for (; nxt < order.size(); nxt++) process(nxt);
    out.cands = C; out.p_keep = pk;
    if (C.empty()) out.scenes.push_back({0, T - 1}); else decide(*M, T, nullptr, out);
    if (keep_features) out.X = X;
    out.t_us[0] = t_cand; out.t_us[2] = t_feat; out.t_us[3] = t_model; out.t_us[4] = now_us() - t0; out.t_us[5] = t_cand + t_feat + t_model + out.t_us[4];
    return true;
  }
};

// Rough count of arithmetic operations (adds, multiplies, compares) for one run.  Counted from the loops above; used for the efficiency table.
// It is an ESTIMATE (order of magnitude), not a profiler measurement.
inline double estimate_flops(const Model& M, int T, int ncand) {
  const Need& nd = M.need; double f = 0;
  if (nd.sig || M.rescue) f += (double)T * (FSZ * 3.0 /*d1*/ + FSZ * 3.0 /*d2*/ + FSZ * 1.0 /*mean*/ + FSZ * 3.0 /*std pass*/);
  double per = 0;
  const double d1win = (nd.sig ? 0.0 : 1.0);
  if (nd.pix) per += FSZ * 3.0 /*d_adj*/ + FSZ * (4 + 4) /*pre/post mean*/ + FSZ * 2.0 /*d_ctx*/ + 30.0 * FSZ * 3.0 * d1win /*spike window*/ + FSZ * 3.0 /*persist*/ + FSZ * 4.0 /*hist*/ + FSZ * 2.0 /*bright*/;
  if (nd.loc) per += 120.0 * FSZ * 3.0 * d1win /*+-60 frame window of frame-to-frame changes*/ + 120.0 * 8.0 /*median*/;
  if (nd.look) per += FSZ * (6 + 2 + 2) + FPIX * 4.0;
  if (nd.pic2) per += FPIX * 6.0 /*diff*/ + FPIX * 8.0 /*fracs, sums*/ + 2 * FPIX * 4.0 /*grey*/ + 25.0 * 23 * 44 * 2.0 /*shifts*/ + 2 * 26 * 47 * 6.0 /*edges*/ + FSZ * (8 + 8) /*wide means*/ + 20.0 * FPIX * 4.0 /*stab*/;
  if (nd.crowd) per += 40.0 + (nd.f[F_dup_p] || nd.f[F_p_rank] ? 6.0 * ncand : 0.0);
  f += per * ncand;
  for (auto& m : M.members) {
    if (m.type == M_LOGREG) f += (2.0 * m.w.size() + 4) * ncand;
    else if (m.type == M_BOOST || m.type == M_BAG) { f += (double)m.tr.root.size() * 10.0 * ncand; }   // ~ depth 8 compares + the add, upper bound
    else f += (2.0 * (24 * TCH * 5 + 24 * 24 * 5 + 32 * 24 * 3) * TW + 2.0 * 32 * 96 + 2.0 * 32 * (64 + 32) + 2.0 * 32 * m.n_tab) * ncand;
  }
  return f;
}

}  // namespace sbdv
