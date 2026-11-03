#include "sbdv.h"
#include "sbdv.hpp"
#include <cstring>

using namespace sbdv;
struct sbdv_model { Model m; bool lenient = false; int last_invalid = 0, last_bad = 0; };

static void set_err(char* err, int errlen, const std::string& s) { if (err && errlen > 0) { std::strncpy(err, s.c_str(), (size_t)errlen - 1); err[errlen - 1] = 0; } }

extern "C" {
const char* sbdv_version(void) { return SBDV_VERSION; }
sbdv_model* sbdv_load(const char* path, char* err, int errlen) {
  auto* h = new sbdv_model(); std::string e;
  if (!h->m.load(path, e)) { set_err(err, errlen, e); delete h; return nullptr; }
  return h;
}
void sbdv_free(sbdv_model* h) { delete h; }
int sbdv_input_kind(const sbdv_model* h) { return h->m.input_kind; }
double sbdv_get(const sbdv_model* h, const char* w) {
  std::string s = w; const Model& M = h->m;
  if (s == "tau") return M.tau;
  if (s == "base_thr") return M.base_thr;
  if (s == "low_thr") return M.low_thr;
  if (s == "params") return (double)M.n_params();
  if (s == "trees") return (double)M.n_trees();
  if (s == "nodes") return (double)M.n_nodes();
  if (s == "nets") return (double)M.n_tcn();
  if (s == "bytes") return (double)M.file_bytes;
  if (s == "rescue") return M.rescue;
  if (s == "dedupe") return M.dedupe;
  if (s == "snap") return M.snap_w;
  if (s == "tau_rescue") return M.tau_rescue;
  if (s == "streamable") return M.streamable() ? 1 : 0;
  if (s == "last_invalid") return h->last_invalid;
  if (s == "last_bad") return h->last_bad;
  return -1;
}
void sbdv_set_tau(sbdv_model* h, double tau, double tau_rescue) { h->m.tau = tau; h->m.tau_rescue = tau_rescue; }
int sbdv_set_option(sbdv_model* h, const char* name, double value) {
  std::string s = name; if (s == "lenient") { h->lenient = value != 0; return 0; } return -1;
}
double sbdv_flops(const sbdv_model* h, int T, int ncand) { return estimate_flops(h->m, T, ncand); }

static void fill_input(Input& in, const float* probs, int nprobs, const int32_t* cuts, const float* scores, int ncuts, const int32_t* ranges, int nranges, const char* labels) {
  in.probs = probs; in.nprobs = nprobs;
  for (int i = 0; i < ncuts; i++) { in.cuts.push_back({cuts[2 * i], cuts[2 * i + 1]}); if (scores) in.scores.push_back(scores[i]); }
  if (nranges > 0) {
    std::vector<std::string> L; if (labels) { std::stringstream ss(labels); std::string ln; while (std::getline(ss, ln)) L.push_back(ln); }
    for (int i = 0; i < nranges; i++) {
      in.ranges.push_back({ranges[2 * i], ranges[2 * i + 1]});
      std::string a = "General", b; if (i < (int)L.size()) { size_t t = L[i].find('\t'); a = L[i].substr(0, t); if (t != std::string::npos) b = L[i].substr(t + 1); if (a.empty() || a == "-") a = "General"; }
      in.intra.push_back(a); in.inter.push_back(b);
    }
  }
}
static std::string invalid_msg(const Output& o) {
  std::string ex; int shown = 0; for (auto& c : o.cands) if (!c.valid && shown < 3) { ex += (shown ? ", " : "") + std::string("(") + std::to_string(c.a) + "," + std::to_string(c.b) + ")"; shown++; }
  return std::to_string(o.n_invalid) + " cut(s) point outside the video (" + std::to_string(o.T) + " frames), e.g. " + ex + ": wrong video, frame rate or frame numbering?";
}
int sbdv_verify(sbdv_model* h, const uint8_t* frames, int T, const float* probs, int nprobs, const int32_t* cuts, const float* scores, int ncuts, const int32_t* ranges,
                int nranges, const char* labels, int32_t* out_ab, double* out_p, uint8_t* out_kept, int32_t* out_src, int cap, int32_t* out_scenes, int cap_sc,
                int* n_scenes, double* timing_us, char* err, int errlen) {
  Input in; fill_input(in, probs, nprobs, cuts, scores, ncuts, ranges, nranges, labels);
  Output o; std::string e;
  try { if (!run_pipeline(h->m, frames, T, in, o, e)) { set_err(err, errlen, e); return -1; } }
  catch (const std::exception& x) { set_err(err, errlen, x.what()); return -1; }
  h->last_invalid = o.n_invalid; h->last_bad = o.n_bad;
  if (o.n_invalid > 0 && !h->lenient) { set_err(err, errlen, invalid_msg(o)); return -1; }
  int n = (int)o.cands.size();
  for (int i = 0; i < n && i < cap; i++) {
    out_ab[2 * i] = o.cands[i].a; out_ab[2 * i + 1] = o.cands[i].b; out_p[i] = o.p_keep.empty() ? 0.0 : o.p_keep[i];
    out_kept[i] = o.kept.empty() ? 0 : o.kept[i]; out_src[i] = o.cands[i].src;
  }
  int ns = (int)o.scenes.size(); if (n_scenes) *n_scenes = ns;
  for (int i = 0; i < ns && i < cap_sc; i++) { out_scenes[2 * i] = o.scenes[i][0]; out_scenes[2 * i + 1] = o.scenes[i][1]; }
  if (timing_us) for (int i = 0; i < 6; i++) timing_us[i] = o.t_us[i];
  return n;
}
int sbdv_n_features(void) { return NF; }
const char* sbdv_feature_name(int i) { return (i >= 0 && i < NF) ? FEAT_NAME[i] : ""; }
int sbdv_features(sbdv_model* h, const uint8_t* frames, int T, const float* probs, int nprobs, const int32_t* cuts, const float* scores, int ncuts, const int32_t* ranges,
                  int nranges, const char* labels, int32_t* out_ab, int32_t* out_src, float* out_feat, int cap, char* err, int errlen) {
  Input in; fill_input(in, probs, nprobs, cuts, scores, ncuts, ranges, nranges, labels);
  Need nd; for (int i = 0; i < NF; i++) nd.f[i] = !(i == F_as_p_at || i == F_as_p_near || i == F_as_agrees);   // everything except the hybrid diagnostic columns
  nd.win = false; Output o; std::string e;
  Model tmp = h->m;                                                       // candidates use the model's own rules; models are not evaluated
  tmp.members.clear();
  try {
    // run only up to the feature table: use a model with one dummy logistic member so the prediction step is cheap
    Member dm; dm.type = M_LOGREG; dm.cols = {0}; dm.w = {0.0}; dm.b = 0; tmp.members.push_back(dm);
    if (!run_pipeline(tmp, frames, T, in, o, e, true, &nd)) { set_err(err, errlen, e); return -1; }
  } catch (const std::exception& x) { set_err(err, errlen, x.what()); return -1; }
  h->last_invalid = o.n_invalid; h->last_bad = o.n_bad;
  if (o.n_invalid > 0 && !h->lenient) { set_err(err, errlen, invalid_msg(o)); return -1; }
  int n = 0;
  for (size_t i = 0; i < o.cands.size(); i++) {
    if (!o.cands[i].valid) continue;
    if (n < cap) {
      out_ab[2 * n] = o.cands[i].a; out_ab[2 * n + 1] = o.cands[i].b; out_src[n] = o.cands[i].src;
      std::memcpy(out_feat + (size_t)n * NF, &o.X[i * (size_t)NF], sizeof(float) * NF);
    }
    n++;
  }
  return n;
}
}
