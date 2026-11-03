/* sbdv.h - C interface of the SBD-Verifier (for video editors, Python ctypes, other languages).  Version 2.
 *
 *   sbdv_model* m = sbdv_load("generic.sbdm", err, sizeof err);
 *   // frames: uint8 [T][27][48][3] RGB, the video shrunk to 48x27 (ffmpeg: -s 48x27 -pix_fmt rgb24)
 *   int n = sbdv_verify(m, frames, T, NULL, 0, cuts, NULL, ncuts, NULL, 0, NULL, ...);
 *
 * One call does everything: proposes candidates from the detector output, measures the pictures around them,
 * decides KEEP/DROP and builds the final shot list.  A model object can be used by one thread at a time (it holds
 * the "last run" counters); give every thread its own sbdv_model.
 * This C interface runs in RAM mode (all frames in memory).  For videos too long for memory use the command line
 * tool, which streams the frames through a small ring buffer.
 */
#ifndef SBDV_H
#define SBDV_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#if defined(_WIN32)
#define SBDV_API __declspec(dllexport)
#else
#define SBDV_API __attribute__((visibility("default")))
#endif

typedef struct sbdv_model sbdv_model;

SBDV_API const char* sbdv_version(void);
SBDV_API sbdv_model* sbdv_load(const char* path, char* err, int errlen);   /* NULL on error, message in err */
SBDV_API void sbdv_free(sbdv_model*);
SBDV_API int sbdv_input_kind(const sbdv_model*);                           /* 0 = per-frame probabilities, 1 = list of cuts */
/* "tau", "tau_rescue", "base_thr", "low_thr", "params", "trees", "nodes", "nets", "bytes", "rescue", "dedupe", "snap", "streamable",
 * and about the last sbdv_verify / sbdv_features call: "last_invalid" (cuts pointing outside the video), "last_bad" (cuts with a < 0 or b <= a) */
SBDV_API double sbdv_get(const sbdv_model*, const char* what);
SBDV_API void sbdv_set_tau(sbdv_model*, double tau, double tau_rescue);
/* options: "lenient" = 1: cuts pointing outside the video are skipped (default 0: sbdv_verify returns -1 with a message) */
SBDV_API int sbdv_set_option(sbdv_model*, const char* name, double value);
SBDV_API double sbdv_flops(const sbdv_model*, int T, int ncand);           /* arithmetic operations, ESTIMATED by counting the loops */

/* Run the verifier.
 *   frames  uint8 [T][27][48][3]
 *   probs   float32 [nprobs]            (kind 0 models; NULL otherwise)
 *   cuts    int32 [ncuts][2]            (kind 1 models: a = last frame before the cut, b = first frame after); optional float32 scores[ncuts]
 *   ranges  int32 [nranges][2] half-open with labels = "intra\tinter\n" per range (OmniShotCut), pass cuts=NULL then
 * outputs (caller allocates, cap = capacity of the candidate arrays):
 *   out_ab int32 [cap][2], out_p double [cap] (-1 for a skipped candidate), out_kept uint8 [cap], out_src int32 [cap]
 *   out_scenes int32 [cap_sc][2]  inclusive [start,end] of the final shots, *n_scenes their count
 *   timing_us double[6] = candidates, signals, features, models, decisions, total
 * returns the number of candidates (can exceed cap: then only cap are written), or -1 on error (message in err).
 */
SBDV_API int sbdv_verify(sbdv_model* m, const uint8_t* frames, int T, const float* probs, int nprobs, const int32_t* cuts, const float* scores, int ncuts,
                         const int32_t* ranges, int nranges, const char* labels, int32_t* out_ab, double* out_p, uint8_t* out_kept, int32_t* out_src, int cap,
                         int32_t* out_scenes, int cap_sc, int* n_scenes, double* timing_us, char* err, int errlen);

/* Feature table for training your own verifier: same candidates as sbdv_verify, but every feature is computed (no model needed
 * except for the candidate rules).  out_feat float32 [cap][sbdv_n_features()].  Candidates outside the video are left out. */
SBDV_API int sbdv_n_features(void);
SBDV_API const char* sbdv_feature_name(int i);
SBDV_API int sbdv_features(sbdv_model* m, const uint8_t* frames, int T, const float* probs, int nprobs, const int32_t* cuts, const float* scores, int ncuts,
                           const int32_t* ranges, int nranges, const char* labels, int32_t* out_ab, int32_t* out_src, float* out_feat, int cap, char* err, int errlen);

#ifdef __cplusplus
}
#endif
#endif
