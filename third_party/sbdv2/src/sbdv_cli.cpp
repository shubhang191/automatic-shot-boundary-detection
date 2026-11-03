// sbdv - command line tool of the SBD-Verifier (version 2).
//   sbdv info   MODEL.sbdm
//   sbdv verify --model MODEL.sbdm --video clip.mp4 --scenes shots.txt --out result.json
// Run `sbdv` without arguments for the full help.  Linux / macOS (uses fork, pipe, mmap).
#include "sbdv.hpp"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>

using namespace sbdv;

static const char* HELP =
    "sbdv " SBDV_VERSION " - learned cut verifier that sits on top of any shot-boundary detector\n"
    "\n"
    "  sbdv info   MODEL.sbdm\n"
    "  sbdv verify --model MODEL.sbdm  (--video FILE | --frames RAW.u8 | --npy FRAMES.npy)  <detector output>  [options]\n"
    "  sbdv --version\n"
    "\n"
    "pictures:  --video FILE   any video; decoded with ffmpeg to 48x27 RGB (frame i = i-th decoded frame, no frame duplication/dropping)\n"
    "           --frames FILE  raw uint8 [T][27][48][3]            --npy FILE  a numpy file of that array\n"
    "detector output (what YOUR detector found; frame numbers must refer to the frames of THIS video):\n"
    "  --scenes FILE   one shot per line: START END      (default: 0-based frame numbers, END = last frame of the shot, inclusive)\n"
    "  --cuts FILE     one cut per line:  A B [score]    (A = last frame of the old shot, B = first frame of the new shot)\n"
    "  --times FILE    one cut per line:  SECONDS [score] (time of the first frame of the new shot; needs the frame rate)\n"
    "  --probs FILE    one boundary probability per frame (.npy float32/float64, or a text file with one number per line)\n"
    "  --ranges FILE   OmniShotCut style: START END INTRA INTER per line (half-open ranges + labels, '-' for none)\n"
    "  how to read the numbers in --scenes / --cuts:\n"
    "  --base 0|1      are frame numbers counted from 0 (default) or from 1?\n"
    "  --end inclusive|exclusive   is END (in --scenes) the last frame of the shot (default) or the first frame of the next shot?\n"
    "  --seconds       the numbers in --scenes are times in seconds (START = first frame, END = time where the shot stops)\n"
    "  --fps X         frame rate for --seconds / --times (default: read from the video with ffprobe)\n"
    "output:\n"
    "  --out FILE      result as JSON (default: print to the screen)\n"
    "  --shots FILE    the cleaned shot list, one \"START END\" per line (0-based, inclusive)\n"
    "  --cuts-out FILE every candidate: A B p_keep kept(0/1)\n"
    "  --quiet         print only the final shots\n"
    "options:\n"
    "  --tau X         keep threshold (default: the one stored in the model)\n"
    "  --lenient       cuts that point outside the video are skipped with a warning instead of stopping with an error\n"
    "  --ram / --stream  force RAM mode / constant-memory stream mode (default: stream for --video when the model allows it)\n"
    "  --cfr FPS       decode at a constant frame rate (use when the detector worked on a constant-frame-rate copy)\n"
    "  --repeat N      RAM mode: run the verifier N times and report the median time (for benchmarks)\n"
    "exit codes: 0 ok, 1 error, 2 wrong usage, 3 the detector output does not fit this video\n";

// ------------------------------------------------------------------------------------------------ files
struct Mapped {
  const uint8_t* p = nullptr; size_t n = 0; int fd = -1;
  bool open_ro(const std::string& path) {
    fd = ::open(path.c_str(), O_RDONLY); if (fd < 0) return false;
    struct stat st; if (fstat(fd, &st) != 0) return false; n = (size_t)st.st_size;
    if (n == 0) { p = (const uint8_t*)""; return true; }
    void* m = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0); if (m == MAP_FAILED) return false; p = (const uint8_t*)m; return true;
  }
  ~Mapped() { if (p && n) munmap((void*)p, n); if (fd >= 0) ::close(fd); }
};
struct Npy { std::string descr; std::vector<long> shape; size_t off = 0; bool ok = false; };
static Npy parse_npy(const uint8_t* b, size_t nb) {
  Npy r; if (nb < 10 || std::memcmp(b, "\x93NUMPY", 6) != 0) return r;
  int major = b[6]; size_t hl, hs; if (major == 1) { hl = b[8] | (b[9] << 8); hs = 10; } else { if (nb < 12) return r; hl = b[8] | (b[9] << 8) | (b[10] << 16) | ((size_t)b[11] << 24); hs = 12; }
  if (hs + hl > nb) return r;
  std::string h((const char*)b + hs, hl);
  size_t p = h.find("'descr'"); if (p == std::string::npos) return r;
  size_t q1 = h.find('\'', h.find(':', p) + 1), q2 = h.find('\'', q1 + 1); r.descr = h.substr(q1 + 1, q2 - q1 - 1);
  if (h.find("'fortran_order': True") != std::string::npos) return r;
  size_t sp = h.find("'shape'"); size_t l = h.find('(', sp), rr = h.find(')', l); if (sp == std::string::npos || l == std::string::npos || rr == std::string::npos) return r;
  std::string s = h.substr(l + 1, rr - l - 1);
  std::stringstream ss(s); std::string tok; while (std::getline(ss, tok, ',')) { if (tok.find_first_of("0123456789") != std::string::npos) r.shape.push_back(atol(tok.c_str())); }
  r.off = hs + hl; r.ok = true; return r;
}
static bool load_probs(const std::string& path, std::vector<float>& out, std::string& err) {
  Mapped mf; if (!mf.open_ro(path)) { err = "cannot read " + path; return false; }
  Npy np = parse_npy(mf.p, mf.n);
  if (np.ok) {
    long n = 1; for (long s : np.shape) n *= s;
    size_t es = (np.descr == "<f4" || np.descr == "=f4") ? 4 : ((np.descr == "<f8" || np.descr == "=f8") ? 8 : 0);
    if (!es) { err = "probs .npy must be float32 or float64"; return false; }
    if (n < 0 || np.off + (size_t)n * es > mf.n) { err = "probs .npy is truncated"; return false; }
    out.resize((size_t)n);
    if (es == 4) std::memcpy(out.data(), mf.p + np.off, (size_t)n * 4);
    else { const double* d = (const double*)(mf.p + np.off); for (long i = 0; i < n; i++) out[i] = (float)d[i]; }
    return true;
  }
  std::stringstream ss(std::string((const char*)mf.p, mf.n)); double v; out.clear();
  while (ss >> v) out.push_back((float)v);
  if (out.empty()) { err = "no numbers found in " + path; return false; }
  return true;
}
// numeric rows of a text file; '#' starts a comment; commas / semicolons count as spaces.  A header (a line that STARTS with text) is allowed only
// as the first line of the file; text anywhere else, or text after numbers on one line (e.g. a CSV with time codes), is an error, so that a file in
// the wrong format can never be read as something else by accident.
static bool read_rows(const std::string& path, std::vector<std::vector<double>>& rows, std::string& err) {
  std::ifstream f(path); if (!f) { err = "cannot read " + path; return false; } std::string ln; rows.clear(); int lineno = 0; bool seen_content = false;
  while (std::getline(f, ln)) {
    lineno++;
    size_t h = ln.find('#'); if (h != std::string::npos) ln = ln.substr(0, h);
    for (char& ch : ln) if (ch == ',' || ch == ';' || ch == '\t') ch = ' ';
    std::stringstream ss(ln); std::string t; std::vector<double> r; bool header = false;
    while (ss >> t) {
      char* e = nullptr; double v = std::strtod(t.c_str(), &e);
      if (e == t.c_str() || *e != 0) {
        if (r.empty() && !seen_content) { header = true; break; }
        if (r.empty()) { err = path + ", line " + std::to_string(lineno) + ": found text \"" + t + "\" where numbers are expected (a text header is only allowed as the first line)"; return false; }
        err = path + ", line " + std::to_string(lineno) + ": found text \"" + t + "\" after numbers; this file is not a plain list of numbers (see the README: sbdv_io.py converts other formats)"; return false;
      }
      r.push_back(v);
    }
    if (header || !r.empty()) seen_content = true;
    if (!header && !r.empty()) rows.push_back(r);
  }
  return true;
}

// ------------------------------------------------------------------------------------------------ ffmpeg / ffprobe without a shell
static bool run_capture(const std::vector<std::string>& args, std::string& out, int* rc = nullptr) {
  int fds[2]; if (pipe(fds) != 0) return false;
  pid_t pid = fork(); if (pid < 0) return false;
  if (pid == 0) {
    dup2(fds[1], 1); close(fds[0]); close(fds[1]); int dn = ::open("/dev/null", O_WRONLY); if (dn >= 0) { dup2(dn, 2); close(dn); }
    std::vector<char*> av; for (auto& s : args) av.push_back((char*)s.c_str()); av.push_back(nullptr);
    execvp(av[0], av.data()); _exit(127);
  }
  close(fds[1]); out.clear(); char buf[4096]; ssize_t n;
  while ((n = read(fds[0], buf, sizeof buf)) > 0) out.append(buf, (size_t)n);
  close(fds[0]); int st = 0; waitpid(pid, &st, 0);
  if (rc) *rc = WIFEXITED(st) ? WEXITSTATUS(st) : -1;
  return true;
}
static double parse_rate(const std::string& s) {      // "30000/1001" or "25"
  size_t sl = s.find('/'); if (sl == std::string::npos) return atof(s.c_str());
  double a = atof(s.substr(0, sl).c_str()), b = atof(s.c_str() + sl + 1); return b > 0 ? a / b : 0;
}
struct Probe { double fps = 0; long nb_frames = 0; };
static Probe probe_video(const std::string& path) {
  Probe p; std::string o;
  run_capture({"ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,r_frame_rate,nb_frames", "-of", "default=noprint_wrappers=1", path}, o);
  std::stringstream ss(o); std::string ln; double avg = 0, rr = 0;
  while (std::getline(ss, ln)) {
    size_t e = ln.find('='); if (e == std::string::npos) continue; std::string k = ln.substr(0, e), v = ln.substr(e + 1);
    if (k == "avg_frame_rate") avg = parse_rate(v); else if (k == "r_frame_rate") rr = parse_rate(v); else if (k == "nb_frames") p.nb_frames = atol(v.c_str());
  }
  p.fps = avg > 0 ? avg : rr; return p;
}
struct Decoder {
  pid_t pid = -1; int fd = -1; struct rusage ru; bool reaped = false; double wait_us = 0;
  std::vector<uint8_t> first; bool have_first = false;     // the first frame, read while checking that ffmpeg accepted the options
  bool spawn(const std::string& video, double cfr, bool new_flag, std::string& err) {
    std::vector<std::string> a = {"ffmpeg", "-nostdin", "-v", "error", "-i", video};
    if (cfr > 0) { a.push_back("-vf"); a.push_back("fps=" + std::to_string(cfr)); }
    a.push_back("-an"); a.push_back("-sn"); a.push_back("-dn");
    if (cfr <= 0) { if (new_flag) { a.push_back("-fps_mode"); a.push_back("passthrough"); } else { a.push_back("-vsync"); a.push_back("0"); } }
    for (const char* s : {"-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "48x27", "-"}) a.push_back(s);
    int fds[2]; if (pipe(fds) != 0) { err = "pipe failed"; return false; }
#ifdef F_SETPIPE_SZ
    fcntl(fds[1], F_SETPIPE_SZ, 1 << 20);
#endif
    pid = fork(); if (pid < 0) { err = "fork failed"; return false; }
    if (pid == 0) {
      dup2(fds[1], 1); close(fds[0]); close(fds[1]);
      std::vector<char*> av; for (auto& s : a) av.push_back((char*)s.c_str()); av.push_back(nullptr);
      execvp(av[0], av.data()); std::fprintf(stderr, "cannot start ffmpeg (is it installed?)\n"); _exit(127);
    }
    close(fds[1]); fd = fds[0]; reaped = false; return true;
  }
  // Old ffmpeg only knows -vsync, new ffmpeg prefers -fps_mode.  Try -vsync 0 first; if ffmpeg dies before the first frame, try the other.
  bool start(const std::string& video, double cfr, std::string& err) {
    for (int attempt = 0; attempt < 2; attempt++) {
      if (!spawn(video, cfr, attempt == 1, err)) return false;
      first.resize(FSZ); have_first = raw_read(first.data(), FSZ);
      if (have_first || cfr > 0) return true;
      int rc = finish(); if (rc == 127) { err = "cannot start ffmpeg (is it installed?)"; return false; }
      if (attempt == 0 && rc != 0) continue;            // maybe the option was refused: try the other spelling
      return true;                                        // ffmpeg ran fine but produced nothing: report "could not decode" later
    }
    return true;
  }
  bool raw_read(uint8_t* dst, size_t n) {
    size_t got = 0; double t0 = now_us();
    while (got < n) { ssize_t r = read(fd, dst + got, n - got); if (r > 0) got += (size_t)r; else if (r == 0) break; else if (errno != EINTR) break; }
    wait_us += now_us() - t0; return got == n;
  }
  bool read_exact(uint8_t* dst, size_t n) {              // false at the end of the stream (a partial last frame is dropped)
    if (have_first) { have_first = false; std::memcpy(dst, first.data(), n); return true; }
    return fd >= 0 && raw_read(dst, n);
  }
  int finish() {                                          // returns ffmpeg's exit status
    if (fd >= 0) { close(fd); fd = -1; }
    int st = 0; if (pid > 0 && !reaped) { wait4(pid, &st, 0, &ru); reaped = true; } return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
  }
  double peak_rss_mb() const {
#ifdef __APPLE__
    return reaped ? ru.ru_maxrss / (1024.0 * 1024.0) : 0;
#else
    return reaped ? ru.ru_maxrss / 1024.0 : 0;
#endif
  }
};

// ------------------------------------------------------------------------------------------------ output helpers
static std::string jesc(const std::string& s) {
  std::string o; for (char ch : s) { if (ch == '"' || ch == '\\') { o += '\\'; o += ch; } else if ((unsigned char)ch < 0x20) o += ' '; else o += ch; } return o;
}
static double self_peak_rss_mb() {                        // peak resident memory of THIS process (VmHWM; ru_maxrss would be inherited from the parent)
  double mb = 0; std::ifstream st("/proc/self/status"); std::string ln;
  while (std::getline(st, ln)) if (ln.rfind("VmHWM:", 0) == 0) mb = atof(ln.c_str() + 6) / 1024.0;
  if (mb == 0) { struct rusage ru; getrusage(RUSAGE_SELF, &ru);
#ifdef __APPLE__
    mb = ru.ru_maxrss / (1024.0 * 1024.0);
#else
    mb = ru.ru_maxrss / 1024.0;
#endif
  }
  return mb;
}
static const char* SRC_NAME[] = {"base", "low", "detector", "rescue", "user"};

int main(int argc, char** argv) {
  if (argc < 2) { fputs(HELP, stdout); return 0; }
  std::string cmd = argv[1];
  if (cmd == "--version" || cmd == "version") { printf("sbdv %s\n", SBDV_VERSION); return 0; }
  if (cmd == "info") {
    if (argc < 3) { fputs(HELP, stdout); return 2; }
    Model M; std::string err; if (!M.load(argv[2], err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    printf("model          : %s\ninput          : %s\nmembers        : %zu (trees %zu, tree nodes %zu, neural nets %zu)\nparameters     : %zu numbers\nfile size      : %.1f KB\n"
           "keep threshold : %.4f%s\nmemory mode    : %s\n",
           M.name.c_str(), M.input_kind == 0 ? "per-frame probabilities (AutoShot style)" : "list of cuts (any detector)", M.members.size(), M.n_trees(), M.n_nodes(), M.n_tcn(),
           M.n_params(), M.file_bytes / 1024.0, M.tau, M.rescue ? " (+ rescue candidates at picture jumps)" : "",
           M.streamable() ? "stream (constant memory, any video length)" : "RAM (keeps all frames: 3.9 KB per frame)");
    if (M.input_kind == 0) printf("candidates     : probability > %.3f (base) and %.3f..%.3f (borderline)\n", M.base_thr, M.low_thr, M.base_thr);
    if (M.dedupe) printf("near-duplicate : keep only the surer of two cuts within %d frames\n", M.dedupe);
    if (M.snap_w) printf("snap           : hard cuts move to the biggest change within +-%d frames\n", M.snap_w);
    printf("features used  :"); int c = 0; for (int i = 0; i < NF; i++) if (M.need.f[i]) { printf(" %s", FEAT_NAME[i]); c++; } printf("  (%d)\n", c);
    return 0;
  }
  if (cmd != "verify") { fputs(HELP, stdout); return 2; }

  std::string model, video, frames_raw, npy, probs, cuts, scenes, ranges, times, outp, shots_out, cuts_out, endmode = "inclusive";
  double tau = NAN, fps_arg = 0, cfr = 0; int repeat = 1, base = 0; bool quiet = false, lenient = false, seconds = false, force_ram = false, force_stream = false;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i]; auto nx = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "missing value after %s\n", a.c_str()); exit(2); } return argv[++i]; };
    if (a == "--model") model = nx(); else if (a == "--video") video = nx(); else if (a == "--frames") frames_raw = nx(); else if (a == "--npy") npy = nx();
    else if (a == "--probs") probs = nx(); else if (a == "--cuts") cuts = nx(); else if (a == "--scenes") scenes = nx(); else if (a == "--ranges") ranges = nx(); else if (a == "--times") times = nx();
    else if (a == "--out") outp = nx(); else if (a == "--shots") shots_out = nx(); else if (a == "--cuts-out") cuts_out = nx();
    else if (a == "--tau") tau = atof(nx().c_str()); else if (a == "--repeat") repeat = std::max(1, atoi(nx().c_str()));
    else if (a == "--base") base = atoi(nx().c_str()); else if (a == "--end") endmode = nx(); else if (a == "--fps") fps_arg = atof(nx().c_str()); else if (a == "--cfr") cfr = atof(nx().c_str());
    else if (a == "--seconds") seconds = true; else if (a == "--lenient") lenient = true; else if (a == "--ram") force_ram = true; else if (a == "--stream") force_stream = true;
    else if (a == "--quiet") quiet = true; else { fprintf(stderr, "unknown option %s\n\n%s", a.c_str(), HELP); return 2; }
  }
  if ((base != 0 && base != 1) || (endmode != "inclusive" && endmode != "exclusive")) { fprintf(stderr, "error: --base must be 0 or 1, --end must be inclusive or exclusive\n"); return 2; }
  if (force_ram && force_stream) { fprintf(stderr, "error: --ram and --stream exclude each other\n"); return 2; }
  Model M; std::string err;
  if (model.empty() || !M.load(model, err)) { fprintf(stderr, "error: %s\n", model.empty() ? "--model is required" : err.c_str()); return 1; }
  if (!std::isnan(tau)) M.tau = tau;
  int nsrc = (!video.empty()) + (!frames_raw.empty()) + (!npy.empty());
  if (nsrc != 1) { fprintf(stderr, "error: give exactly one of --video, --frames, --npy\n"); return 2; }
  int ndet = (!probs.empty()) + (!cuts.empty()) + (!scenes.empty()) + (!ranges.empty()) + (!times.empty());
  if (ndet != 1) { fprintf(stderr, "error: give exactly one detector output: --scenes, --cuts, --times, --probs or --ranges\n"); return 2; }
  if (M.input_kind == 0 && probs.empty()) { fprintf(stderr, "error: model '%s' needs per-frame probabilities (--probs); for a list of shots/cuts use models/generic.sbdm\n", M.name.c_str()); return 2; }
  if (M.input_kind == 1 && !probs.empty()) { fprintf(stderr, "error: model '%s' works on a list of cuts, not on probabilities\n", M.name.c_str()); return 2; }
  std::vector<std::string> warnings;

  // ---- the detector's output
  double fps = fps_arg;
  if ((seconds || !times.empty()) && fps <= 0) {
    if (video.empty()) { fprintf(stderr, "error: seconds need a frame rate: give --fps (or use --video)\n"); return 2; }
    fps = probe_video(video).fps; if (fps <= 0) { fprintf(stderr, "error: could not read the frame rate of %s: give --fps\n", video.c_str()); return 1; }
  }
  Input in; std::vector<float> pr;
  if (!probs.empty()) { if (!load_probs(probs, pr, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; } in.probs = pr.data(); in.nprobs = (int)pr.size(); }
  std::vector<std::vector<double>> rows;
  if (!cuts.empty()) {
    if (!read_rows(cuts, rows, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    bool with_score = !rows.empty() && rows[0].size() >= 3;
    for (auto& r : rows) {
      if (r.size() < 2) continue;
      in.cuts.push_back({(int)r[0] - base, (int)r[1] - base});
      if (with_score) in.scores.push_back(r.size() >= 3 ? (float)r[2] : 1.f);
    }
  }
  if (!times.empty()) {
    if (!read_rows(times, rows, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    bool with_score = !rows.empty() && rows[0].size() >= 2;
    for (auto& r : rows) {
      int b = (int)std::lround(r[0] * fps); in.cuts.push_back({b - 1, b});
      if (with_score) in.scores.push_back(r.size() >= 2 ? (float)r[1] : 1.f);
    }
  }
  if (!scenes.empty()) {
    if (!read_rows(scenes, rows, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    std::vector<Pair> sc;
    for (auto& r : rows) {
      if (r.size() < 2) continue;
      if (seconds) { int s0 = (int)std::lround(r[0] * fps), e0 = (int)std::lround(r[1] * fps) - 1; sc.push_back({s0, e0}); }
      else sc.push_back({(int)r[0] - base, (int)r[1] - base - (endmode == "exclusive" ? 1 : 0)});
    }
    for (size_t i = 0; i + 1 < sc.size(); i++) in.cuts.push_back({sc[i][1], sc[i + 1][0]});
    if (sc.size() < 2) warnings.push_back("the shot list has fewer than 2 shots: nothing to verify");
  }
  if (!ranges.empty()) {
    std::ifstream f(ranges); if (!f) { fprintf(stderr, "error: cannot read %s\n", ranges.c_str()); return 1; } std::string ln;
    while (std::getline(f, ln)) {
      std::stringstream ss(ln); std::string s0, s1, a, b; if (!(ss >> s0 >> s1)) continue; if (!isdigit((unsigned char)s0[0])) continue; ss >> a >> b; if (a == "-") a = "General"; if (b == "-") b = "";
      in.ranges.push_back({atoi(s0.c_str()), atoi(s1.c_str())}); in.intra.push_back(a.empty() ? "General" : a); in.inter.push_back(b);
    }
  }
  const bool cuts_input = (M.input_kind == 1 && in.ranges.empty());
  if (cuts_input && in.cuts.empty() && warnings.empty()) warnings.push_back("the detector output contains no cuts: nothing to verify");

  // ---- run
  Output res; double wall0 = now_us(), decode_ms = 0, wait_ms = 0, dec_rss = 0; int T = 0; std::vector<double> tot;
  std::string mode; int ffrc = 0;
  const bool want_stream = !video.empty() ? (M.streamable() && !force_ram) : force_stream;
  if (want_stream && !M.streamable()) { fprintf(stderr, "error: this model needs the whole video in memory (old v1-v3/omni models); drop --stream\n"); return 2; }
  if (want_stream) {
    mode = "stream"; Streamer S; if (!S.init(M, in, false, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
    if (!video.empty()) {
      Decoder D; if (!D.start(video, cfr, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }
      while (D.read_exact(S.slot(), FSZ)) S.commit();
      ffrc = D.finish(); wait_ms = D.wait_us / 1000; dec_rss = D.peak_rss_mb();
    } else {
      Mapped mf; const uint8_t* base_p = nullptr; size_t nfr = 0;
      if (!frames_raw.empty()) { if (!mf.open_ro(frames_raw) || mf.n % FSZ) { fprintf(stderr, "error: bad --frames file\n"); return 1; } base_p = mf.p; nfr = mf.n / FSZ; }
      else {
        if (!mf.open_ro(npy)) { fprintf(stderr, "error: cannot read %s\n", npy.c_str()); return 1; }
        Npy np = parse_npy(mf.p, mf.n);
        if (!np.ok || np.descr.find("u1") == std::string::npos || np.shape.size() != 4 || np.shape[1] != FH || np.shape[2] != FW || np.shape[3] != FC) { fprintf(stderr, "error: --npy must be uint8 [T,27,48,3]\n"); return 1; }
        nfr = (size_t)np.shape[0]; if (np.off + nfr * FSZ > mf.n) { fprintf(stderr, "error: --npy is truncated\n"); return 1; }
        base_p = mf.p + np.off;
      }
      for (size_t t = 0; t < nfr; t++) { std::memcpy(S.slot(), base_p + t * FSZ, FSZ); S.commit(); }
    }
    if (!S.finish(res, err)) { fprintf(stderr, "error: %s\n", (video.empty() ? err : err + " (ffmpeg could not decode " + video + ")").c_str()); return 1; }
    T = res.T; tot.push_back(res.t_us[5]);
  } else {
    mode = "ram"; Mapped mf; const uint8_t* fp = nullptr; std::unique_ptr<uint8_t, void (*)(void*)> heap(nullptr, std::free); double t0 = now_us();
    if (!video.empty()) {
      Probe pb = probe_video(video); size_t cap_frames = pb.nb_frames > 0 ? (size_t)pb.nb_frames + 8 : 4096;
      uint8_t* buf = (uint8_t*)std::malloc(cap_frames * FSZ); size_t nfr = 0; Decoder D;
      if (!buf || !D.start(video, cfr, err)) { fprintf(stderr, "error: %s\n", err.empty() ? "out of memory" : err.c_str()); return 1; }
      for (;;) {
        if (nfr == cap_frames) { cap_frames += cap_frames / 2; uint8_t* nb = (uint8_t*)std::realloc(buf, cap_frames * FSZ); if (!nb) { fprintf(stderr, "error: out of memory (video too long for RAM mode)\n"); return 1; } buf = nb; }
        if (!D.read_exact(buf + nfr * FSZ, FSZ)) break;
        nfr++;
      }
      ffrc = D.finish(); wait_ms = D.wait_us / 1000; dec_rss = D.peak_rss_mb();
      heap.reset(buf); fp = buf; T = (int)nfr;                  // the decoded frames are used where they are (no copy)
    } else if (!frames_raw.empty()) {
      if (!mf.open_ro(frames_raw) || mf.n % FSZ) { fprintf(stderr, "error: bad --frames file\n"); return 1; } fp = mf.p; T = (int)(mf.n / FSZ);
    } else {
      if (!mf.open_ro(npy)) { fprintf(stderr, "error: cannot read %s\n", npy.c_str()); return 1; }
      Npy np = parse_npy(mf.p, mf.n);
      if (!np.ok || np.descr.find("u1") == std::string::npos || np.shape.size() != 4 || np.shape[1] != FH || np.shape[2] != FW || np.shape[3] != FC) { fprintf(stderr, "error: --npy must be uint8 [T,27,48,3]\n"); return 1; }
      T = (int)np.shape[0]; if (np.off + (size_t)T * FSZ > mf.n) { fprintf(stderr, "error: --npy is truncated\n"); return 1; } fp = mf.p + np.off;
    }
    decode_ms = (now_us() - t0) / 1000;
    if (T < 2) { fprintf(stderr, "error: %s\n", video.empty() ? "need at least 2 frames" : ("ffmpeg could not decode " + video + " (exit " + std::to_string(ffrc) + ")").c_str()); return 1; }
    for (int r = 0; r < repeat; r++) { if (!run_pipeline(M, fp, T, in, res, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; } tot.push_back(res.t_us[5]); }
  }
  double wall_ms = (now_us() - wall0) / 1000;
  std::sort(tot.begin(), tot.end()); double med = tot[tot.size() / 2];

  // ---- does the detector output fit this video?
  if (!probs.empty() && std::abs(in.nprobs - T) > 0) {
    char b[200]; snprintf(b, sizeof b, "the detector gave %d probabilities but the video has %d frames (different frame rate or another file?)", in.nprobs, T); warnings.push_back(b);
  }
  if (res.n_bad > 0) { char b[200]; snprintf(b, sizeof b, "%d cut(s) in the detector output make no sense (first frame < 0, or B <= A) and were ignored", res.n_bad); warnings.push_back(b); }
  if (res.n_invalid > 0) {
    std::string ex; int shown = 0; for (auto& c : res.cands) if (!c.valid && shown < 3) { ex += (shown ? ", " : "") + std::string("(") + std::to_string(c.a) + "," + std::to_string(c.b) + ")"; shown++; }
    char b[400]; snprintf(b, sizeof b, "%d cut(s) point outside the video (it has %d frames), e.g. %s: wrong video, frame rate or --base?", res.n_invalid, T, ex.c_str());
    if (!lenient) { fprintf(stderr, "error: %s\n       (use --lenient to skip them)\n", b); return 3; }
    warnings.push_back(std::string(b) + " - skipped");
  }
  if (ffrc != 0 && !video.empty()) warnings.push_back("ffmpeg exited with status " + std::to_string(ffrc) + " (the video may be damaged; " + std::to_string(T) + " frames were decoded)");

  if (!shots_out.empty()) { std::ofstream o(shots_out); for (auto& s : res.scenes) o << s[0] << " " << s[1] << "\n"; }
  if (!cuts_out.empty()) { std::ofstream o(cuts_out); o << "# A B p_keep kept\n"; for (size_t i = 0; i < res.cands.size(); i++) o << res.cands[i].a << " " << res.cands[i].b << " " << res.p_keep[i] << " " << (int)res.kept[i] << "\n"; }
  if (quiet) { for (auto& s : res.scenes) printf("%d %d\n", s[0], s[1]); return 0; }
  int nk = 0; for (auto k : res.kept) nk += k;
  std::ostringstream js; js.setf(std::ios::fixed);
  js << "{\n  \"tool\": \"sbdv " << SBDV_VERSION << "\",\n  \"model\": \"" << jesc(M.name) << "\",\n  \"mode\": \"" << mode << "\",\n  \"frames\": " << T
     << ",\n  \"candidates_total\": " << res.cands.size() << ",\n  \"kept\": " << nk << ",\n  \"skipped_invalid\": " << res.n_invalid << ",\n  \"warnings\": [";
  for (size_t i = 0; i < warnings.size(); i++) js << (i ? ", " : "") << "\"" << jesc(warnings[i]) << "\"";
  js << "],\n  \"scenes\": [";
  for (size_t i = 0; i < res.scenes.size(); i++) js << (i ? "," : "") << "[" << res.scenes[i][0] << "," << res.scenes[i][1] << "]";
  js << "],\n  \"candidates\": [\n";
  for (size_t i = 0; i < res.cands.size(); i++) {
    js << "    {\"a\": " << res.cands[i].a << ", \"b\": " << res.cands[i].b << ", \"src\": \"" << SRC_NAME[res.cands[i].src] << "\", \"p_keep\": ";
    js.precision(6); if (res.cands[i].valid) js << res.p_keep[i]; else js << "null";
    js << ", \"kept\": " << (res.kept[i] ? "true" : "false") << "}" << (i + 1 < res.cands.size() ? "," : "") << "\n";
  }
  js.precision(3);
  // verify_median = compute time of the verifier alone (candidates + features + model + decisions).  In stream mode decoding happens in
  // parallel, so the cost the verifier ADDS to a decode-only run is wall_ms - decoder_wait_ms (small), reported separately.
  js << "  ],\n  \"timing_ms\": {\"wall\": " << wall_ms << ", \"decode_or_load\": " << decode_ms << ", \"decoder_wait\": " << wait_ms << ", \"verify_median\": " << med / 1000
     << ", \"per_frame_us\": " << med / std::max(T, 1) << ", \"candidates\": " << res.t_us[0] / 1000 << ", \"features\": " << res.t_us[2] / 1000 << ", \"models\": " << res.t_us[3] / 1000
     << ", \"decisions\": " << res.t_us[4] / 1000 << "},\n  \"peak_rss_mb\": " << self_peak_rss_mb() << ",\n  \"decoder_peak_rss_mb\": " << dec_rss << "\n}\n";
  if (outp.empty()) fputs(js.str().c_str(), stdout);
  else { std::ofstream o(outp); o << js.str(); printf("wrote %s: %d shots, %d of %zu candidates kept, verifier %.3f ms (%s mode)\n", outp.c_str(), (int)res.scenes.size(), nk, res.cands.size(), med / 1000, mode.c_str()); }
  return 0;
}
