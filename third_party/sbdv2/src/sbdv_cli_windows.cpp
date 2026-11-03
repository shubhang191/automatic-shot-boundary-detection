// Windows transport for the UNCHANGED V40 run_pipeline/Streamer algorithms.
// Input is RGB uint8 [T,27,48,3]. Python owns FFmpeg decoding using argument arrays.
#include "sbdv.hpp"
#include <iostream>
#include <iomanip>
#ifdef _WIN32
#include <windows.h>
#include <shellapi.h>
#endif
using namespace sbdv;

static int execute(const std::vector<std::string>& args) {
  try {
    if(args.size()<2) throw std::runtime_error("Usage: sbdv info MODEL | verify --model MODEL --frames RAW --cuts CUTS [--ram|--stream] [--out JSON]");
    Model m; std::string err;
    if(args[1]=="--version") { std::cout << SBDV_VERSION << "\n"; return 0; }
    if(args[1]=="info") {
      if(args.size()!=3 || !m.load(args[2],err)) throw std::runtime_error(err.empty()?"Expected model":err);
      std::cout << "{\"version\":\"" << SBDV_VERSION << "\",\"trees\":" << m.n_trees()
        << ",\"nodes\":" << m.n_nodes() << ",\"bytes\":" << m.file_bytes << ",\"tau\":" << m.tau
        << ",\"streamable\":" << (m.streamable()?"true":"false") << "}\n"; return 0;
    }
    if(args[1]!="verify") throw std::runtime_error("Unknown command");
    std::string model,frames,cuts,out; bool stream=false;
    for(size_t i=2;i<args.size();++i) {
      const auto& a=args[i];
      if(a=="--stream") {stream=true;continue;}
      if(a=="--ram") {stream=false;continue;}
      if(i+1>=args.size()) throw std::runtime_error("Missing option value");
      const auto v=args[++i];
      if(a=="--model") model=v; else if(a=="--frames") frames=v; else if(a=="--cuts") cuts=v; else if(a=="--out") out=v;
      else throw std::runtime_error("Unsupported Windows option: "+a+". Use scripts/verify_video_cli.py for video input.");
    }
    if(model.empty()||frames.empty()||cuts.empty()) throw std::runtime_error("Supply --model, --frames and --cuts");
    if(!m.load(model,err)) throw std::runtime_error(err);
    Input input; std::ifstream cf(std::filesystem::u8path(cuts));
    if(!cf) throw std::runtime_error("Cannot read cuts");
    std::string line;
    while(std::getline(cf,line)) {
      if(line.empty())continue;
      std::istringstream ss(line); int a,b; std::string extra;
      if(!(ss>>a>>b)||(ss>>extra)||a<0||b<=a) throw std::runtime_error("Invalid cuts row; expected zero-based integer A B");
      input.cuts.push_back({a,b});
    }
    std::ifstream raw(std::filesystem::u8path(frames),std::ios::binary);
    if(!raw) throw std::runtime_error("Cannot read frames");
    Output result;
    if(stream) {
      Streamer engine; if(!engine.init(m,input,false,err))throw std::runtime_error(err);
      while(true) {
        raw.read(reinterpret_cast<char*>(engine.slot()),FSZ); auto n=raw.gcount();
        if(n==0)break;
        if(n!=FSZ)throw std::runtime_error("Partial RGB frame");
        engine.commit();
      }
      if(!engine.finish(result,err))throw std::runtime_error(err);
    } else {
      std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(raw)),{});
      if(bytes.size()%FSZ || bytes.size()/FSZ>18000)throw std::runtime_error("Partial or excessive frames");
      if(!run_pipeline(m,bytes.data(),static_cast<int>(bytes.size()/FSZ),input,result,err))throw std::runtime_error(err);
    }
    if(result.n_invalid||result.n_bad) {std::cerr<<"Cut indices outside video or invalid\n";return 3;}
    std::ostringstream json; json << std::setprecision(17) << "{\"mode\":\"" << (stream?"stream":"ram") << "\",\"frames\":" << result.T << ",\"scenes\":[";
    for(size_t i=0;i<result.scenes.size();++i)json<<(i?",":"")<<"["<<result.scenes[i][0]<<","<<result.scenes[i][1]<<"]";
    json<<"],\"candidates\":[";
    for(size_t i=0;i<result.cands.size();++i)json<<(i?",":"")<<"{\"a\":"<<result.cands[i].a<<",\"b\":"<<result.cands[i].b
      <<",\"p_keep\":"<<result.p_keep[i]<<",\"kept\":"<<(result.kept[i]?"true":"false")<<"}";
    json<<"]}\n";
    if(out.empty())std::cout<<json.str(); else {std::ofstream f(std::filesystem::u8path(out));if(!f)throw std::runtime_error("Cannot write JSON");f<<json.str();}
    return 0;
  } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
int main(int argc,char** argv) {
  std::vector<std::string> args;
#ifdef _WIN32
  int n=0; LPWSTR* wide=CommandLineToArgvW(GetCommandLineW(),&n);
  for(int i=0;i<n;++i) {
    int sz=WideCharToMultiByte(CP_UTF8,0,wide[i],-1,nullptr,0,nullptr,nullptr);
    std::string text(sz,'\0'); WideCharToMultiByte(CP_UTF8,0,wide[i],-1,text.data(),sz,nullptr,nullptr);
    text.pop_back();args.push_back(text);
  }
  LocalFree(wide);
#else
  for(int i=0;i<argc;++i)args.push_back(argv[i]);
#endif
  return execute(args);
}
