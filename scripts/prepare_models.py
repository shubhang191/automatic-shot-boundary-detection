"""Fetch pinned official Omni weights; import only the checksum-matched AutoShot file."""
from pathlib import Path
import argparse,hashlib,json,shutil,urllib.request
ROOT=Path(__file__).resolve().parents[1]
MODELS={
 'preview':('models/omnishotcut/OmniShotCut_ckpt.pth','5948ea78e00626c0e6c5e742e64873ef872cf4a5071d2a0841aed51c3e686cfa','https://huggingface.co/uva-cv-lab/OmniShotCut/resolve/7b59cc525168ea13d36dcf08db6885fd7edc2215/OmniShotCut_ckpt.pth'),
 'v15':('models/omnishotcut_legacy_v15/OmniShotCut_ckpt.pth','5bafe33d8d0dde16c6e24fc63dfc580da9d5731804d6e51e1fc8de21d4600999','https://huggingface.co/uva-cv-lab/OmniShotCut_v1.5/resolve/1bc678977ad48c65d57d600296d95f97c1be293f/OmniShotCut_ckpt.pth')}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def prepare(autoshot=None):
 for name,(rel,digest,url) in MODELS.items():
  dest=ROOT/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists():
   if sha(dest)!=digest:raise RuntimeError('Existing model has incorrect SHA; refusing overwrite: '+str(dest))
   print(name,'already verified',flush=True);continue
  tmp=dest.with_suffix('.download');print('Downloading official',name,flush=True)
  with urllib.request.urlopen(url,timeout=180) as response,tmp.open('wb') as f:shutil.copyfileobj(response,f)
  if sha(tmp)!=digest:raise RuntimeError('Downloaded SHA mismatch: '+name)
  tmp.rename(dest);print(name,'verified',flush=True)
 dest=ROOT/'models/autoshot/ckpt_0_200_0.pth';digest='3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6'
 if autoshot:
  source=Path(autoshot).resolve()
  if sha(source)!=digest:raise RuntimeError('AutoShot file is not the supported original checkpoint')
  dest.parent.mkdir(parents=True,exist_ok=True)
  if dest.exists() and sha(dest)!=digest:raise RuntimeError('Refusing overwrite of different AutoShot model')
  if source!=dest.resolve() and not dest.exists():shutil.copy2(source,dest)
 if not dest.exists() or sha(dest)!=digest:raise RuntimeError('AutoShot required: obtain ckpt_0_200_0.pth from authors and rerun with --autoshot downloads/ckpt_0_200_0.pth; see docs/MODELS.md')
 print('All pretrained weights verified. No fitting or model conversion.',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--autoshot',type=Path);prepare(p.parse_args().autoshot)
