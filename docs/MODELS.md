# Models, rights and exact provisioning

AutoShot is pretrained work by Wentao Zhu and colleagues, not our architecture. Source is MIT (third_party/autoshot/LICENSE). The off-repository checkpoint's separate redistribution terms have not been established, so this repository does not rehost it. Obtain **ckpt_0_200_0.pth** using the model link in the [authors README](https://github.com/wentaozhu/AutoShot): [official Baidu share](https://pan.baidu.com/s/1CdCVNzFdF3U6I4ajfejYNQ?pwd=sfkq), extraction code `sfkq`. Access may require a Baidu account/client. No substitute or fabricated direct download is offered. Place your lawfully obtained file in downloads/ and run prepare_models.py with --autoshot. Required SHA: `3e85290546ce6d32f4a3581ec2cae87aedd2402246a0d46b4d361a330b4b1fa6`. A different model is refused. Acquisition through that third-party login flow is not automated.

OmniShotCut is pretrained work by Boyang Wang, Guangyi Xu, Jiahui Zhang, Zhipeng Tang and Zezhou Cheng (UVA). Its official models are MIT; notices/cards are retained. prepare_models.py downloads official immutable revisions, never Git mirrors or our rehosted weights:

| ID | Official repository/revision | Parameters | SHA-256 |
|---|---|---:|---|
| omnishotcut | uva-cv-lab/OmniShotCut @ 7b59cc525168ea13d36dcf08db6885fd7edc2215 | 40,990,775 | 5948ea78e00626c0e6c5e742e64873ef872cf4a5071d2a0841aed51c3e686cfa |
| omnishotcut_legacy_v15 | uva-cv-lab/OmniShotCut_v1.5 @ 1bc678977ad48c65d57d600296d95f97c1be293f | 52,772,087 | 5bafe33d8d0dde16c6e24fc63dfc580da9d5731804d6e51e1fc8de21d4600999 |

Preview is the earlier six-encoder/hidden-384 model; v1.5 has three encoders/hidden-576. Both have six decoders, ResNet18, 128x96 model pictures, window 100, overlap 10. Current identity is proven by official file SHA, strict state loading and architecture; old historical JSON provenance is a separate question.

Project-origin bundled artifacts: clean TinyCut S single member (18,433 parameters, 31 selected frame-derived features, epoch 10), frozen decoder, AutoShot 43-feature fitted verifier and Omni 40-feature fitted verifier. Both original joblib bundles are unchanged. Omni uses native/rescue thresholds 0.35/0.50, rescue ratio 4, minimum 0.05 and snap window 3; AutoShot uses 0.40 with base/low proposals 0.296/0.10. No retraining occurs at setup. Only trusted fixed checksummed bundles are deserialized; never load arbitrary pickle/joblib uploads.

C++ generic_local.sbdm is the frozen project classifier: 60 trees, 30 local features, threshold 0.4. Bundled Windows DLL/EXE and source are retained. The project has not selected a blanket open-source license: public visibility alone grants no general reuse license for project-owned code/models. Third-party files retain their original licenses; see THIRD_PARTY_NOTICES.md.
