# Historical research results (not a new packaging benchmark)

200 videos; 2,516 annotated transition events; official two-frame-tolerance interval protocol. Micro precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Counts transcribed from the locked academic evidence ledger and final reviewed report. The eight Raw/C++ rows retain the cached summary; three fitted Python rows are the reported apparent evaluations. All eleven configurations happen to be production-selectable in this release; that does not establish identity between historical cached predictions and current live inference.

| Configuration | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| AutoShot_CPP | 2083 | 208 | 433 | 0.9092 | 0.8279 | 0.8667 |
| TinyCut_Clean_CPP | 2078 | 252 | 438 | 0.8918 | 0.8259 | 0.8576 |
| AutoShot_Raw_CACHE | 2098 | 377 | 418 | 0.8477 | 0.8339 | 0.8407 |
| Omni_Preview_CPP | 1993 | 236 | 523 | 0.8941 | 0.7921 | 0.8400 |
| TinyCut_Clean_Raw | 2097 | 424 | 419 | 0.8318 | 0.8335 | 0.8326 |
| Omni_v15_CPP | 1939 | 221 | 577 | 0.8977 | 0.7707 | 0.8293 |
| Omni_Preview_Raw_CACHE | 2018 | 613 | 498 | 0.7670 | 0.8021 | 0.7841 |
| Omni_v15_Raw_CACHE | 1960 | 592 | 556 | 0.7680 | 0.7790 | 0.7735 |
| AutoShot_Python_apparent | 2199 | 232 | 317 | 0.9046 | 0.8740 | 0.8890 |
| Omni_Preview_archive_Python_apparent | 2167 | 256 | 349 | 0.8943 | 0.8613 | 0.8775 |
| Omni_v15_Python_apparent | 2173 | 219 | 343 | 0.9084 | 0.8637 | 0.8855 |

TinyCut here is the clean deployed single-member S model (18,433 parameters), not the older five-member ensemble. C++ median classifier-stage times in the accompanying CSV exclude decoding, model inference, UI and export; they are not end-to-end latency.

SHOT200 was reused during model selection and development. The fitted Python scores may be in-sample; source-disjoint independent evaluation was not completed. Neither apparent Python gains nor software tests establish unseen-source superiority. The current Preview weights have verified official identity, but the historical Preview prediction JSON has no creation-time checkpoint SHA: its lineage to the current Preview file remains unresolved. The archived count audit recorded 189 fully agreeing and 11 discrepant rows among 200, plus distinct terminal omissions; this repository does not repair those predictions. Duplicate content was also identified (199 unique files in 200), so no independent-sample inference is claimed. This project does not claim state-of-the-art performance. No SHOT/BBC/ad media or annotations are redistributed.
