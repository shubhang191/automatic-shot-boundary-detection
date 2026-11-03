# Included engineering sample

`inputs/sintel excerpt.mp4`: a six-second montage from the **Sintel trailer**, © copyright Blender Foundation | [durian.blender.org](https://durian.blender.org/), licensed [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/). [Original licensing statement](https://durian.blender.org/sharing/). No endorsement is implied.

Source: [Sintel trailer 480p MP4](https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4). Changes: three excerpts at [12,14), [23,25), [38,40) seconds, concatenated, resampled to 25 fps, resized to 640×360 and H.264 encoded without audio. Sample SHA-256: `8d1ec553892f6b1fc9b562ef07f2a68d35c3d61e57f788b188d7966e4b6809e7`.

This is actual encoded film content used for engineering validation. It has no full ground-truth shot annotations. Expected outputs record observed behavior, not an F1 score. The 150 decoded frames are partitioned by every exported shot list. The deliberate excerpt joins are not the only possible transitions in the footage.
