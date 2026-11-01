# BSR-VideoMAE

This is the official implementation of the ICML 2026 paper: Harnessing Spectrum Video for Subject-Level Few-Shot and Cross-Montage EEG Generalization.

Considering the original 2-stage version is hard for evaluation and benchmarking, we implement the integrated version BSR-VideoMAE-AS, which could directly input eeg and output task classifications.

For reproducing the original version in the paper, refer to BSR-VideoMAE/bsr-videomae-2stage.


## Updates in 26.11.01

1. BSR-VideoMAE-AS, which could be use just like other eeg models and do not use mp4 files as intermediate data. Besides, Uniform Temporal Subsampling used in the old implementation is harmful to BCI decoding tasks due to inappropriate frame sampling. We fix this defect using adaptive sampling which make the model make full use of the eeg data.
2. Preprocessing segment length of SEED and SEED-VII in the formal version is 1500(in a outdated version) instead of 2000(in the paper). In this version we correct this mistake.
3. In code review we found an inappropriate data loading in LaBraM few-shot implementation. q95 (used by BIOT and other baselines) is used in seed, seedvii and bciciv-2a instead of direct uv input. 

The updated subject-level few-shot learning results (b-acc) of the BSR-VideoMAE-AS and LaBraM (q95 and uv):

| method | shumi | bcic | seed | seed-vii |
| :----- | :------: | :------: | :------: | :------: |
| LaBraM(q95) |0.5365|0.2870|0.5203|0.1587|
| LaBraM(uv) |0.5314|0.3870|0.5187|0.2483|
| BSR-VideoMAE-AS |0.5447|0.6003|0.6008|0.2595|

## BSR-VideoMAE-AS

Integrated EEG-to-classification version (no mp4 intermediate). Code: [`bsr-videomae-as/`](bsr-videomae-as/).
Released backbone: <https://huggingface.co/SII-WANGWei/BSR-VideoMAE-AS>
(`S2B2A_c70ceb7e.pt`, sha256 `c70ceb7ecc6b42b0e10be5ce654e2442babe1fddb2db66c93c8a4f1b7d80b65b`).

**Adaptive temporal sampling.** Instead of fixed-fps subsampling, the renderer
computes the STFT hop per trial as `hop = stft_hop_for_frames(L, 16)`, the largest
hop for which `torch.stft(center=True)` yields exactly 16 frames
(`1 + L // hop == 16`; e.g. hop 133 for L=2000 and hop 53 for L=800); the 16
spectral frames therefore always cover the complete 800/2000-sample EEG window
rather than a sub-sampled slice.

### Data preparation, protocol, and running

See [`bsr-videomae-as/README.md`](bsr-videomae-as/README.md) for the full,
publicly reproducible workflow: environment variables for preprocessing, split
manifest / lock generation, the four benchmark commands, and the per-benchmark
protocol details.

## ToDo.

We are exploring more approaches using the spatialization strategies for building more powerful eeg foundation models ...


## Credits

- **BSR-VideoMAE-AS design and project planning**: project author.
- **BSR-VideoMAE-AS implementation** (code, experiments, engineering and release): DeepSeek Harness.

