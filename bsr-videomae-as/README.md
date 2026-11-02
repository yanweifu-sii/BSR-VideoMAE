# BSR-VideoMAE-AS

The released backbone is distributed separately on Hugging Face:
<https://huggingface.co/SII-WANGWei/BSR-VideoMAE-AS>
(`S2B2A_c70ceb7e.pt`, sha256 `c70ceb7ecc6b42b0e10be5ce654e2442babe1fddb2db66c93c8a4f1d80b65b`).
Download it to `ckpts/` or pass `CKPT=<path>` to the scripts.

Pretraining data: a mixture of public EEG corpora from NeuralBench/MOABB/BCI collections — TUAB, TUAR, TUeV, DICE, HBN (R1), Kappenman ERPs (ERN/LRP/MMN/N170/N2PC/N400), Chen2023, Dan2023, Ghassemi2018, Hinss2023, Kemp2000, Levy2026, Liu2024, Mumtaz2018, Nieuwland2018, Schreuder2010, Singh2021, Srisrisawang2024, Thielen2021, Zyma2019, Lee2019 (MI/ERP/SSVEP), BNCI2014-002/004, BNCI2015-001, PhysioNet-MI, Wang2017, Stieger2021, EEGMAT, Kaggle-BCI, ERP-BCI, SEED-IV/DV/VIG/SD, FACED, DREAMER, Yang2025, Ma2020, Weibo2014, Jeong2020, Cho2017.

## 1. Environment

```bash
pip install -r requirements.txt        # at the repository root
```

## 2. Data

Preprocess each benchmark with the scripts in `../preprocess/` (roots are
controlled by environment variables, so no code edits are needed):

```bash
export BSR_DATA_ROOT=/path/to/eeg          # outputs: $BSR_DATA_ROOT/<corpus>_fewshot/subjectN/{train,val|validation,test}
export BCIC2A_RAW=/path/to/bcic-iv/2a
export SEED_RAW=/path/to/SEED_EEG/Preprocessed_EEG
export SEEDVII_RAW=/path/to/SEED-VII/EEG_preprocessed
export SHUMI_RAW=/path/to/shumi

python ../preprocess/process_bcic2a.py
python ../preprocess/process_seed_fewshot.py
python ../preprocess/process_seedvii_fewshot.py
python ../preprocess/process_shumi_fewshot.py
```

Expected protocol (raw directories are authoritative):

| bench | train / val / test | stage | selection | EA |
|---|---|---|---|---|
| bcic2a | 288 / – / 288 | fixed | N=19, last | inductive, λ=1e-5·tr/C² |
| seedvii | 698 / – / 669 | fixed | N=8, last | – |
| seed | 308 / 308 / 308 | strict | max 16, best/p5 | transductive, λ=1e-3·tr/C |
| shumi | ses-01/02 / ses-03 / ses-04/05 | strict | max 25, best/p15 | – |

Generate the split manifests and the local lock from *your* data copy:

```bash
python scripts/make_splits.py --data-root "$BSR_DATA_ROOT"
python scripts/make_lock.py --ckpt ckpts/S2B2A_c70ceb7e.pt
```

`make_splits.py` writes `manifests/splits_dirs/<bench>/subjectN.json` with
absolute paths + row sha256; `make_lock.py` fills `manifests/final_v3_lock.json`
with the checkpoint hash and the per-subject manifest hashes, so `finetune.py`
can fail-closed if the data/checkpoint differ from the preregistered protocol.

## 3. Run the four benchmarks

```bash
export CKPT=$PWD/ckpts/S2B2A_c70ceb7e.pt
bash scripts/finetune_bcic2a.sh    # fixed, N19, last
bash scripts/finetune_seedvii.sh   # fixed, N8, last
bash scripts/finetune_seed.sh      # strict, max 16, patience 5
bash scripts/finetune_shumi.sh     # strict, max 25, patience 15
```

Outputs (`summary.json`, per-subject `final.pt`, predictions/labels) go to
`outputs/<bench>/`. `--stage fixed` never opens validation and selects the last
epoch; `--stage strict` uses the official validation split only for early
stopping/selection and opens the test split exactly once.

## 4. Reference results (released backbone)

| bench | b-acc |
|---|---:|
| bcic2a | 0.6003 |
| seed | 0.6008 |
| seedvii | 0.2595 |
| shumi | 0.5447 |

## 5. Notes

* The checkpoint is the compact BSR-VideoMAE-AS format; load it with
  `torch.load(..., map_location="cpu", weights_only=False)`.
* Paths in `manifests/splits_dirs` are local by construction; never commit them
  with absolute paths.
