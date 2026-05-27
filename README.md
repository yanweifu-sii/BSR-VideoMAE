# BSR-VideoMAE

This is the official implementation of the ICML 2026 paper: Harnessing Spectrum Video for Subject-Level Few-Shot and Cross-Montage EEG Generalization

## Env Setting
```
conda create -n bsr python=3.11
conda activate bsr
pip install torch torchvision numpy==1.26.3 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

## Pretraining the BSR Renderer
```
python pretrain_bsr.py
```

## Preprocessing EEG datasets and Rendering Spectrum Video
```
python preprocess/preprocess_seed_fewshot.py
python rendering/render_seed_clahe_fewshot.py
```
Dir of i/o can be modified in the scripts. 

All EEG should be saved using the save_eeg method in preprocess/utils.py.

## Download VideoMAE from Huggingface
```
huggingface-cli download MCG-NJU/videomae-base --local-dir ./videomae-base
```

## Pretraining VideoMAE
```
python pretrain_vmae_unsupervised.py
```

## Few-shot Learning
```
python fewshot_seed.py
```
