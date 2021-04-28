# AMSS-Net: Audio Manipulation on User-Specified Sources with Textual Queries

An official implementation of the paper: "AMSS-Net: Audio Manipulation on User-Specified Sources with Textual Queries"

- [Demonstration link](https://kuielab.github.io/AMSS-Net/)

---

This repository does not contain the complete source code yet.

We will upload codes sooner or later, after refactorization, for better readability.

## 1. Installation

(Optional)
```
conda create -n amss
conda activate amss
```

(Install)
```
conda install pytorch=1.7.1 cudatoolkit=11.0 -c pytorch
conda install -c conda-forge ffmpeg librosa
conda install -c anaconda jupyter
pip install torchtext musdb museval pytorch_lightning wandb pydub pysndfx
```

Also, you have to install sox,
- for linux: ```conda install -c conda-forge sox```
- for Windows: [download](https://sourceforge.net/projects/sox/```)

## 2. Dataset: Musdb18

### 1. Download

1. [Full dataset](https://sigsep.github.io/datasets/musdb.html)
    
    - The entire dataset is hosted on Zenodo and requires that users request access.
    - The tracks can only be used for academic purposes. 
    - They manually check requests. 
- After your request is accepted, then you can download the full dataset!
    
2. or Sample Dataset
    - download sample version of MUSDB18 which includes 7s excerpts using this script

        ```python
        import musdb
        musdb.DB(root='etc/musdb18_dev', download=True)
        ```

### 2. Generate wave files

- run this!

    ```shell
    musdbconvert <your_DIR> <target_DIR> 
    ```

- musdbconvert is automatically installed if you have installed musdb with:

    ```shell
    pip install musdb
    ```

## 3. Train script example

- AMSS-Net

``` shell
python train.py --musdb_root ../../repos/musdb18_wav --pre_trained_word_embedding glove.6B.100d --embedding_dim 100 --task task2 --model isolasion_smpocm --n_fft 4096 --gpus 4 --distributed_backend ddp --sync_batchnorm True --save_top_k 3 --min_epochs 100 --num_head 6 --num_latent_source 8 --optimizer adam --batch_size 4 --enable_pl_optimizer True --train_loss spec_mse --val_loss raw_l1 --check_val_every_n_epoch 10 --lr 0.0001 --precision 16 --num_worker 32 --pin_memory True --seed 2020 --deterministic True --n_blocks 9 --run_id your_run_id --log wandb
```

## 3. Evaluation script example


```shell
auto_task2_eval.py --musdb_root ../../repos/musdb18_wav --ckpt_root etc/checkpoints/ --model isolasion_smpocm --cuda True --batch_size 8 --logger wandb
```

