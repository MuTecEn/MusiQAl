# MusiQAl: Music Question-Answering through Audio-Video Fusion

# Description

The MusiQAl Dataset is a comprehensive collection designed to advance the field of music question-answering (MQA). It features 292 carefully selected videos and 11,793 question-answer pairs, totaling five hours of music performance footage. Performances are sourced from notable projects and public datasets, along with a diverse range of YouTube videos.

## Content and Representation
47 Music Instruments: Spanning a wide range of musical traditions and genres.
Performances Across 18 Countries and Five Continents: Showcasing a global perspective on music.
11 Question Types: Covering audio, visual, and audio-visual scenarios.
The dataset highlights four specific aspects of music performance: instrument playing, dancing, singing, and their combinations. Recognizing the cultural significance of rhythm and dance, MusiQAl includes music made for dance, integral to many traditions.

## Data Availability
For access to the dataset, please visit this link: https://zenodo.org/records/13623449

# Requirements

```bash 
pip install -r requirements.txt
```

# Usage

## Clone the repository

``` bash
git clone https://github.com/MuTecEn/MusiQAl.git
```

## Download data

Please make sure to download the wav files, video frames, features, and JSON files from this link: https://zenodo.org/records/13623449

## Run AVST

```bash
python ${DIR}/net_grd_avst/main_avst.py --label_train ${DIR}/data/json/avqa-train.json --label_val ${DIR}/data/json/avqa-val.json --label_test ${DIR}/data/json/avqa-test.json --batch-size 16 --epochs 30 --mode train --model_save_dir ${DIR}/net_grd_avst/avst_models/
```

## Run LAVISH 

```bash
python ${DIR}/net_grd_avst/main_avst.py --Adapter_downsample=8 --audio_dir=${DIR}/data/feats/vggish --epochs=30 -- is_before_layernorm=1 --is_bn=0 --is_gate=1 --is_multimodal=1 --is_post_layernorm=1 --is_vit_ln=1 --num_conv_group=4 --num_tokens=64 --num_workers=16 --video_res14x14_dir=${DIR}/data/feats/res18_14x14 --video_dir=${DIR}/data/frames --wandb=1 --mode train
```

## Test and Compare Results

AVST:

```bash
python ${DIR}/net_grd_avst/main_avst.py --label_train ${DIR}/data/json/avqa-train.json --label_val ${DIR}/data/json/avqa-val.json --label_test ${DIR}/data/json/avqa-test.json --batch-size 16 --epochs 30 --mode test --model_save_dir ${DIR}/net_grd_avst/avst_models/
```

LAVISH: 
```bash
python ${DIR}/net_grd_avst/main_avst.py --Adapter_downsample=8 --audio_dir=${DIR}/data/feats/vggish --epochs=30 -- is_before_layernorm=1 --is_bn=0 --is_gate=1 --is_multimodal=1 --is_post_layernorm=1 --is_vit_ln=1 --num_conv_group=4 --num_tokens=64 --num_workers=16 --video_res14x14_dir=${DIR}/data/feats/res18_14x14 --video_dir=${DIR}/data/frames --wandb=1 --mode test
```
## Results Table

![image](https://github.com/user-attachments/assets/6d38831d-67ef-45e2-918c-842d616e46a3)

