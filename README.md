# MusiQAl: Music Question-Answering through Audio-Video Fusion

# Description

The MusiQAl Dataset is a comprehensive collection designed to advance the field of music question-answering (MQA). It features 292 carefully selected videos and 11,834 question-answer pairs, totaling five hours of music performance footage. Performances are sourced from notable projects and public datasets, along with a diverse range of YouTube videos.

## Content and Representation
47 Music Instruments: Spanning a wide range of musical traditions and genres.
Performances Across 18 Countries and Five Continents: Showcasing a global perspective on music.
11 Question Types: Covering audio, visual, and audio-visual scenarios.
The dataset highlights four specific aspects of music performance: instrument playing, dancing, singing, and their combinations. Recognizing the cultural significance of rhythm and dance, MusiQAl includes music made for dance, integral to many traditions.

## Data Availability
For access to the dataset, please visit this link.

interactive figures

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

Please make sure to download the wav files, video frames, and features.
Json files of QA Pairs: 

## Run AVST

```bash
python net_grd_avst/main_avst.py --mode train --audio_dir = ./directory/to/your/audio/features/ --video_res14x14_dir = ./directory/to/your/video/features
```

## Run LAVISH 

```bash
python3 net_grd_avst/main_avst.py --Adapter_downsample=8 --audio_dir=./directory/to/your/audio/features/ --batch-size=1 --early_stop=5 --epochs=30 --is_before_layernorm=1 --is_bn=0 --is_gate=1 --is_multimodal=1 --is_post_layernorm=1 --is_vit_ln=1 --lr=8e-05 --lr_block=3e-06 --num_conv_group=4 --num_tokens=64 --num_workers=16 --video_res14x14_dir=./directory/to/your/video/features --wandb=1
```

## Test and Compare Results

AVST:

```bash
python net_grd_avst/main_avst.py --mode test --audio_dir = ./directory/to/your/audio/features/ --video_res14x14_dir = ./directory/to/your/video/features
```

LAVISH: 

## Results Table
