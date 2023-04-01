# Told you where to look - a multimodal approach for saliency prediction utilizing image captions

## Documentation 🗎

This repository contains:

- a final paper and a short video about this project in the folder [report](report)
- the training history, logs and predicted images as well as our saved models in the folder [results](results)

## Usage 🧠

### Datasets 👀

*SALICON:* Download the training and validation images as well as the training and validation annotations from the [salicon-website](http://salicon.net/download/). All the images in the SALICON dataset were taken from MS COCO 2014 release, therefore download the respective captions of that release from the [mscoco-website](https://cocodataset.org/#download).

*capgaze1*: Download the `transcribed_text.zip`, `images.zip` and `gaze_converted_2.zip`from the original author's [GoogleDrive](https://drive.google.com/open?id=1qlOCr8TX6dmAxhlCob79X29riyQ_MRlq).

After you have downloaded everything, unzip it and place it so that the folder structure of [data](data) looks like this:

```
data/
├── capgaze1/
│   ├── gaze_converted_2/
│   │   ├── 1/
│   │   |   └── gaze/
│   │   │       ├── 2008_000032.npy
│   │   │       ├── ...
│   │   │       └── 2008_008755.npy
│   │   ├── 2/
│   │   |   └── gaze/
│   │   │       ├── 2008_000032.npy
│   │   │       ├── ...
│   │   │       └── 2008_008755.npy
│   │   ├── 3/
│   │   |   └── gaze/
│   │   │       ├── 2008_000032.npy
│   │   │       ├── ...
│   │   │       └── 2008_008755.npy
│   │   ├── 4/
│   │   |   └── gaze/
│   │   │       ├── 2008_000032.npy
│   │   │       ├── ...
│   │   │       └── 2008_008755.npy
│   │   └── 5/
│   │       └── gaze/
│   │           ├── 2008_000032.npy
│   │           ├── ...
│   │           └── 2008_008755.npy
│   ├── images/
│   │   ├── 2008_000032.npy
│   │   ├── ...
│   │   └── 2008_008755.npy
│   └── transcribed_text
|       ├── 1/
│       |   ├── 2008_000032.json
│       |   ├── ...
│       │   └── 2008_008755.json
|       ├── 2/
│       |   ├── 2008_000032.json
│       |   ├── ...
│       │   └── 2008_008755.json
|       ├── 3/
│       |   ├── 2008_000032.json
│       |   ├── ...
│       │   └── 2008_008755.json
|       ├── 4/
│       |   ├── 2008_000032.json
│       |   ├── ...
│       │   └── 2008_008755.json
│       └── 5/
│           ├── 2008_000032.json
│           ├── ...
│           └── 2008_008755.json
├── SALICON/
|   ├── annotations/
│   │   ├── train/
│   │   │   ├── captions_train2014.json
│   │   │   └── fixations_train2014.json
│   │   └── val/
│   │       ├── captions_val2014.json
│   │       └── fixations_val2014.json
|   └── images/
|       ├── train/
|       |   ├── COCO_train2014_000000000009.jpg
│       |   ├── ...
|       |   └── COCO_train2014_000000581797.jpg
|       └── val/
|           ├── COCO_val2014_000000000133.jpg
│           ├── ...
|           └── COCO_val2014_000000581899.jpg
├── dataloader_capgaze1.py
├── dataloader_salicon.py
└── DatasetInspection.ipynb
```

Then you can create the two datasets and store them as as `TFRecords` dataset:

```shell
cd data/
pip install -r requirements.txt
python dataloader_salicon.py train2014
python dataloader_salicon.py val2014
python dataloader_salicon.py
```

### Models 🤖

The full source code for our models can be found in the folder [model](model). They can either be run directly from the command line or within Google Colab. By passing arguments to the main.py file, you can choose whether to use the baseline or multimodal model. See the table below for all the different arguments you can use: 

|     Argument     |  Defaul  |   Options   |
| :--------------: | :------: | :---------: |
|      --data      | SALICON  |  capgaze1   |
|     --colab      |    1     |      0      |
|     --model      | baseline | multimodal  |
|      --save      |    0     |      1      |
|   --load_model   |    0     |      1      |
|  -- config_name  |   None   |     str     |
| --use_pretrained |    1     |      0      |
|   --fine_tune    |    0     |     int     |
|       --lr       |   1e-4   |    float    |
|    --lr_sched    |    0     |     int     |
|     --optim      |   Adam   | SGD/Adagrad |
|   --step_size    |    5     |     int     |
|    --l1_norm     |   None   |    float    |
|    --l2_norm     |   None   |    float    |
| --text_with_dense|    1     |      0      |
|  --start_epoch   |    0     |     int     |
|   --no_epochs    |    24    |     int     |
|   --batch_size   |    32    |     int     |

We trained our models with the following settings:

```shell
cd model/

python main.py --model multimodal --data SALICON --config_name multimodal_SALICON_v1 
--fine_tune 2 --start_epoch 0 --no_epochs 10 --save 1

python main.py --model baseline --data SALICON --config_name baseline_SALICON_v1 
--fine_tune 2 --start_epoch 0 --no_epochs 10 --save 1

python main.py --model multimodal --data capgaze1 --config_name multimodal_SALICON_v1 
--fine_tune 2 --start_epoch 0 --no_epochs 10 --load 1

python main.py --model baseline --data capgaze1 --config_name multimodal_SALICON_v1 
--fine_tune 2 --start_epoch 0 --no_epochs 10 --load 1
```
