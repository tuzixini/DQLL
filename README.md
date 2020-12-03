# DQLL
---
This repo is the official implementation of [paper **Deep Reinforcement Learning Based Lane Detection and Localization**](https://www.sciencedirect.com/science/article/abs/pii/S0925231220310833)  

## Getting Started
```$CODEROOT```: the path where you put this code.  
```$DATAROOT```: the path where you put the dataset.  

## Preparation
- Prerequisites
  - Python 3.x
  - Pytorch 1.x: http://pytorch.org .
  <!-- - other libs in ```requirements.txt```, run ```pip install -r requirements.txt```. -->

- Installation
  - Clone this repo:
    ```
    git clone https://github.com/tuzixini/DQLL.git  
    ```
  - Make sure your code dir ```$CODEROOT``` is in this folder tree:
    ```
    --$CODEROOT  
      --DQLL  
        |-- config.py  
        |-- dataset.py  
        |-- genMyData.py  
        |-- ...  
    ```
---
## Data Preparation
### TuSimple Dataset
- Download Processed dataset from [tusimple-benchmark](https://github.com/TuSimple/tusimple-benchmark/issues/3)(**Pay Attention:** what we need is the data from"LANE DETECTION CHALLENGE", like following zips.)  
    ```
    dataset  
    - train_set.zip  
    - test_set.zip  
    - test_baseline.json  
    ground truth  
    - test_label.json  
    ```

- Unzip all zips into ```$DATAROOT``` dir.  
  Make sure it has following folder tree:  
  ```
  -- $DATAROOT
      |-- test_set
      |  -- clips
      |  -- test_tasks_0627.json
      |  -- readme.md
      |-- train_set
      |  -- clips
      |  -- label_data_0313.json
      |  -- label_data_0531.json
      |  -- label_data_0601.json
      |  -- readme.md
      |-- test_label.json
  ```

- Use ```genMyData.py``` and ```genMeanImg.py``` to generate new data.  
  1. Modify variable ```DATAROOT``` (file ```getMyData.py```, line 210) to real ```$DATAPATH``` you use (here is r"/opt/disk/zzy/dataset/TuSimple").  
        ```python
        DATAROOT = r"/opt/disk/zzy/dataset/TuSimple"
        ```
  2. Go to ```$CODEPATH``` and run ```genMyData.py```, and wait if finish.  
        ```shell
        cd $CODEPATH
        python genMyData.py
        ```
   3. Modify variable ```DATAROOT``` (file ```genMeanImg.py```, line 22) to real ```$DATAPATH``` you use (here is r"/opt/disk/zzy/dataset/TuSimple").  
        ```python
        DATAROOT = r"/opt/disk/zzy/dataset/TuSimple"
        ```

  4. Go to ```$CODEPATH``` and run ```genMeanImg.py```, and wait if finish.  
        ```shell
        cd $CODEPATH
        python genMeanImg.py
        ```

  5. Check. After all above operations, the folder tree of ```$DATAPATH``` should become like this:  
        ```
        -- $DATAROOT
            |-- test_set
            |  -- clips
            |  -- test_tasks_0627.json
            |  -- readme.md
            |-- train_set
            |  -- clips
            |  -- label_data_0313.json
            |  -- label_data_0531.json
            |  -- label_data_0601.json
            |  -- readme.md
            |-- MyTuSimpleLane
            |  -- test
            |    -- bbox
            |      -- XXXXfiles
            |    -- DRL
            |    -- img
            |    -- mask
            |    -- mask_color
            |  -- train
            |    -- bbox
            |    -- DRL
            |    -- img
            |    -- mask
            |    -- mask_color
            |-- test_label.json
            |-- failList.json
            |-- meanImgTemp.npy
            |-- train_img_list.json
            |-- train_DRL_list.json
            |-- test_img_list.json
            |-- test_DRL_list.json
        ```

---
## Train the model
### Modify the ```config.py```
- Modify variable ```__C.DATAROOT``` (file ```config.py```, line 39) to real ```$DATAPATH``` you use (here is r"/opt/disk/zzy/dataset/TuSimple").  
    ```python
    __C.DATAROOT = r'/opt/disk/zzy/datasets/TuSimpleLane'
    ```
- Modify GPU settings according to your devices(```config.py```,line 26, 27, 28):  
    ```python
    __C.TRAIN.USE_GPU = True
    if __C.TRAIN.USE_GPU:
        __C.TRAIN.GPU_ID = [2, 3]
    ```
- For other parameter setting, please it on file ```config.py```.  
### Train the model:  
- Go to ```$CODEPATH``` and train.  
    ```shell
    cd $CODEPATH
    python train.py
    ```
---
## Val/Test and Visualization  
- Modify variable ```cfg.TRAIN.RESUME_PATH``` (file ```visAllVal.py```, line 335) to real model check point path you use (here is r'/opt/disk/zzy/project/DRL_lane/DRL_Code_TuSimple/DRL_Lane_Pytorch/exp/20-04-19-23-36_TuSimpleLane/EP_62_HitRat0.86465.pth').  
    ```python
    cfg.TRAIN.RESUME_PATH = '/opt/disk/zzy/project/DRL_lane/DRL_Code_TuSimple/DRL_Lane_Pytorch/exp/20-04-19-23-36_TuSimpleLane/EP_62_HitRat0.86465.pth'
    ```
- Go to ```$CODEPATH``` and run ```visAllVal.py```.  
    ```shell
    cd $CODEPATH
    python visAllVal.py
    ```
- All visualization will be save in folder ```cfg.EXP.PATH/_VAL_VIS```
---
# Citation
If you find this project is useful for your research, please cite:  
```
@article{zhao2020deep,
  title={Deep reinforcement learning based lane detection and localization},
  author={Zhao, Zhiyuan and Wang, Qi and Li, Xuelong},
  journal={Neurocomputing},
  volume={413},
  number={6},
  pages={328-338},
  doi={10.1016/j.neucom.2020.06.094},
  year={2020}
}
```
