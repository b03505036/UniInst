
## UniInst: Towards End-to-End Instance Segmentation with Unique Representation [UniInst](https://arxiv.org/)

Name | inf. time | mask AP | download
--- |:---:|:---:|:---:
[UniInst_MS_R_50_3x](configs/UniInst/UniInst_MS_R_50_3x.yaml) | 20 FPS | 38.4 | [model](https://drive.google.com/file/d/1Akh7nEJWWt7TCAHaNI4a-mcjKOJY1yIv/view?usp=sharing)
[UniInst_MS_R_50_6x](configs/UniInst/UniInst_MS_R_50_6x.yaml) | 20 FPS | 38.9 | [model](https://drive.google.com/file/d/1fNNagWfLUYNso7P60D30vQJr4NW2XM3X/view?usp=sharing)
[UniInst_MS_R_101_3x](configs/UniInst/UniInst_MS_R_101_3x.yaml) | 16 FPS  | 39.7 | [model](https://drive.google.com/file/d/1BkS67s0Ql2ESmfKHNG90x1ZMizC0QERH/view?usp=sharing)
[UniInst_MS_R_101_6x](configs/UniInst/UniInst_MS_R_101_6x.yaml) | 16 FPS  | 40.2 | [model](https://drive.google.com/file/d/1bBr1DwYBAb13nvomuKqM8sghRoJjPbXq/view?usp=sharing)

For more models and information, please refer to CondInst [README.md](configs/CondInst/README.md).

Note that:
- Inference time for all projects is measured on a NVIDIA V100 with batch size 1.
- APs are evaluated on COCO2017 test split unless specified.


## Tencent Quick Installaion
### 1. Load Docker

```
mirrors.tencent.com/rpf_detectronv2/rpf_detectronv2_oym:version1.0
```
### 2. Set Variavles

```
# Take youtu for Example
export TORCH_HOME=/youtu/xlab-team4/share/pretrained
export DETECTRON2_DATASETS='/youtu/xlab-team4/share/datasets/'
export FVCORE_CACHE='/youtu/xlab-team4/share/pretrained/'
```

### 3. Setup

```
python3 setup.py build develop
```
## Installation for others

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).


Then build UniInst with:

```
python3 setup.py build develop
```


Some projects may require special setup, please follow their own `README.md` in [configs](configs).

## Start Train and Demo 

### Inference with Pre-trained Models

1. Pick a model and its config file, for example, `UniInst_R_50_3x.yaml`.
2. Download the model 
3. Run the demo with
```
python demo/demo.py \
    --config-file configs/UniInst/UniInst_MS_R_50_3x.yaml \
    --input input1.jpg input2.jpg \
    --opts MODEL.WEIGHTS UniInst_R_50_3x.pth
```

### Train Your Own Models

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md),
then run:

```
OMP_NUM_THREADS=1 python3 tools/train_net.py \
    --config-file configs/UniInst/UniInst_MS_R_50_3x.yaml \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/UniInst_R_50_3x
```
To evaluate the model after training, run:

```
OMP_NUM_THREADS=1 python3 tools/train_net.py \
    --config-file configs/UniInst/UniInst_MS_R_50_3x.yaml \
    --eval-only \
    --num-gpus 8 \
    OUTPUT_DIR training_dir/UniInst_R_50_3x \
    MODEL.WEIGHTS training_dir/UniInst_R_50_3x/model_final.pth
```
Note that:
- The configs are made for 8-GPU training. To train on another number of GPUs, change the `--num-gpus`.
- If you want to measure the inference time, please change `--num-gpus` to 1.
- We set `OMP_NUM_THREADS=1` by default, which achieves the best speed on our machines, please change it as needed.
- This quick start is made for FCOS. If you are using other projects, please check the projects' own `README.md` in [configs](configs). 


## Citing AdelaiDet

Note that ourwork is based on the AdelaiDet. If you use our code in your reaserch or works, please also cite AdelaiDet.

Please use the following BibTeX entries:

```BibTeX

@misc{tian2019adelaidet,
  author =       {Tian, Zhi and Chen, Hao and Wang, Xinlong and Liu, Yuliang and Shen, Chunhua},
  title =        {{AdelaiDet}: A Toolbox for Instance-level Recognition Tasks},
  howpublished = {\url{https://git.io/adelaidet}},
  year =         {2019}
}

#UniInst arxiv#

```

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).
