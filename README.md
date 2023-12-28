
## UniInst: Towards End-to-End Instance Segmentation with Unique Representation [UniInst](https://www.sciencedirect.com/science/article/pii/S0925231222012048)

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


## Installation

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
@article{ou2022uniinst,
  title={UniInst: Unique representation for end-to-end instance segmentation},
  author={Ou, Yimin and Yang, Rui and Ma, Lufan and Liu, Yong and Yan, Jiangpeng and Xu, Shang and Wang, Chengjie and Li, Xiu},
  journal={Neurocomputing},
  volume={514},
  pages={551--562},
  year={2022},
  publisher={Elsevier}
}
```

## License

MIT License

Copyright (c) 2021 Yimin Ou

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

