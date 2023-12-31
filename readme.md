# Meta-causal

The code for **Meta-causal Learning for Single Domain Generalization [CVPR2023]**. Our code is based on the method of PDEN(https://github.com/lileicv/PDEN/).

### Dataset

- Download the data and model from [Baidu Cloud Disk](https://pan.baidu.com/s/14pdVbNAHWKeC4AE7QqtFmw) (password:pxvt ). 
- Place the dataset files in the path `./data/` and the model files in the path `./`

### Environment

Please refer to `env.yaml`

### Train and Test
- For digit, run the command `bash run_my_joint_test.sh 0` under the path `./run_digits/` .
- For PACS, when using art_painting as the source domain, run the command `bash run_my_joint_v13_test.sh 0` under the path `./run_PACS/` .

### If this code is helpful, please cite our paper

```
@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Jin and Gao, Zhi and Wu, Xinxiao and Luo, Jiebo},
    title     = {Meta-Causal Learning for Single Domain Generalization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {7683-7692}
}
```

### Contact
gaozhi_2017@126.com

