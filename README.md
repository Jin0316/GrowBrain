# Growing a Brain with Sparsity-Inducing Generation for Continual Learning
This repos contains code for continually training video action recognition task from our Growing a Brain with Sparsity-Inducing Generation for Continual Learning (ICCV 2023).
Please see our paper for more detailed information.
<div align="center">
  
![ICCV](https://img.shields.io/badge/ICCV-2023-blue)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9.0-%237732a8?style=flat-square&logo=PyTorch&color=EE4C2C)](https://pytorch.org/)

![GrowBrain](images/ICCV23_main_fig.jpg)
</div>


## Requirements 
Before running the code, please install the requirements listed in the requirements.txt file. 

## Run the code
This repository supports the video action recognition experiment with UCF-101 in the original paper.

```bash
python3 -u ucf_main.py | tee growbrain.log
```
Before running the codes, you have to download the video datasets and extract the frames of videos. 
We followed the video action recognition benchmark provided from [[vCLIMB]](https://github.com/ojedaf/vCLIMB_Benchmark).
Each video is split into three segments of equal duration. 
In each segment, a frame is selected randomly. 

## Citation
```bash
@inproceedings{jin2023growing,
  title={Growing a Brain with Sparsity-Inducing Generation for Continual Learning},
  author={Jin, Hyundong and Kim, Gyeong-hyeon and Ahn, Chanho and Kim, Eunwoo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18961--18970},
  year={2023}
}
```
