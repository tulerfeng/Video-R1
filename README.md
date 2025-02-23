# Video-R1: Towards Super Reasoning Ability in Video Understanding 

This work aims to integrate deep thinking capabilities into video understanding tasks through the R1 paradigm. 

For the first time, we achieved a simultaneous increase in both accuracy and thinking length in video understanding domain.

This is a preliminary repo, and we will continue to develop our Video-R1 model in the future.

## Updates
- [2025/02/23] We release training code and data of Video-R1

  

## Findings

### *Shared Growth of Accuracy and Thinking Length is Possible in Video*

In many previous multimodal R1 repositories, the thinking length either showed little to no increase (e.g., [Open R1 Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video?tab=readme-ov-file) ) or even decreased (e.g., [R1-V](https://github.com/Deep-Agent/R1-V) ). 

In this work, we demonstrate that this issue can be addressed by using an appropriate base model and a strong reasoning dataset. We train [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) using GRPO with accuracy and format rewards on the [DVD-counting](https://huggingface.co/datasets/Video-R1/DVD-counting) dataset. Training the 7B model for 900 steps can be completed in approximately 10 hours using 4 x A100 (80G) GPUs. The training curve is as follows:

<img src="\images\7B_curve.png" alt="7B_curve" style="zoom:70%;" />



### *Weak Base Model Hinders the Emergence of Deep Thinking in Video* 

We train [Qwen2-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)  using the same setting on the DVD-counting dataset. In contrast, this model shows a decrease in thinking length. 

In some cases, the model even skips the thinking process and outputs sentences like this: `<think>\n</think>\n<answer>2</answer>`.



<img src="\images\2B_curve.png" alt="2B_curve" style="zoom:70%;" />




### *Weak Reasoning Data Maybe Not Beneficial for Reinforcing Deep Thinking* 

We train [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) on a subset of NExT-QA dataset with little reasoning. We can notice that there is almost no increase in thinking length. This indicates that reinforcing deep thinking may require strong reasoning data.



<img src="\images\7B_nextqa.png" alt="7B_nextqa" style="zoom:70%;" />





## Datasets

The video files are in the zip file and the train/test splits are in the jsonl file.

[🤗 Video-R1 Dataset: DVD-counting](https://huggingface.co/datasets/Video-R1/DVD-counting)

This dataset is extracted from "DVD: A Diagnostic Dataset for Multi-step Reasoning in Video Grounded Dialogue"

## Performance

We can observe that RL training results in an accuracy boost of around 10% on DVD-counting-test


<div align="center">

| Dataset           | Qwen2-VL-7B-Instruct | Video-R1-7B |
| ----------------- | -------------------- | ----------- |
| DVD-counting-test | 25.0                 | 34.5        |
</div>


Reasoning Samples:


<div align="center">
  <img src="\images\CATER_new_003595.gif" alt="Descriptive alt text" width="40%">
</div>


<div align="center">
  <img src="\images\sample.png" alt="Descriptive alt text" width="75%">
</div>






## Set up

```bash
git clone https://github.com/tulerfeng/Video-R1
cd Video-R1

# build environment
conda create -n video-r1 python=3.11 
conda activate video-r1
bash setup.sh

# qwen video extraction setting
cd src/qwen-vl-utils
pip install -e .
cd ..

# download dataset
git lfs install
git clone https://huggingface.co/datasets/Video-R1/DVD-counting
```

Please put the downloaded dataset to `src/r1-v/data/` 

## Training

Train Qwen2-VL-7B-Instruct with GRPO

```bash
bash src/scripts/run_grpo_video.sh
```



## Evaluation

Evaluation on video counting task

```bash
python ./src/eval/test_qwen2vl_video_counting.py
```



## Acknowledgements

We sincerely appreciate the contributions of the open-source community. The related projects are as follows:

+ [R1-V](https://github.com/Deep-Agent/R1-V)  (our initial codebase)
+ [Open R1 Video](https://github.com/Wang-Xiaodong1899/Open-R1-Video?tab=readme-ov-file)  (concurrent work)

- [open-r1-multimodal](https://github.com/EvolvingLMMs-Lab/open-r1-multimodal)
- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-R1) 
- [open-r1](https://github.com/huggingface/open-r1)

  
