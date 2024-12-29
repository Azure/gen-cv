# Fine-Tuning GPT-4o for Action Recognition in Video Clips

This Solution Accelerator demonstrates how to use Azure OpenAI GPT-4o vision fine-tuning to improve the model's performance in detecting human activities in video clips. The project utilizes the [UCF101 - Action Recognition](https://www.kaggle.com/datasets/matthewjansen/ucf101-action-recognition) dataset from Kaggle, a comprehensive video dataset featuring 101 distinct human action categories such as "playing guitar," "surfing," and "knitting." It contains 13,320 video clips, each labeled with a single action category.

Below are examples of video frames representing 8 of the 101 classes in the dataset:

<img src="frame-samples.png" alt="Frame Samples" width="1000">

__Acknowledgements:__

- Dataset: https://www.crcv.ucf.edu/research/data-sets/ucf101/
- Citation: https://arxiv.org/abs/1212.0402

This Solution Accelerator provides reusable code to help you apply vision fine-tuning for video analysis in various use cases.

> **Note:** While many training images from the selected dataset were rejected by the Fine-Tuning API due to current restrictions on images containing people and faces (for AI safety considerations), the code remains fully applicable to your own datasets and use cases.  
> We hope these API limitations will be relaxed in the future. If not, we will work on creating a fine-tuning example using alternative datasets.

## Get started

Create and activate a virtual Python environment for running the code.
The following example shows how to create a Conda environment named `video-ft`:

```bash
conda create -n video-ft python=3.12
conda activate video-ft
```

Install the required packages. Navigate to the `02-AOAI-video-fine-tuning` folder and execute the following:

```bash
pip install -r requirements.txt
```

__Required Services:__
- An Azure OpenAI resource with the following model deployments:
   - GPT-4o (2024-08-06)

__Optional Services:__
- Azure AI Foundry
- An Azure Storage Account

Rename the environemt file template `.env.template` to `.env` and add your credentials by editing the file.

Navigate to the video fine-tuning notebook:

- [Fine-Tuning GPT-4o for Action Recognition in Video Clips](fine-tune-aoai-gpt4o-action-detection.ipynb)

__Note:__ If you encounter the following error at the start of the application: `ImportError: libGL.so.1: cannot open shared object file: No such file or directory`  
In this case, your system is missing the shared `libGL.so.1` library which is required by OpenCV.  
On a Ubuntu system, you can install the missing library as follows:
```bash
sudo apt update
sudo apt install libgl1-mesa-glx
```