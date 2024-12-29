# Fine-tuning Azure OpenAI GPT-4o for Chart Analysis

This notebook demonstrates the process of vision fine-tuning of GPT-4o leveraging a chart analysis benchmark dataset for visual and logical reasoning. It covers data preparation, fine-tuning, deployment, and evaluation against GPT-4o's baseline performance.

We are using the **ChartQA** dataset, a benchmark designed for question answering tasks involving chart images. Each entry in the dataset comprises a chart image, an associated question, and the corresponding answer, facilitating the development and evaluation of models that integrate visual and logical reasoning to interpret and analyze information presented in graphical formats.

<img src="qna.png" alt="Frame Samples" width="1000">

__Acknowledgements:__  
This project utilizes the ChartQA dataset introduced by Masry et al. in their paper, *ChartQA: A Benchmark for Question Answering about Charts with Visual and Logical Reasoning* (Findings of ACL 2022). We acknowledge the authors for providing this valuable resource. For more details, refer to the publication: [ChartQA: ACL 2022](https://aclanthology.org/2022.findings-acl.177).

## Get started

Create and activate a virtual Python environment for running the code.
The following example shows how to create a Conda environment named `vision-ft`:

```bash
conda create -n vision-ft python=3.12
conda activate vision-ft
```

Install the required packages. Navigate to the `01-AOAI-vision-fine-tuning-starter` folder and execute the following:

```bash
pip install -r requirements.txt
```

__Required Services:__
- An Azure OpenAI resource with the following model deployments:
   - GPT-4o

__Optional Services:__
- Azure AI Foundry

Rename the environemt file template `.env.template` to `.env` and add your credentials by editing the file.

Navigate to the vision fine-tuning notebook:

- [Fine-tuning Azure OpenAI GPT-4o for Chart Analysis](fine-tune-aoai-gpt4o-for-chart-analysis.ipynb)
