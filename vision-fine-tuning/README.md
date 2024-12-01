# Vision Fine-tuning examples

This asset collection is intended to provide reusable examples for vision fine-tuning based on Azure OpenAI and custom OSS models. We have started by providing an end to end example of GPT-4o fine-tuning based on a public dataset and intend to add further examples in the near future.

## Get started

Create and activate a virtual Python environment for running the app.
The following example shows how to create a Conda environment named `vision-ft`:

```bash
conda create -n vision-ft python=3.12
conda activate vision-ft
```

Install the required packages. Navigate to the `vision-fine-tuning` folder and execute the following:

```bash
pip install -r requirements.txt
```

__Required Services:__
- An Azure OpenAI resource with the following model deployments:
   - GPT-4o

__Optional Services:__
- Azure AI Foundry

Navigate to the starter notebook:

- [Fine-tuning Azure OpenAI GPT-4o for Chart Analysis](01-AOAI-vision-fine-tuning-starter/fine-tune-aoai-gpt4o-for-chart-analysis.ipynb)
