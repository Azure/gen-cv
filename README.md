# Vision AI Solution Accelerator

<img src="./media/gen-cv.png" alt="drawing" style="width:1200px;"/>

This repository serves as a rich resource offering numerous examples of synthetic image generation, manipulation, and reasoning. Utilizing Azure Machine Learning, Computer Vision, OpenAI, and widely acclaimed open-source frameworks like Stable Diffusion, it equips users with practical insights into the application of these powerful tools in the realm of image processing.

### Use cases

- [Explore the OpenAI DALL E-2 API](dalle2-api/DALLE2-api-intro.ipynb)
- [Create images with the Azure OpenAI DALL E-2 API](dalle2-api/Florenace-AOAI-DALLE2.ipynb)
- [Remove background from images using the Florence foundation model](dalle2-api/Remove-background.ipynb)
- [Create and find images with Stable Diffusion and Florence Vector Search](image-embeddings/generate-and-search-images.ipynb)
- [Add your own Objects and Styles to Image Generation Models](generation-finetuning/README.md)

### Getting Started
The code within this repository has been tested on both __Github Codespaces__ compute and an __Azure Machine Learning Compute Instance__. Although the use of a GPU is not a requirement, it is highly recommended if you aim to generate a large number of sample images using Stable Diffusion.

Follow these steps to get started:

1.  Clone this repository on your preferred compute using the following command:  
```bash
git clone https://github.com/Azure/gen-cv.git
```

2. Create your Python environment and install the necessary dependencies. For our development, we utilized Conda. You can do the same with these commands:

```bash
conda create -n gen-cv python=3.10
conda activate gen-cv
pip install -r requirements.txt
```

3. From the list provided above, select a sample notebook. After making your selection, configure the Jupyter notebook to use the kernel associated with the environment you set up in Step 2.
4. Copy the `.env.template` file to `.env` to store your parameters.
```bash
cp .env.template .env
```
5. Add the required parameters and keys for your services to the `.env` file.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
