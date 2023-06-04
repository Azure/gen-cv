import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:100"

from accelerate.utils import write_basic_config
write_basic_config()