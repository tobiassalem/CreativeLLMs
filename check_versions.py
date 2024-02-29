import torch
from transformers import is_torch_available

print(torch.__version__)
print(is_torch_available())