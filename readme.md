### Creative LLMs
Simple experiments with Python, PyTorch, and datasets from huggingface. ^^

### Workflow for tokenizing your dataset 
Tokenizing is a common form of pre-processing for your dataset when using LLMs.

1) 
Start by loading the rotten_tomatoes dataset and the tokenizer corresponding to a pretrained BERT model. 
Using the same tokenizer as the pretrained model is important because you want to make sure the text is split in the same way.

2) 
Call your tokenizer on the first row of text in the dataset:

The tokenizer returns a dictionary with three items:
* input_ids: the numbers representing the tokens in the text.
* token_type_ids: indicates which sequence a token belongs to if there is more than one sequence.
* attention_mask: indicates whether a token should be masked or not.

These values are actually the model inputs.

3)
 The fastest way to tokenize your entire dataset is to use the map() function. 
This function speeds up tokenization by applying the tokenizer to batches of examples instead of individual examples. 
Set the batched parameter to True:

```
def tokenization(example):
    return tokenizer(example["text"])

dataset = dataset.map(tokenization, batched=True)
```

4)
Set the format of your dataset to be compatible with your machine learning framework.
Use the set_format() function to set the dataset format to be compatible with PyTorch:
```
dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
dataset.format['type']
'torch'
```
5) The dataset is now ready for training with your machine learning framework!

### References
* Python - Ref. https://www.python.org/
* PyTorch - Ref. https://pytorch.org/
* Huggingface Datasets - Ref. https://huggingface.co/docs/datasets/use_dataset
