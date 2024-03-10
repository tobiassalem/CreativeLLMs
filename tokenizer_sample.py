from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# 1) Load the model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", load_in_4bit=True
)

# 2) Pre-process your input with a tokenizer
# When batching inputs, make sure you pad your inputs properly.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
model_inputs = tokenizer(["A list of colors: red, blue", "France is a "],
                         return_tensors="pt", padding=True).to("cuda")

# 3) After tokenizing the inputs, you can call the generate() method to returns the generated tokens.
# The generated tokens then should be converted to text before printing.
generated_ids = model.generate(**model_inputs)
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(f"LLM decoded result: {result}")