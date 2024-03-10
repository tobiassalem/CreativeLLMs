import transformers
model = transformers.AutoModelForCausalLM.from_pretrained(
  'mosaicml/mpt-7b-storywriter',
  trust_remote_code=True
)

story = model.predict("Once upon a time there was a hobbit...")
print(story)
