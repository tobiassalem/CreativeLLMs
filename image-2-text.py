from transformers import pipeline


def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=1000)

    text = image_to_text(url)[0]["generated_text"]
    print(f"Generated text from image: {text}")
    return text


# test image-to-text generation
scenarioOne = img2text("bruce-lee-operation-dragon-1973.jpg")
scenarioTwo = img2text("people-on-boat.jpg")