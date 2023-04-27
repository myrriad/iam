from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import requests

# load image from the IAM database
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# image = Image.open('iamdataset/words-sample/a01/a01-007/a01-007-00-00.png').convert("RGB")

path = 'iamdataset/sentences-sample/a01/a01-003/a01-003-s01-03.png'
# path = 'iamdataset/forms-sample/a01-000u.png'
image = Image.open(path).convert("RGB")

# life peers into existence, they should not
image = image.rotate(0) # you engage happy " aerospace open record of
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten')
pixel_values = processor(images=image, return_tensors="pt").pixel_values

generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)