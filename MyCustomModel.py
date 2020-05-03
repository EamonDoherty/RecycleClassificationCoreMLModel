import turicreate
import turicreate as tc
import os

#load images
data = turicreate.image_analysis.load_images("images/")

data["label"] = data["path"].apply(lambda path: os.path.basename(os.path.dirname(path)))
data.save("recycleMaterial.sframe")


data = turicreate.SFrame("recycleMaterial.sframe")
testing, training = data.random_split(0.8)
classifier = turicreate.image_classifier.create(testing, target="label", model="resnet-50")

testing = classifier.evaluate(training)
print testing["accuracy"]

classifier.save("recycleMaterial.model")
classifier.export_coreml("recycleMaterial.mlmodel")

