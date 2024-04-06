from flask import Flask, request, flash, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import shutil
import torch
from torch import nn
from utils import load_model
from torchvision.models import resnet34
from PIL import Image
from torchvision import transforms
import json


app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Model
filename = "resnet34_weights_best_acc.tar"
model = resnet34(num_classes=1081)  # 1081 classes in Pl@ntNet-300K
load_model(model, filename=filename, use_gpu=False)
model.fc = nn.Sequential(model.fc, nn.Softmax(dim=1))


def get_class_labels():
    f = open("species.json")
    species = json.load(f)
    f.close()

    maps = []
    for i in sorted(species.items(), key=lambda x: x[0]):
        maps.append(i[1])


def classify(filename) -> torch.Tensor:
    target = os.path.join(APP_ROOT, 'static/images/')
    destination = os.path.join('/', target, filename)
    image = Image.open(destination).convert('RGB')
    img_transforms = transforms.Compose([transforms.Resize(size=256), transforms.CenterCrop(size=224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])
                                         ])
    image: torch.Tensor = img_transforms(image)

    with torch.no_grad():
        model.eval()
        results = model(image.unsqueeze(0))
        return results


def get_predictions(filename, k=4):
    # Return list of predictions
    f = open("species.json", encoding="utf-8")
    species = json.load(f)
    f.close()

    maps = []
    for i in sorted(species.items(), key=lambda x: x[0]):
        maps.append(i[1])

    results = torch.topk(classify(filename), k)
    indices = results.indices.tolist()[0]
    values = results.values.tolist()[0]
    predictions = [{'class': maps[index], 'confidence': value} for index,
                   value in zip(indices, values)]
    return predictions


def clear_folder():
    folder = os.path.join(APP_ROOT, 'static/images/')
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/classify", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/images/')
    # create image directory if not found
    if not os.path.isdir(target):
        os.mkdir(target)

    # check if the post request has the file part
    if 'image' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['image']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        return redirect(request.url)
    # file support verification
    ext = os.path.splitext(file.filename)[1]
    if (ext == ".jpg") or (ext == ".png") or (ext == ".jpeg"):
        print("File accepted")
    else:
        return "The selected file is not supported", 400
    filename = secure_filename(file.filename)
    clear_folder()
    file.save(os.path.join('/', target, filename))
    return get_predictions(filename)
