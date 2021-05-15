"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, request, url_for, flash, redirect
from werkzeug.exceptions import abort
from PIL import Image
from werkzeug.utils import secure_filename
from torchvision import datasets, transforms
import torchvision, torch
import numpy as np
import torchvision.models as models
from skin_cancer_detector import app
import os



dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = dir_path + '/static/img'
ALLOWED_EXTENSIONS = set(['jpg'])



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def detect_cancer(IMG_PATH):

    classes = ["Доброкачественное новообразование", "Злокачественное новообразование"]

    net = models.resnet50(pretrained=True)
    net.fc = torch.nn.Linear(2048, 2)
    best_model = net
    best_model = torch.load('./models/best_model_resnet50_pretrained_0.9167_accuracy.pth',  map_location=torch.device('cpu'))
    best_model.eval()
    
    img = np.asarray(Image.open(IMG_PATH).convert("RGB"))/255
    img_to_tensor = torch.FloatTensor(img)
    img_permuted  = img_to_tensor.permute(2, 1, 0)

    img_to_predict = img_permuted.reshape(1,3,224,224)
    predicted_number = best_model.forward(img_to_predict).detach().numpy()

    return classes[np.argmax(predicted_number)]




@app.route('/')
@app.route('/add_img', methods=('GET', 'POST'))
def add_img():

    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename('image.jpg')
        
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            IMAGE_NAME = filename
            return redirect(url_for('show_result'))
   
    return render_template(
        'add_img.html',
        title='Загрузка изображения',
        year=datetime.now().year
    )


@app.route('/show_result', methods=['GET', 'POST'])
def show_result():
    path = UPLOAD_FOLDER + '\image.jpg'
    return render_template(
        'show_result.html',
        title = 'Результат',
        year=datetime.now().year,
        path = path,
        result = detect_cancer(path)
    )

