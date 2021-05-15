"""
The flask application package.
"""

from flask import Flask
import os
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['SECRET_KEY'] = '124fg89dfh908605497j45hk7hk45jh7k4j5h7kj54h7kj5fdty78fthkjh7k4j5h7kj54hj7h45gi43876i'
app.config['UPLOAD_FOLDER'] = os.path.dirname(os.path.realpath(__file__)) + '\static\img'

import skin_cancer_detector.views
