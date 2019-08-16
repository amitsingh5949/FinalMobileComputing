import statistics
from scipy.stats.stats import pearsonr
from scipy.fftpack import fft, ifft
import os
import numpy
import pickle
import pandas as pd
from pandas import DataFrame
import urllib.request
from app import app
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
	if request.method == 'POST':
        # check if the post request has the file part
		if 'file' not in request.files:
			flash('No file part')
			return redirect(request.url)
		file = request.files['file']
		if file.filename == '':
			flash('No file selected for uploading')
			return redirect(request.url)
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			flash('File successfully uploaded')
			return redirect('/')
		else:
			flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
			return redirect(request.url)


@app.route('/run', methods=['GET'])
def run():
	return '1_'

@app.route('/predict', methods=['GET'])
def predict():
    preprocess()
    df = pd.read_csv('/home/ec2-user/upload/preprocess.csv')
    y_pred = svmIrisModel.predict(df)
    output = ""
    for index, row in df.iterrows():
        output = output + str(y_pred[index]) + "_"

    os.remove('/home/ec2-user/upload/preprocess.csv')
    os.remove('/home/ec2-user/upload/data_file.csv')
    return str(output)


def load_model():
    global svmIrisModel
    svmIrisFile = open('./gesture.pckl', 'rb')
    svmIrisModel = pickle.load(svmIrisFile)
    svmIrisFile.close()


def get_energy(values):
    fft2 = fft(values)
    fft2 = numpy.abs(fft2)
    energy = sum(i * i for i in fft2) / len(fft2)
    return energy


def preprocess():
    csv2 = pd.read_csv('/home/ec2-user/upload/data_file.csv')
    count = csv2['ID'].nunique()
    train = []
    for i in range(1, count+1):

        csv = csv2[csv2['ID'] == i]

        x = csv['x'].tolist()
        y = csv['y'].tolist()
        z = csv['z'].tolist()
        

        x_mean = numpy.mean(x)
        y_mean = numpy.mean(y)
        z_mean = numpy.mean(z)

        x_stddev = statistics.stdev(x)
        y_stddev = statistics.stdev(y)
        z_stddev = statistics.stdev(z)

        x_energy = get_energy(x)
        y_energy = get_energy(y)
        z_energy = get_energy(z)

        x_y_coeff, _ = pearsonr(x, y)
        x_z_coeff, _ = pearsonr(x, z)
        y_z_coeff, _ = pearsonr(y, z)

        data = {'x_mean': x_mean, 'y_mean': y_mean, 'z_mean': z_mean,
                'x_stddev': x_stddev, 'y_stddev': y_stddev, 'z_stddev': z_stddev,
                'x_energy': x_energy, 'y_energy': y_energy, 'z_energy': z_energy,
                'x_y_coeff': [x_y_coeff], 'x_z_coeff': [x_z_coeff], 'y_z_coeff': [y_z_coeff]}
        df = DataFrame(data, columns=['x_mean', 'y_mean', 'z_mean',
                                      'x_stddev', 'y_stddev', 'z_stddev',
                                      'x_energy', 'y_energy', 'z_energy',
                                      'x_y_coeff', 'x_z_coeff', 'y_z_coeff'])

        header = True
        if os.path.exists(r'/home/ec2-user/upload/preprocess.csv'):
            header = False

        df.to_csv(r'/home/ec2-user/upload/preprocess.csv', mode='a', index=None, header=header)

if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0', debug=True)
