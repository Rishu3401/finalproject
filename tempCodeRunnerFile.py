import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from bson.objectid import ObjectId

app = Flask(__name__)
client = MongoClient("mongodb+srv://rishukumar3401:DAXpqiez0lRKZa16@cluster0.7gh8lxb.mongodb.net/finalproject")
db = client.finalproject
doctors_collection = db.doctors

model_path = r'C:\Users\rishu\Downloads\final\final\BrainTumor10EpochsCategorical.h5'
model = load_model(model_path)
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return (
            "Glioma Tumor"
            " - Gliomas are tumors that arise from glial cells, which are cells in the brain that support neurons. Gliomas are classified based on the type of glial cell they arise from, such as astrocytomas, oligodendrogliomas, and ependymomas."
        )
    elif classNo == 2:
        return (
            "Meningioma Tumor"
            " - Meningioma is a type of tumor that grows from the protective membranes, called meninges, which surround the brain and spinal cord. Meningiomas occur most commonly in older women."
        )
    elif classNo == 3:
        return (
            "Pituitary Tumor"
            " - Pituitary tumors are growths that develop in your pituitary gland. They can be noncancerous (benign) or cancerous (malignant). Pituitary tumors are generally divided into three categories based on their size: microadenomas, macroadenomas, and giant adenomas."
        )

def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=np.argmax(model.predict(input_img), axis=1)  
    return result

@app.route('/')
def index():
    # Fetch doctor information from MongoDB collection
    doctors = doctors_collection.find()
    return render_template('index.html', doctors=doctors)

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value) 
        return result
    return "Prediction failed."

@app.route('/admin')
def admin():
    # Fetch all doctors from MongoDB collection
    doctors = doctors_collection.find()
    return render_template('admin.html', doctors=doctors)

@app.route('/add_doctor', methods=['POST'])
def add_doctor():
    if request.method == 'POST':
        name = request.form['name']
        address = request.form['address']
        doctors_collection.insert_one({'name': name, 'address': address})
        return redirect('/admin')

@app.route('/update/<string:doctor_id>', methods=['GET', 'POST'])
def update_doctor(doctor_id):
    if request.method == 'POST':
        name = request.form['name']
        address = request.form['address']
        doctors_collection.update_one({'_id': ObjectId(doctor_id)}, {'$set': {'name': name, 'address': address}})
        return redirect('/admin')
    else:
        doctor = doctors_collection.find_one({'_id': ObjectId(doctor_id)})
        return render_template('update.html', doctor=doctor)

@app.route('/delete/<string:doctor_id>')
def delete_doctor(doctor_id):
    doctors_collection.delete_one({'_id': ObjectId(doctor_id)})
    return redirect('/admin')

if __name__ == '__main__':
    app.run(debug=True)
