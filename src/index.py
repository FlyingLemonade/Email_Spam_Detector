from flask import Flask, render_template, request
import os
import mailparser
import cv2 as cv
import numpy as np
from tensorflow.keras import models
from Email_Detector import email_detectoring
from ImageSpamRecon import spamRecon

app = Flask(__name__)

# Set up upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def parse_email_from_file(filepath):
    try:
        email = mailparser.parse_from_file(filepath)
        return email
    except Exception as e:
        return None
    
# def load_and_predict_image(img_path):
#     class_names = ['ham', 'spam']
    
#     # Load the pre-trained model
#     model = models.load_model('C:\\Users\\Lenovo\\Downloads\\ImageSpamRecon.keras')
    
#     # Load and preprocess the image
#     img = cv.imread(img_path)
#     img = cv.resize(img, (128, 128))
#     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#     img = np.array([img]) / 255.0  # Normalize the image
    
#     # Perform prediction
#     prediction = model.predict(img)
    
#     # Interpret prediction
#     if prediction[0] > 0.5:
#         return "Spam"
#     else:
#         return "Ham"
        
@app.route("/", methods=['GET', 'POST'])
def home():
    email_result = None
    photo_result = None
    if request.method == 'POST':
        # Check if the post request has the file part
        # if 'email_file' not in request.files:
        #     return 'No file part'
        
        # if 'foto_file' not in request.files:
        #     return 'No file part'
        
        file = request.files['email_file']
        
        foto = request.files['foto_file']
        
        email = request.form.get('email_text')
       
        # If user does not select file, browser also submit an empty part without filename
        # if file.filename == '':
        #     return 'No selected file'
        if foto :
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], foto.filename)
            foto.save(filepath)
            print(filepath)
            photo_result = spamRecon(img_path=filepath)
            return render_template("Proyek.html", email_result=email_result, photo_result=photo_result)
        
        
            # Call the image detection function from ImageSpamRecon.py
            # photo_result = spamRecon(foto)
            # if photo_result:
            #     return render_template("Proyek.html", email_result=email_result, photo_result=photo_result)
            # else:
            #     return 'Error processing email or file not found.'
            
        if email :
            # Call the email detection function from Email_Detector.py
            email_result = email_detectoring(email)
            
            if email_result:
                return render_template("Proyek.html", email_result=email_result, photo_result=photo_result)
            else:
                return 'Error processing email or file not found.'
        # else:
        #     return 'Error parsing email'  

        if file :
            # Save the file to the uploads directory
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            # Parse email from the uploaded file
            parsed_email = parse_email_from_file(filepath)
            
            if parsed_email:
                # Manipulate or process parsed email content here
                text_email = ''.join(parsed_email.text_plain)
                
                # Call the email detection function from Email_Detector.py
                email_result = email_detectoring(text_email)
                
                if email_result:
                    return render_template("Proyek.html", email_result=email_result, photo_result=photo_result)
                else:
                    return 'Error processing email or file not found.'
            else:
                return 'Error parsing email'
    
    return render_template("Proyek.html")

if __name__ == "__main__":
    app.run(debug=True)

