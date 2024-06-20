import streamlit as st
import os
import fitz
import os
from PIL import Image
import pandas as pd
from collections import Counter
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Function to check if the uploaded file is a PDF
def is_pdf(file):
    return file.type == 'application/pdf'

# Function to clear the upload directory
def clear_upload_dir(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            st.error(f"Error occurred while deleting {file_path}: {e}")

# Backend working of the UI
labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3:'Proliferate_DR', 4: 'Severe'}

#pdftoimage
def pdf(file_path, images_path):
    # Open PDF file
    pdf_file = fitz.open(file_path)
    # Get the number of pages in PDF file
    page_nums = len(pdf_file)
    # Create empty list to store images information
    images_list = []
    # Extract all images information from each page
    for page_num in range(page_nums):
        page_content = pdf_file[page_num]
        images_list.extend(page_content.get_images())
    # Raise error if PDF has no images
    if len(images_list)==0:
        raise ValueError(f'No images found in {file_path}')
    # Save all the extracted images
    for i, img in enumerate(images_list, start=1):
        # Extract the image object number
        xref = img[0]
        # Extract image
        base_image = pdf_file.extract_image(xref)
        # Store image bytes
        image_bytes = base_image['image']
        # Store image extension
        image_ext = base_image['ext']
        # Generate image file name
        image_name = str(i) + '.' + image_ext
        # Save image
        with open(os.path.join(images_path, image_name) , 'wb') as image_file:
            image_file.write(image_bytes)
            image_file.close()

def work():
    # Clearing image folder 
    dir_path = "images"
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        os.remove(file_path)
    file_path = 'upload/up.pdf'
    images_path = 'images/'
    pdf(file_path, images_path)
    if not os.listdir(dir_path):
        print("Directory is empty")
    else:
        image = []
        # Loop through all files in the directory
        for filename in os.listdir(dir_path):
            # Check if the file is an image (you can add more extensions if needed)
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                # Construct the full file path
                file_path = os.path.join(dir_path, filename)
                # Open the image
                img = Image.open(file_path)
                img = img.resize((256,256))
                # Appending image to the image list
                image.append(img)
        
        # Load the model with modified optimizer configuration
        # lm = tf.keras.models.load_model('saved_model/my_model.keras')
        lm=tf.keras.models.load_model('saved_model/my_model.keras', compile=False)
        # Compile the model with a new optimizer configuration
        optimizer = Adam(learning_rate=0.001)  # Customize the optimizer as needed
        lm.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        lab=[]
        for img in image:
            img = np.asarray(img, dtype= np.float32)
            # Normalizing the image
            img = img / 255
            # Reshaping the image into a 4D array
            img = img.reshape(-1,256,256,3)
            # Making prediction of the model
            predict = lm.predict(img)
            # Getting the index corresponding to the highest value in the prediction
            predict = np.argmax(predict)
            # Appending the predicted class to the list
            lab.append(labels[predict])
        
        counts = Counter(lab)
        # Find the string with the maximum count
        max_occuring_string = max(counts, key=counts.get)
        # Print the maximum occurring string
        return(max_occuring_string)


#switch case for description
def sw(res):
    if res == "Mild":
        s = "For mild cases, you may advise regular eye check-ups and monitoring of blood sugar levels. It's essential to manage diabetes effectively through lifestyle changes, medication adherence, and possibly insulin therapy as prescribed by a healthcare professional."
        e = "😦"  # Emoji for the "Mild" case
        return(s,e)
    elif res == "Moderate":
        s = "Moderate cases may require more frequent monitoring and treatment adjustments. In addition to managing diabetes, you might recommend lifestyle modifications such as diet and exercise to help control blood sugar levels. Consultation with an ophthalmologist for further evaluation and treatment options is also advisable."
        e = "😓"  # Emoji for the "Moderate" case
        return(s,e)
    elif res == "No_DR":
        s = "In cases where diabetic retinopathy is not detected, it's still crucial to continue regular eye check-ups and maintain good control of diabetes. Emphasize the importance of lifestyle modifications, medication adherence, and routine medical care to prevent the development of diabetic retinopathy and other complications."
        e = "🥸"  # Emoji for the "No_DR" case
        return(s,e)
    elif res == "Proliferate_DR":
        s = "Proliferative diabetic retinopathy requires prompt intervention to prevent vision loss and further complications. Treatment options may include laser photocoagulation, intraocular injections, or vitrectomy surgery. Urgent referral to an ophthalmologist or retina specialist is necessary for comprehensive evaluation and management."
        e = "😱"  # Emoji for the "Proliferate_DR" case
        return(s,e)
    elif res == "Severe":
        s = "Severe cases of diabetic retinopathy demand immediate attention and intervention to preserve vision and prevent blindness. Treatment modalities may include laser therapy, intravitreal injections, or surgical procedures such as vitrectomy. Close collaboration with healthcare providers, including ophthalmologists and endocrinologists, is essential for optimal management and monitoring of the condition."
        e = "😵"  # Emoji for the "Severe" case
        return(s,e)



# Streamlit app
def main():
    st.set_page_config(
    page_title = 'RETINOPATHY DETECTION',
    )

    st.header("TIMELY DETECTION OF DIABETIC RETINOPATHY")
    st.title("PDF File Dropzone")

    st.write("Drag and drop a PDF file here or click to browse.")
    st.markdown('<img src="https://media3.giphy.com/media/3o6fIPKZvhaCEZ7xTi/giphy.gif?cid=ecf05e47gtuzj0zzqlnrne6szupykr4cd7rkxnf6j5mcnupk&ep=v1_gifs_search&rid=giphy.gif&ct=g" width="300" height="300">', unsafe_allow_html=True)
    # Create a dropzone for file upload
    uploaded_file = st.file_uploader("", type=['pdf'], accept_multiple_files=False, key="file_uploader")

    # Define the folder to save the PDF
    folder_path = "upload"

    # Check if a file was uploaded
    if uploaded_file is not None:
        if is_pdf(uploaded_file):
            st.success("File successfully uploaded!")
            # Clear the upload directory
            clear_upload_dir(folder_path)
            # Get the file name
            filename = uploaded_file.name
            # Save the PDF file with a new name
            new_file_path = os.path.join(folder_path, "up.pdf")
            with open(new_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.write("Filename:", filename)

            res=work()
            st.subheader("Prediction is:")
            st.subheader(res)
            s,e=sw(res)
            st.subheader("Description:")
            st.write(s)
            st.markdown(f'<span style="font-size:50px">{e}</span>', unsafe_allow_html=True)
        else:
            st.error("Please upload a valid PDF file.")

if __name__ == "__main__":
    main()
