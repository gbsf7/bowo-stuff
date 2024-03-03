import sklearn
from sklearn.metrics import confusion_matrix , classification_report
import cv2 as cv
import cv2
import numpy
import datetime
import csv
import pandas
import matplotlib
import os
#Library OCR
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import segmentation_models as sm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from tensorflow.keras import optimizers
import seaborn as sns
import time

import os


from tensorflow.keras.models import load_model
#model_path = '/home/jetson/Documents/Skripsi Bowo/saved_model/cnn_ocr_vgg19_part2.pb'
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, input_shape=(3,)),
    tf.keras.layers.Softmax()])
model.save("my_model.h5")
loaded_model = tf.keras.models.load_model('my_model.h5')
                         
path_to_saved_model_dir = 'my_saved_model'
tf.saved_model.save(model,path_to_saved_model_dir)     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
# Variabel yang akan digunakan untuk deteksi plat nomor
plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Membuka akses webcam
cam = cv.VideoCapture(0)

# Mengatur ukuran window
cam.set(3, 640)
cam.set(4, 480)
cam.set(10, 150)

count = 1
minArea = 500
start_time = str(datetime.datetime.now())
times_list = []

# Membuat perulangan untuk menampilkan webcam
while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break

    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    numberPlates = plateCascade.detectMultiScale(frameGray, 1.1, 4)
    # Menambahkan timestamp
    datet = str(datetime.datetime.now())
    cv2.putText(frame, datet, (25, 25), cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2, cv.LINE_AA)
    
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            # Membuat box pada ROI
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Menambahkan teks plat
            cv.putText(frame,"NumberPlate",(x,y-5),cv.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            imgRoi = frame[y:y+h,x:x+w]
            cv.imshow("ROI", imgRoi)
            cv.imshow("Hasil", frame)

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break

    elif k & k%256 == 32: #Space untuk save image
        cv.imwrite('/Documents/Skripsi_Bowo/Data Timestamp\platetimestamp'+str(count)+".jpg"+str(count)+".jpg", frame)
        Gambarplat = cv.imwrite('/Documents/Skripsi_Bowo/Data_Plat\plate'+str(count)+".jpg", imgRoi)
        cv.rectangle(frame, (0,200), (640,300), (0,255,0), cv.FILLED)
        cv.putText(frame,"Gambar Tersimpan", (15,265), cv.FONT_HERSHEY_COMPLEX, 1.75, (0,0,255), 2)
        cv.imshow("Hasil", frame)
        cv.waitKey(500)
        count+=1
        times = str(datetime.datetime.now())
        times_list.append(times)
        
cam.release()
cv.destroyAllWindows()

# Match contours to license plate or character template
def find_contours(dimensions, img) :
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    # Find all contours in the image
    cntrs, _ = cv.findContours(img.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 5 or  15 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    
    ii = cv.imread('contour.jpg')#dont change the name
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs :
        #detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        #checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44,24))
            #extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY+intHeight, intX:intX+intWidth]
            char = cv2.resize(char, (20, 40))
            
            cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

#             Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[2:42, 2:22] = char
            char_copy[0:2, :] = 0
            char_copy[:, 0:2] = 0
            char_copy[42:44, :] = 0
            char_copy[:, 22:24] = 0

            img_res.append(char_copy) #List that stores the character's binary image (unsorted)
            
    #Return characters on ascending order with respect to the x-coordinate (most-left character first)
            
    plt.show()
    #arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res

# Find characters in the resulting images
def segment_characters(image) :

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[0]
    LP_HEIGHT = img_binary_lp.shape[1]

    # Make borders white
    img_binary_lp[0:3,:] = 255
    img_binary_lp[:,0:3] = 255
    img_binary_lp[72:75,:] = 255
    img_binary_lp[:,330:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/6,
                       LP_WIDTH/2,
                       LP_HEIGHT/16,
                       2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.show()
    cv2.imwrite('contour.jpg',img_binary_lp)#dont change the name

    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary_lp)

    return char_list

# import required module
import os
# assign directory
directory = '/Documents/Skripsi_Bowo/Data_Plat'
dataset_plat = []
# iterate over files in
# that directory
for filename in os.scandir(directory):
    if filename.is_file():
            dataset_plat.append(filename.path)
        
print(filename.path)


all_data = []

for img_path, timestamp in zip(dataset_plat, times_list):
    im = cv2.imread(img_path)
  
    print(img_path)
    if im is not None:
        
        char = segment_characters(im)
        def fix_dimension(img): 
            new_img = np.zeros((32,32,3))
            for i in range(3):
                new_img[:,:,i] = img
            return new_img
  
        def show_results():
            dic = {}
            characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            for i,c in enumerate(characters):
                dic[i] = c

            output = []
            for i,ch in enumerate(char): #iterating over the characters
                img_ = cv2.resize(ch, (32,32), interpolation=cv2.INTER_AREA)
                img = fix_dimension(img_)
                img = img.reshape(1,32,32,3) #preparing image for the model
                # y_ = model.predict(img)[0] #predicting the class
                y_ =np.argmax(model.predict(img), axis=1)[0] #predicting the class
                character = dic[y_] #
                output.append(character) #storing the result in a list
        
            plate_number = ''.join(output)
    
            return plate_number

        print(show_results())
        end_time = str(datetime.datetime.now())
        
        output_directory = '/Documents/Skripsi_Bowo/Output CSV'
        #Fungsi untuk mengambil informasi dari gambar dan mengembalikan data yang sesuai
        current_data = {
            'Gambar_Pelat': Gambarplat,
            'Timestamp': timestamp,  
            'Pelat_Nomor': show_results(),
            'Start_Time': start_time,
            'End_Time': end_time
        }

        all_data.append(current_data)


        df = pd.DataFrame(all_data)
        
        file_name = 'Hasil_Prediksi.csv'
        output_file_path = os.path.join(output_directory, file_name)
        #Save Dataframe to CSV
        
        df.to_csv(path_or_buf=output_file_path, mode='a', header =not os.path.exists(output_file_path), index =False)
        
        print("Data from images has been saved to Hasil_Prediksi.csv")
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
