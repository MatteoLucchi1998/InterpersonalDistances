'''
The Region of Interest(ROI) is defined via the first four mouse clicks, prespective transformation is applied creating the Bird Eye View.
In the Bird Eye View points are uniformally distributed both horizontally and vertically.
The scale for horizontal and vertical scale is defined via the last three mouse clicks.
The human detection is obtained using YOLOv3, the bounding boxes are then trasformed into points, considering the bottom center.
Interpersonal distance is calculated from the Bird Eye View using Euclidean Distance.
'''
'''
La Regione di Interesse(ROI) viene definita con i primi quattro click del mouse, viene poi applicata una trasformazione prospettica, creando la Vista a Volo d'Uccello.
Nella Vista a Volo d'Uccello i punti dell'immagine sono uniformemente distribuiti orizzontalmente e verticalmente.
La scala orizzontale e verticale viene definita con gli ultimi tre click del mouse.
L'identificazione delle persone viene ottenuta con YOLOv3, i riquadri di contorno vengono poi trasformati in punti, considerando il centro del lato basso.
Le distanze interpersonali vengono poi calcolate usando la Distanza Euclidea sulla Vista a Volo d'Uccello.
'''
'''
This project was inspired, uses some lines of code and strategies from Deepak Birla's project "Social Distancing AI".
The original project license was "MIT", i'd like to mention:

Copyright (c) <year> <copyright holders>

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
'''

# imports
import sys
sys.path.append('/usr/include/opencv4/opencv2')
import cv2
import numpy as np
import time
import argparse
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import os
import shutil
# modules
import utills, plot

confid = 0.5    #confidedence threshold - soglia di confidenza
thresh = 0.5   
mouse_pts = []  #selected points - punti selezionati sul video


# Function to get points for Region of Interest(ROI) and distance scale. It will take 8 points on first frame using mouse click    
# event.
# First four points will define ROI where we want to moniter social distancing. Also these points should form parallel  
# lines in real world if seen from above(birds eye view).
# Next 3 points will define 1.5m(unit length) distance in horizontal and vertical direction and those should form parallel
# lines with ROI.
# Unit length we can take based on choice.
# Points should pe in pre-defined order - bottom-left, bottom-right, top-right, top-left,
# point 5 and 6 should form horizontal line and point 5 and 7 should form verticle line.
# Horizontal and vertical scale will be different. 

# Function will be called on mouse events                                                      

def get_mouse_points(event, x, y, flags, param):

    global mouse_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_pts) < 4:  #this points define ROI ---> red - punti che definiscono la ROI ---> rosso
            cv2.circle(image, (x, y), 2, (0, 0, 255), 5)
        else:                   #this points define 1.5m distances ---> blue - punti che definiscono le distanze ---> blu
            cv2.circle(image, (x, y), 2, (255, 0, 0), 5)
            
        if len(mouse_pts) >= 1 and len(mouse_pts) <= 3:     #draw ROI boundaries - disegno contorni della ROI
            cv2.line(image, (x, y), (mouse_pts[len(mouse_pts)-1][0], mouse_pts[len(mouse_pts)-1][1]), (70, 70, 70), 2)
            if len(mouse_pts) == 3:                         #closes the square - chiudo il rettangolo
                cv2.line(image, (x, y), (mouse_pts[0][0], mouse_pts[0][1]), (70, 70, 70), 2)
        
        #if no point is been declared - se nessun punto è stato dichiarato
        if "mouse_pts" not in globals():
            mouse_pts = []
        mouse_pts.append((x, y))

def calculate_social_distancing(vid_path, net, output_dir, output_vid, ln1):
    
    count = 0
    vs = cv2.VideoCapture(vid_path)    

    # Get video height, width and fps
    height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vs.get(cv2.CAP_PROP_FPS))
    
    # Set scale for bird eye view
    # Bird eye view will only show ROI
    #Use utills library - Uso libreria utills
    scale_w, scale_h = utills.get_scale(width, height)

    #FourCC is a 4-byte code used to specify the video codec. List of codes can be obtained at Video Codecs by FourCC
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    output_movie = cv2.VideoWriter((output_video_folder + "distancing.avi"), fourcc, fps, (width, height))
    bird_movie = cv2.VideoWriter((output_video_folder + "bird_eye_view.avi"), fourcc, fps, (int(width * scale_w), int(height * scale_h)))

    points = []
    global image
    
    #main loop
    while True:

        #select a frame - prendo un frame
        (grabbed, frame) = vs.read()

        if not grabbed:
            print('here')
            break
            
        (H, W) = frame.shape[:2]
        
        # first frame will be used to draw ROI and horizontal and vertical 150 cm distance(unit length in both directions)
        if count == 0:
            while True:
                image = frame
                cv2.imshow("DrawROI", image)
                #wait event for a second - aspetta l'evento per 1 secondo
                cv2.waitKey(1)
                #closes windoes for ROI selection - chiudo la finestra per la scelta della ROI all'ottavo click
                if len(mouse_pts) == 8:
                    cv2.destroyWindow("DrawROI")
                    break
               
            points = mouse_pts      
                 
        # Using first 4 points or coordinates for perspective transformation. The region marked by these 4 points are 
        # considered ROI. This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view. 
        # This bird eye view then has the property that points are distributed uniformally horizontally and 
        # vertically(scale for horizontal and vertical direction will be different). So for bird eye view points are 
        # equally distributed, which was not case for normal view.

    ##############################################################################################################
        src = np.float32(np.array(points[:4]))
        dst = np.float32([[0, H], [W, H], [W, 0], [0, 0]])
        prespective_transform = cv2.getPerspectiveTransform(src, dst)

        # using next 3 points for horizontal and vertical unit length(in this case 150 cm)
        pts = np.float32(np.array([points[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, prespective_transform)[0]
        
        # since bird eye view has property that all points are equidistant in horizontal and vertical direction.
        # distance_w and distance_h will give us 150 cm distance in both horizontal and vertical directions
        # (how many pixels will be there in 150cm length in horizontal and vertical direction of birds eye view),
        # which we can use to calculate distance between two humans in transformed view or bird eye view
        distance_w = np.sqrt((warped_pt[0][0] - warped_pt[1][0]) ** 2 + (warped_pt[0][1] - warped_pt[1][1]) ** 2)
        distance_h = np.sqrt((warped_pt[0][0] - warped_pt[2][0]) ** 2 + (warped_pt[0][1] - warped_pt[2][1]) ** 2)
        pnts = np.array(points[:4], np.int32)
        #draw ROI - disegno la ROI
        cv2.polylines(frame, [pnts], True, (70, 70, 70), thickness=2)
    
    ####################################################################################
        #Human Recognition - Riconoscimento delle persone
        #Frame Preprocessing - Preprocessing del frame prima di darlo in pasto alla rete neurale.
        #Parameters for blobFromImage: immagine, fattore di scala, dimensione, mean-substraction, swapRB, crop.
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        #Blob is the neural network input - Blob è l'input della rete neurale
        net.setInput(blob)
        start = time.time()
        #Uses Neural Nerwork saving elapsed time - Utilizzo la neural network calcolando il tempo impiegato.
        layerOutputs = net.forward(ln1)
        end = time.time()
        boxes = []
        confidences = []
        classIDs = []
        for output in layerOutputs:
            #For every object an array with lenght 85 will be returned - Per ogni oggetto trovato viene restituito un np.array lungo 85
            #The first four elements are the bounding boxe parameters, the fifth is the confidence score - da 1 a 4 sono parametri della bounding box il quinto è il confidence score della box
            #The eighty other are the classes confidence scores - i restanti ottanta sono le confidece dell'oggetto per ognuna delle 80 classi
            for detection in output:
                scores = detection[5:]
                #Identify the highest ranking confidence score - Identifico l'oggetto con la classe con confidence score più alto
                classID = np.argmax(scores)
                confidence = scores[classID]
                #Class 0 identifies people - La classe 0 corrisponde agli oggetti identificati come persone.
                if classID == 0:
                    #If confidence is enough - Se mi trovo al di sopra del confidence score, nel nostro caso 0,5 considero valida la rilevazione
                    if confidence > confid:
                        #Select bounding box - individuo la bounding box
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        #Insert results in the respective np.arrays - Inserisco i risultati negli np.array corrispondenti.
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
                    
        #Soppression of non maximum - Soppressione dei non massimi
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, confid, thresh)
        font = cv2.FONT_HERSHEY_PLAIN
        boxes1 = []
        for i in range(len(boxes)):
            if i in idxs:
                boxes1.append(boxes[i])
                x,y,w,h = boxes[i]
        ################################################################################################
                
        if len(boxes1) == 0:
            count = count + 1
            continue
            
        # Here we will be using bottom center point of bounding box for all boxes and will transform all those
        # bottom center points to bird eye view
        person_points = utills.get_transformed_points(boxes1, prespective_transform)

        ################################################################################################
        # Here we will calculate distance between transformed points(humans)
        distances_mat, bxs_mat = utills.get_distances(boxes1, person_points, distance_w, distance_h)
        risk_count = utills.get_count(distances_mat)
    
        frame1 = np.copy(frame)
        
        # Draw bird eye view and frame with bouding boxes around humans according to risk factor    
        bird_image = plot.bird_eye_view(frame, distances_mat, person_points, scale_w, scale_h, risk_count, prespective_transform)
        img = plot.social_distancing_view(frame1, bxs_mat, boxes1, confidences, classIDs, risk_count)
        
        #################################################################################################

        # Show/write image and videos
        if count != 0:
            output_movie.write(img)
            bird_movie.write(bird_image)
    
            cv2.imshow('Bird Eye View', bird_image)
            cv2.imwrite(output_dir+"frame%d.jpg" % count, img)
            cv2.imwrite(output_dir+"bev%d.jpg" % count, bird_image)
    
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    vs.release()
    cv2.destroyAllWindows() 
        

if __name__== "__main__":

    #ask for file
    root = Tk()
    root.filename =  filedialog.askopenfilename(initialdir = "data/",title = "choose file",filetypes = (("mp4 files","*.mp4"),("all files","*.*")))
    selected_vid = root.filename
    print (selected_vid)
    root.withdraw()
    path_list = selected_vid.split(os.sep)
    print (path_list[len(path_list)-1])

    print("__________________________________________")
    print(path_list[len(path_list)-1])
    output_frames_folder = 'add your path \\data\\'
    if os.path.isdir(output_frames_folder) and os.path.exists(output_frames_folder):
        shutil.rmtree(output_frames_folder)
    print("-----------------------------------------------")
    print(output_frames_folder)
    os.mkdir(output_frames_folder)

    output_video_folder = 'add your path \\output\\'
    if os.path.isdir(output_video_folder) and os.path.exists(output_video_folder):
        shutil.rmtree(output_video_folder)
    os.mkdir(output_video_folder)
    

    # Receives arguements specified by user
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-v', '--video_path', action='store', dest='video_path', default=selected_vid ,
                    help='Path for input video')
                    
    parser.add_argument('-o', '--output_dir', action='store', dest='output_dir', default=output_frames_folder ,
                    help='Path for Output images')
    
    parser.add_argument('-O', '--output_vid', action='store', dest='output_vid', default=output_video_folder ,
                    help='Path for Output videos')

    parser.add_argument('-m', '--model', action='store', dest='model', default='./models/',
                    help='Path for models directory')
                    
    parser.add_argument('-u', '--uop', action='store', dest='uop', default='NO',
                    help='Use open pose or not (YES/NO)')
                    
    values = parser.parse_args()
    
    print("------------------------------------------------")
    model_path = values.model
    if model_path[len(model_path) - 1] != '/':
        model_path = model_path + '/'
        
    output_dir = values.output_dir
    if output_dir[len(output_dir) - 1] != '/':
        output_dir = output_dir + '/'
    
    output_vid = values.output_vid
    if output_vid[len(output_vid) - 1] != '/':
        output_vid = output_vid + '/'


    # load Yolov3 weights
    
    weightsPath = model_path + "yolov3.weights"
    configPath = model_path + "yolov3.cfg"

    #Reads a network model stored in Darknet model files.
    net_yl = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    #Get the name of all layers of the network.
    ln = net_yl.getLayerNames()
    ln1 = [ln[i[0] - 1] for i in net_yl.getUnconnectedOutLayers()]

    # set mouse callback 

    cv2.namedWindow('DrawROI',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("DrawROI", get_mouse_points)
    np.random.seed(42)
    
    calculate_social_distancing(values.video_path, net_yl, output_dir, output_vid, ln1)



