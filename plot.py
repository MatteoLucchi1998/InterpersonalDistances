'''
In this file are defined functions for drawing Bird Eye View of ROI, bounding boxes and distancing lines.
'''
'''
Questo file contiene le funzioni per il disegno della Vista a Volo d'Uccello della ROI e delle bounding boxes.
'''

# imports
import cv2
import numpy as np
import math

# Function to draw Bird Eye View for region of interest(ROI). Red or Green.
# Red: Risk
# Green: Safe
def bird_eye_view(frame, distances_mat, bottom_points, scale_w, scale_h, risk_count, rotation_matrix):
    h = frame.shape[0]
    w = frame.shape[1]

    red = (0, 0, 255)
    green = (0, 255, 0)

    blank_image = cv2.warpPerspective(frame, rotation_matrix, (w, h))

    warped_pts = []
    r = []
    g = []
    y = []
    for i in range(len(distances_mat)):

        if distances_mat[i][2] == 0:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                r.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                r.append(distances_mat[i][1])

            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), red, 1)
            
    for i in range(len(distances_mat)):
                
        if distances_mat[i][2] == 1:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                y.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                y.append(distances_mat[i][1])
        
            blank_image = cv2.line(blank_image, (int(distances_mat[i][0][0] * scale_w), int(distances_mat[i][0][1] * scale_h)), (int(distances_mat[i][1][0] * scale_w), int(distances_mat[i][1][1]* scale_h)), yellow, 1)
            
    for i in range(len(distances_mat)):
        
        if distances_mat[i][2] == 2:
            if (distances_mat[i][0] not in r) and (distances_mat[i][0] not in g) and (distances_mat[i][0] not in y):
                g.append(distances_mat[i][0])
            if (distances_mat[i][1] not in r) and (distances_mat[i][1] not in g) and (distances_mat[i][1] not in y):
                g.append(distances_mat[i][1])
    
    for i in bottom_points:
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, green, 5)

    for i in r:
        blank_image = cv2.circle(blank_image, (int(i[0]  * scale_w), int(i[1] * scale_h)), 5, red, 5)
        
    return blank_image
    
# Function to draw bounding boxes according to risk factor for humans in a frame and draw lines between
# boxes according to risk factor between two humans.
# Red: Risk
# Green: Safe 
def social_distancing_view(frame, distances_mat, boxes, confidences, classIDs, risk_count):
    
    red = (0, 0, 255)
    green = (0, 255, 0)
    
    for i in range(len(boxes)):

        x,y,w,h = boxes[i][:]
        
        #################################
        name = str(i)
        
        (sx,sy), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_PLAIN, 0.75, 1)
        
        frame = cv2.rectangle(frame, (x,y), (x+w, y+h+11), green, 1)
        frame = cv2.putText(frame, name, (x,y+baseline+6), cv2.FONT_HERSHEY_PLAIN, 0.75, (0,0,0), lineType=cv2.LINE_AA)
        frame = cv2.putText(frame, f'{confidences[i]:.0%}', (x,y+baseline+12+h), cv2.FONT_HERSHEY_PLAIN, 0.75, (0,0,0), lineType=cv2.LINE_AA)
        ################################
            
    for i in range(len(distances_mat)):

        per1 = distances_mat[i][0]
        per2 = distances_mat[i][1]
        closeness = distances_mat[i][2]
        
        if closeness == 0:
            x,y,w,h = per1[:]
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h+11), red, 1)
                
            x1,y1,w1,h1 = per2[:]
            frame = cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1+11), red, 1)

            frame = cv2.line(frame, (int(x+w/2), int(y+h/2)), (int(x1+w1/2), int(y1+h1/2)),red, 1)

            #Calculates distance between points - calcolo la distanza tra i due punti
            dist = math.sqrt((int(x1+w1/2) - (int(x+w/2)))**2 + (int(y1+h1/2) - int(y+h/2))**2)
            #Finds medium point in the line - trovo il punto medio della linea disegnata
            x_m_point = (int(x+w/2) + int(x1+w1/2))/2
            y_m_point = (int(x+w/2) + int(y+h/2))/2
            #cv2.putText(frame, str(dist), (int(x_m_point), int(y_m_point)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (66, 221, 245), 1)
            
    pad = np.full((140,frame.shape[1],3), [110, 110, 100], dtype=np.uint8)
    cv2.putText(pad, "Bounding box shows the level of risk to the person.", (50, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 0), 2)
    cv2.putText(pad, "-- RISK : " + str(risk_count[0]) + " people", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(pad, "-- SAFE : " + str(risk_count[2]) + " people", (50,  100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    frame = np.vstack((frame,pad))
            
    return frame

