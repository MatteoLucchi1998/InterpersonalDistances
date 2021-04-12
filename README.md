# InterpersonalDistances
During the COVID-19 pandemic it is necessary, to prevent contagion, to keep a safe distance from other individuals, in particular in public places. This project was created to provide a tool that, through Neural Networks and Artificial Vision, exploits images collected by surveillance video systems, producing useful information for respecting interpersonal distancing.

## Introduction
As a starting point for the development i used Social-Distancing-AI, an open source project from Deepak Birla.

The following functionalities were preserved:
* Directories organization;
* Source code organization;
* Basic functions;
* Deep Learning model.

The new functionalities added by me are:
* Adding a file slection tool for multiple input sources;
* Removing of the "Medium Risk";
* Different production of the Bird Eye View;
* Production of Heatmaps;
* Introduction of Ground Truth for exact measures;
* Increasing accuracy for measures;
* Production of more versatile data structures;

## Project Structure
* data: directory containing the input videos;
* output: directory containing the output frames, organized in sub-directories named as the input video;
* output_vid: directory containing the output videos, both Bird Eye View and ROI;
* models: you must add yolov3.weights , you can download them here: https://pjreddie.com/media/files/yolov3.weights;
* main.py: the main project file;
* utills.py, plot.py: functions files;
* heatmap.ipynb: Jupiter Notebook explaining a simple way to generate heatmaps;

## Human Recognition
The first functionality is the recognition of people and their position in the region using **YOLOv3**.
![alt text](https://github.com/[MatteoLucchi1998]/[InterpersonalDistances]/blob/[main]/1.PNG?raw=true)

## Production of Bird Eye View
By applying a prospective transformation on the **Region Of Interest** (selected with four mouse clicks) the **Bird Eye View** is produced, with horizontal and vertical ratios (obtained with three mouse clicks). Inside the Bird Eye View points are equally distributed, that means that we can claculate the distance between the position of people in the Bird Eye View.

## Calculation of Interpersonal Measures
Thanks to the Bird Eye View we can produce a **Distance Matrix**, containing the Euclidean Distance between every person in the Region of Interest.
If the distance between two person i greater than 1.5m they'll be indicated in **green**, otherwise, if any person is closer than 1.5m to someone else they'll be colored in **red**.

## Ground Truth
To appurate the accuracy of the measures some videos have been recorded, with calibration measures taken on the site.
That way i've been able to indicate for every frame:
* Time of the day;
* Type of Background (Homogeneous or Irregular);
* Number of Revealed People;
* Number of Not Revealed People;
* Number of People Wrongly Revealed;
* Number of People correctly marked as Safe;
* Number of People correctly marked as at Risk;
* Number of People wrongly marked as Safe;
* Number of People wrongly marked as at Risk;

## Production of Confusion Matrix

## Problems and Possible Solutions
The main cause of errors is the detection of humans, a solution can be the use of a different Neural Network, the re-training the current one or the use of filters.
The second and third causes are the **Darkness** of the frames (during nighttime) and the **Irregularity of the Background**. The first can also be solved using high contrast filters, the second instead can be reduced changing the point of view of the camera.

## Possible Evolutions
Associating *Timestamps* to the frames can improve the system behaviour during different time of the day.
The introduction of a **Gathering Threshold**, that indicates the number of people not respecting the interpersonal distances. Surpassing the threshold could produce an acoustic signal for the people to hear.
The introduction of **Human Tracking** can also improve the project, following a person's movement inside a network of cameras.
