# Robotic Manipulation with Deep Learning

<img src="robot_pick_and_place.gif" width="900" height="450" />

### Full Demo Link

[![Watch the video](https://img.youtube.com/vi/NvVBQbD16_E/maxresdefault.jpg)](https://youtu.be/NvVBQbD16_E)

Welcome to the Robotic Manipulation project! This project demonstrates an advanced robotic pick and place system leveraging Deep Learning techniques, Simulation data, and Differential Inverse Kinematics Controller. The goal is to achieve precise object detection, segmentation, and manipulation in a simulated environment.

## Project Overview

This project involves the following key components:
- **Deep Learning**: Utilizes Mask R-CNN for object segmentation.
- **Simulation Data**: Uses simulation datasets for training and testing the model.
- **3D Transformations**: Handles 3D transformations and point cloud filtering.
- **Differential Inverse Kinematics**: Implements IK for robotic movements.
- **Antipodal Grasps**: Ensures secure and efficient object grasping.

### Key Features

- **Object Detection and Segmentation**: Fine-tuned Mask R-CNN model on simulation dataset to accurately detect and segment objects.
- **3D Point Cloud Filtering**: Processes and filters point clouds to identify and manipulate specified objects.
- **Inverse Kinematics Controller**: Utilizes IK for smooth and precise robotic arm movements.
- **Simulation Environment**: All operations are performed within the Drake simulation software.


