---

# Beam Detection System: Enhancing E-Scooter Safety at Chulalongkorn University

The **Beam Detection System** is a safety solution designed to monitor and prevent accidents caused by overcrowded e-scooter rides at **Chulalongkorn University**. This system uses advanced **machine learning-based object detection** to detect the number of people riding an e-scooter. When two or more people are detected on a single e-scooter, the system triggers an audible alert, promoting safety and raising awareness about potential risks, ultimately aiming to reduce accident rates on campus.

## Overview

E-scooters are a popular mode of transportation at Chulalongkorn University, providing a convenient way for students to travel around campus. However, a common issue arises when two or more people ride on a single e-scooter, which significantly increases the risk of accidents. The **Beam Detection System** aims to address this issue by using **real-time video surveillance** powered by **machine learning** to monitor and detect unsafe e-scooter usage.
When the system detects multiple riders on a single e-scooter, an audible sound is triggered as an alert, allowing riders to become aware of the safety violation and take corrective actions.
## Features

- **Real-time Object Detection:** Uses machine learning models to identify e-scooters and the number of riders in real-time.
- **Instant Alerts:** Emits a sound alert when two or more riders are detected on a single e-scooter, ensuring immediate awareness.
- **AI-Powered Monitoring:** Powered by object detection models that can analyze live video feeds from cameras installed on campus.
- **Improved Campus Safety:** Helps reduce accidents caused by overcrowded e-scooters, promoting a safer environment for students and staff.

## How It Works

The Beam Detection System processes live video feeds captured by cameras installed across the campus. It uses **machine learning models** to detect e-scooters and identify how many riders are on each scooter. When multiple riders are detected, the system emits a sound to alert nearby individuals about the safety violation.

### Steps:

1. **Live Video Feed:** The system uses cameras connected to a processing unit to capture real-time footage of campus areas where e-scooters are commonly used.
2. **Object Detection:** Using machine learning models, the system analyzes the video feed to identify e-scooters and detect the number of people riding each one.
3. **Trigger Alert:** If two or more riders are detected on a single e-scooter, an alert sound is triggered to notify people nearby.
4. **Continuous Monitoring:** The system continuously monitors the campus for safety violations, providing real-time alerts whenever needed.

## Technologies

- **Machine Learning Framework:** TensorFlow or PyTorch for training and deploying object detection models.
- **Computer Vision:** OpenCV for handling video streams and detecting objects in real-time.
- **Object Detection Model:** A pre-trained model such as **YOLO** (You Only Look Once) or **SSD** (Single Shot Multibox Detector) for real-time detection of e-scooters and riders.
- **Audio Alert System:** A simple sound module integrated with the detection system to trigger audible alerts.

## Related Resources

To understand how similar object detection and machine learning technologies are implemented, you can refer to the following:

- [Machine Learning Video Detection Demo](https://www.youtube.com/watch?v=2XMkPW_sIGg): Learn how machine learning models can detect objects in video feeds, which is the foundation for our system's object detection.

- **Research Paper and Proposal**: [Deep Vision Overview](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg)

---

## Installation and Setup

1. **Clone the Repository**:  
   First, clone the repository to your local machine:

   ```bash
   git clone https://github.com/Methasit-Pun/BEAM-detection-system.git
   cd beam-detection-system
   ```

2. **Install Dependencies**:  
   The system requires the following dependencies:
   - **Python** 3.x
   - **OpenCV** for computer vision
   - **TensorFlow** or **PyTorch** for machine learning
   - Additional Python libraries as specified in the `requirements.txt` file

   Install the dependencies with:

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Cameras**:  
   Set up cameras in key campus locations where e-scooter traffic is high. The system supports multiple camera types, and you can select the input source for your feed in the configuration settings.

4. **Run the System**:  
   To start the live object detection, simply run the following:

   ```bash
   python detect_e_scooter.py
   ```

   The system will start processing the video feed, and any violations will trigger an audio alert.

## Future Vision

Our goal is to expand the **Beam Detection System** to other campuses across Thailand, creating a **nationwide network** of e-scooter safety monitoring. This system will not only reduce accident rates but also raise awareness about the importance of safety in campus transportation.

**Planned future enhancements include:**
- Integration with mobile apps for real-time alerts sent to students’ phones.
- Improved detection algorithms to better handle diverse environments and scooter types.
- Expansion of coverage areas on campus with additional cameras and sensors.

## Contributing

We welcome contributions from the community to improve the Beam Detection System. If you have suggestions, bug fixes, or new features to propose, feel free to open a pull request.

### How to Contribute:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Create a pull request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

---

Reference Video for Image detection by Machine Learning
https://www.youtube.com/watch?v=2XMkPW_sIGg


Research Paper and Proposal 



<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg" width="100%">
<p align="right"><sup><a href="pytorch-plants.md">Back</a> | <a href="../README.md#webapp-frameworks">Next</a> | </sup><a href="../README.md#hello-ai-world"><sup>Contents</sup></a>
<br/>
<sup>Transfer Learning - Object Detection</sup></s></p>

# Collecting your own Detection Datasets

The previously used `camera-capture` tool can also label object detection datasets from live video:

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-collection-detect.jpg" >

When the `Dataset Type` drop-down is in Detection mode, the tool creates datasets in [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) format (which is supported during training).

> **note:** if you want to label a set of images that you already have (as opposed to capturing them from camera), try using a tool like [`CVAT`](https://github.com/openvinotoolkit/cvat) and export the dataset in Pascal VOC format.  Then create a labels.txt in the dataset with the names of each of your object classes.

## Creating the Label File

Under `jetson-inference/python/training/detection/ssd/data`, create an empty directory for storing your dataset and a text file that will define the class labels (usually called `labels.txt`).  The label file contains one class label per line, for example:

``` bash
Water
Nalgene
Coke
Diet Coke
Ginger ale
```

If you're using the container, you'll want to store your dataset in a [Mounted Directory](aux-docker.md#mounted-data-volumes) like above, so it's saved after the container shuts down.

## Launching the Tool

The `camera-capture` tool accepts the same input URI's on the command line that are found on the [Camera Streaming and Multimedia](aux-streaming.md#sequences) page. 

Below are some example commands for launching the tool:

``` bash
$ camera-capture csi://0       # using default MIPI CSI camera
$ camera-capture /dev/video0   # using V4L2 camera /dev/video0
```

> **note**:  for example cameras to use, see these sections of the Jetson Wiki: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Nano:&nbsp;&nbsp;[`https://eLinux.org/Jetson_Nano#Cameras`](https://elinux.org/Jetson_Nano#Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- Xavier:  [`https://eLinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras`](https://elinux.org/Jetson_AGX_Xavier#Ecosystem_Products_.26_Cameras) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- TX1/TX2:  developer kits include an onboard MIPI CSI sensor module (0V5693)<br/>

## Collecting Data

Below is the `Data Capture Control` window, after the `Dataset Type` drop-down has been set to Detection mode (do this first).

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/pytorch-collection-detection-widget.jpg" >

Then, open the dataset path and class labels that you created.  The `Freeze/Edit` and `Save` buttons will then become active. 

Position the camera at the object(s) in your scene, and click the `Freeze/Edit` button (or press the spacebar).  The live camera view will then be 'frozen' and you will be able to draw bounding boxes over the objects.  You can then select the appropriate object class for each bounding box in the grid table in the control window.  When you are done labeling the image, click the depressed `Freeze/Edit` button again to save the data and unfreeze the camera view for the next image.

Other widgets in the control window include:

* `Save on Unfreeze` - automatically save the data when `Freeze/Edit` is unfreezed
* `Clear on Unfreeze` - automatically remove the previous bounding boxes on unfreeze
* `Merge Sets` - save the same data across the train, val, and test sets
* `Current Set` - select from train/val/test sets
    * for object detection, you need at least train and test sets
    * although if you check `Merge Sets`, the data will be replicated as train, val, and test
* `JPEG Quality` - control the encoding quality and disk size of the saved images

It's important that your data is collected from varying object orientations, camera viewpoints, lighting conditions, and ideally with different backgrounds to create a model that is robust to noise and changes in environment.  If you find that you're model isn't performing as well as you'd like, try adding more training data and playing around with the conditions.

## Training your Model

When you've collected a bunch of data, then you can try training a model on it using the same `train_ssd.py` script.  The training process is the same as the previous example, with the exception that the `--dataset-type=voc` and `--data=<PATH>` arguments should be set:

```bash
$ cd jetson-inference/python/training/detection/ssd
$ python3 train_ssd.py --dataset-type=voc --data=data/<YOUR-DATASET> --model-dir=models/<YOUR-MODEL>
```

> **note:** if you run out of memory or your process is "killed" during training, try [Mounting SWAP](pytorch-transfer-learning.md#mounting-swap) and [Disabling the Desktop GUI](pytorch-transfer-learning.md#disabling-the-desktop-gui). <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; to save memory, you can also reduce the `--batch-size` (default 4) and `--workers` (default 2)
  
Like before, after training you'll need to convert your PyTorch model to ONNX:

```bash
$ python3 onnx_export.py --model-dir=models/<YOUR-MODEL>
```

The converted model will then be saved under `<YOUR-MODEL>/ssd-mobilenet.onnx`, which you can then load with the `detectnet` programs like we did in the previous examples:

```bash
NET=models/<YOUR-MODEL>

detectnet --model=$NET/ssd-mobilenet.onnx --labels=$NET/labels.txt \
          --input-blob=input_0 --output-cvg=scores --output-bbox=boxes \
            csi://0
```

> **note:** it's important to run inference with the labels file that gets generated to your model directory, and not the one that you originally created for your dataset.  This is because a `BACKGROUND` class gets added to the class labels by `train_ssd.py` and saved to the model directory (which the trained model expects to use).

If you need to, go back and collect more training data and re-train your model again.  You can restart again and pick up where you left off using the `--resume` argument (run `python3 train_ssd.py --help` for more info).  Remember to re-export the model to ONNX after re-training.

<p align="right">Next | <b><a href="../README.md#webapp-frameworks">WebApp Frameworks</a></b>
<br/>
Back | <b><a href="pytorch-ssd.md">Re-training SSD-Mobilenet</a></p>
</b><p align="center"><sup>© 2016-2020 NVIDIA | </sup><a href="../README.md#hello-ai-world"><sup>Table of Contents</sup></a></p>


