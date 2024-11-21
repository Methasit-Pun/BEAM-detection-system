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
   git clone https://github.com/your-username/beam-detection-system.git
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


