# Face Detection and Recognition System

## **About the Project**
This project aims to develop a **face detection and recognition system** utilizing the **TensorFlow** framework alongside other essential libraries. The system is designed to **detect and recognize multiple faces simultaneously** in real-time scenarios.

## **Expected Outcome**
The model is expected to:
- **Detect** multiple faces in an image or video feed.
- **Recognize** and identify each detected face based on pre-trained data.

## **Project Requirements**
To achieve the desired functionality, the following steps are implemented:

- Utilize a **pre-trained face detection network** to identify faces in images or videos.
- Employ a **classification network** to recognize and classify the detected faces.
- Ensure the system operates in **real-time**, handling multiple face detections and recognitions concurrently.

## **Reference Materials**
For implementation guidance, consider the following resources:

- **Face Detection:** [Colab Notebook](https://colab.research.google.com/drive/1QnC7lV7oVFk5OZCm75fqbLAfD9qBy9bw?usp=sharing)
- **Object Detection and Classification:** [Colab Notebook](https://colab.research.google.com/drive/1xdjyBiY75MAVRSjgmiqI7pbRLn58VrbE?usp=sharing)
- **YOLOv3 Configuration:** [Darknet](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg)
- **YOLOv3 Weights:** [Darknet](https://pjreddie.com/media/files/yolov3.weights)
- **FaceNet ONNX Model:** [Github](https://github.com/9bd0f0ac-4525-47ac-880a-0366f82f77b9)

---

## **Project Structure**


📁 data/               # Contains face.cfg, face.weights, and facenet.onnx

📁 scripts/            # Python scripts for automation and inference

📁 known_faces/        # Directory for storing known face images

📄 README.md           # Project documentation (this file)

📄 requirements.txt    # List of required Python libraries

## **Installation & Setup**
To set up the project, follow these steps:

### **Clone the Repository**

```bash
git clone https://github.com/your-repository/Face-Detection-Recognition.git
cd Face-Detection-Recognition
```

### **Install Dependencies**
Ensure you have the required libraries by installing them via requirements.txt:

```bash
pip install -r requirements.txt
```

### **Download Necessary Files**
Before running the application, download the following files and place them in the data/ directory:

FaceNet ONNX Model: [facenet.onnx](https://drive.google.com/file/d/17uuOJu_vSXu6rYm6C4IUUzi0ORv5XBP1/view?usp=sharing)

YOLOv3 Weights: [face.weights](https://drive.google.com/file/d/1F5JgoL2a6P_EUCZLcV9xP_vckLcJILqn/view?usp=sharing)

YOLOv3 Configuration: [face.cfg](https://drive.google.com/file/d/19Ryii_Y8N7NTdiV0QwJsaru41PF-JhGv/view?usp=sharing)

Ensure these files are correctly placed before proceeding.

### **Adding Known Faces for Recognition**
To enable the system to recognize and identify faces, place images of known individuals in the known_faces directory. Each image should represent a person whose face you want the system to recognize based on the pre-trained model.

#### **Important**: The name of each image file (without the extension) will be used as the label for recognition during live camera capture. For example, an image named john_doe.jpg will result in the system identifying the detected face as "John Doe". 

### **Run the Application**
Execute the main script to start the face detection and recognition system:

```bash
python ./scripts/script.py
```

## **Implementation Details**

- **Face Detection**
The system employs the YOLOv3 (You Only Look Once) model for face detection. YOLOv3 is renowned for its speed and accuracy in object detection tasks. The model processes images in real-time, making it suitable for applications requiring prompt responses. For more details, refer to the official YOLO website.

- **Face Recognition (Classification)**
For face recognition, the system utilizes the FaceNet model, which converts facial images into a compact Euclidean space embedding. This allows for efficient and accurate face comparisons. The ONNX version of FaceNet ensures compatibility across different platforms.

- **Real-time Face Detection**
The integration of OpenCV with TensorFlow facilitates real-time face detection and recognition. The system captures video feed from a webcam, detects faces using YOLOv3, and recognizes them using FaceNet embeddings, all in real-time.

## **Technologies Used**
- **TensorFlow/Keras**: Deep Learning framework for building and training models.
- **OpenCV**: Library for image processing and computer vision tasks.
- **NumPy**: Library for numerical operations in Python.
- **Scikit-learn**: Machine learning library for Python, used here for implementing classifiers like K-Nearest Neighbors (KNN) and Support Vector Machine (SVM).
- **ONNX Runtime**: Cross-platform, high-performance scoring engine for Open Neural Network Exchange (ONNX) models.

## **Contributions**
If you want to contribute, follow these steps:

**Fork** the repository.  
**Create a new branch**:
```bash
git checkout -b feature-new-detection
```
**Commit your changes**:
```bash
git commit -m "Added new face detection method"
```
**Push your changes**:
```bash
git push origin feature-new-detection
```
**Open a Pull Request**.

## **📜 License**
This project is licensed under the **MIT License**.