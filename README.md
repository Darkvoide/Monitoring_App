🎓 AI Campus Monitoring & Automatic Attendance System

An AI-powered Campus Monitoring and Automatic Attendance System that uses **Face Recognition** to automatically identify students and staff in real time. The system records attendance, maintains detection logs, and provides a user-friendly desktop interface built with Tkinter.


📌 Features

- 🔍 Real-time face detection and recognition
- 👨‍🎓 Student and Staff identification
- 📝 Automatic attendance logging
- 👤 Register new faces with Name, Role, and Class
- 📷 Live webcam monitoring
- 💾 Face embedding database storage
- 📊 Attendance records stored in CSV format
- 🖥️ Simple and interactive Tkinter GUI
- ⚡ Fast face recognition using DeepFace


🛠️ Technologies Used

- Python
- OpenCV
- DeepFace
- YOLOv8
- TensorFlow
- NumPy
- Pillow (PIL)
- Tkinter
- Pickle
- CSV


📂 Project Structure

Monitoring_App/
│
├── app.py                     # Main application
├── attendance.csv             # Attendance records
├── detection_logs.csv         # Detection logs
├── requirements.txt           # Required Python packages
├── yolov8n.pt                 # YOLOv8 model
│
├── data/
│   └── registered_faces.pkl   # Registered face embeddings
│
├── models/
│   ├── gender_deploy.prototxt
│   └── gender_net.caffemodel
│
└── README.md
```


🖥️ How It Works

Step 1 – Register Users

- Enter Name
- Select Role (Student/Staff)
- Enter Class or Department
- Capture the person's face
- Save the facial embedding

Step 2 – Start Live Monitoring

- Open the webcam
- Detect faces in real time
- Compare detected faces with the registered database
- Identify the person
- Mark attendance automatically
- Store the attendance record in CSV


📊 Attendance Format

Attendance is stored in:

```
attendance.csv
```

Example:

| Timestamp | Name | Role | Class |
|-----------|------|------|-------|
| 2026-07-06 10:30 | John | Student | AI & DS |


📁 Face Database

Registered faces are stored as embeddings inside

```
data/registered_faces.pkl
```

This allows quick face matching without storing multiple images.


🚀 Future Improvements

- Email alert system
- SMS notification system
- Stranger detection
- Visitor management
- Multi-camera support
- Cloud database integration
- Admin dashboard
- Attendance analytics
- Face anti-spoofing
- Improved recognition accuracy using ArcFace or FaceNet


📷 Screenshots

Add screenshots of:

- Registration Window
- Live Monitoring
- Face Detection
- Attendance Log
- Detection Results


🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create a feature branch
```bash
git checkout -b feature-name

3. Commit your changes
```bash
git commit -m "Added new feature"

4. Push to GitHub
```bash
git push origin feature-name

5. Open a Pull Request



📄 License

This project is intended for educational and research purposes.


👨‍💻 Author

**Mohandas S**

Engineering Student – Artificial Intelligence & Data Science

GitHub: https://github.com/mrmohandas143

⭐ If you found this project useful, consider giving it a Star on GitHub!
