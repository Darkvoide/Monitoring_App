import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import csv
import datetime
import pickle
from deepface import DeepFace

# Set environment variable to avoid tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Path to store registered encodings: { "name_role_class": embedding_vector }
FACES_DB_PATH = os.path.join(DATA_DIR, "registered_faces.pkl")

# Path to log attendance
ATTENDANCE_CSV_PATH = os.path.join(APP_DIR, "attendance.csv")

class AttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Automatic Attendance System")
        self.geometry("1100x750")
        self.configure(bg="#2c3e50")

        self.vid_capture = None
        self.is_running = False
        self.current_frame = None
        
        # UI Elements
        self.notebook = None
        self.tab_register = None
        self.tab_attendance = None
        
        self.entry_name = None
        self.combo_role = None
        self.entry_class = None
        self.lbl_reg_status = None
        self.lbl_reg_db_count = None
        
        self.btn_stop = None
        self.canvas = None
        self.canvas_image_id = None
        self.canvas_width = 800
        self.canvas_height = 600
        self.current_display_image = None
        
        self.log_text = None
        self.recent_logs = {}
        
        # Load registered faces db
        self.registered_faces = {}
        if os.path.exists(FACES_DB_PATH):
            try:
                with open(FACES_DB_PATH, 'rb') as f:
                    self.registered_faces = pickle.load(f)
            except Exception as e:
                print(f"Error loading faces DB: {e}")

        # Ensure CSV exists
        if not os.path.exists(ATTENDANCE_CSV_PATH):
            with open(ATTENDANCE_CSV_PATH, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Name", "Role", "Class/Course"])

        self.setup_ui()

    def setup_ui(self):
        # Create Notebook for Tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Tab 1: Registration
        self.tab_register = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_register, text="Register New Person")
        self.setup_registration_tab()

        # Tab 2: Live Attendance
        self.tab_attendance = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_attendance, text="Live Attendance Feed")
        self.setup_attendance_tab()

    # ================= REGISTRATION TAB ================= #
    def setup_registration_tab(self):
        frame = tk.Frame(self.tab_register, bg="#34495e")
        frame.pack(fill=tk.BOTH, expand=True)

        # Input fields
        input_frame = tk.Frame(frame, bg="#34495e", pady=20)
        input_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Label(input_frame, text="Name:", fg="white", bg="#34495e", font=("Helvetica", 12)).grid(row=0, column=0, padx=10, pady=5, sticky=tk.E)
        self.entry_name = tk.Entry(input_frame, font=("Helvetica", 12))
        self.entry_name.grid(row=0, column=1, padx=10, pady=5)

        tk.Label(input_frame, text="Role:", fg="white", bg="#34495e", font=("Helvetica", 12)).grid(row=1, column=0, padx=10, pady=5, sticky=tk.E)
        self.combo_role = ttk.Combobox(input_frame, values=["Student", "Teacher", "Staff"], font=("Helvetica", 12), state="readonly")
        self.combo_role.current(0)
        self.combo_role.grid(row=1, column=1, padx=10, pady=5)

        tk.Label(input_frame, text="Class/Course:", fg="white", bg="#34495e", font=("Helvetica", 12)).grid(row=2, column=0, padx=10, pady=5, sticky=tk.E)
        self.entry_class = tk.Entry(input_frame, font=("Helvetica", 12))
        self.entry_class.grid(row=2, column=1, padx=10, pady=5)

        # Buttons
        btn_frame = tk.Frame(frame, bg="#34495e")
        btn_frame.pack(side=tk.TOP, pady=10)
        
        ttk.Button(btn_frame, text="Select Image to Register", command=self.register_from_image).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Clear Database", command=self.clear_database).pack(side=tk.LEFT, padx=10)
        
        # Display area for registration status
        self.lbl_reg_status = tk.Label(frame, text="", fg="white", bg="#34495e", font=("Helvetica", 12))
        self.lbl_reg_status.pack(side=tk.TOP, pady=20)
        
        self.lbl_reg_db_count = tk.Label(frame, text=f"Total Registered Persons: {len(self.registered_faces)}", fg="#2ecc71", bg="#34495e", font=("Helvetica", 12, "bold"))
        self.lbl_reg_db_count.pack(side=tk.TOP, pady=10)

    def clear_database(self):
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to delete all registered faces and attendance logs? This cannot be undone."):
            self.registered_faces = {}
            if os.path.exists(FACES_DB_PATH):
                os.remove(FACES_DB_PATH)
            if os.path.exists(ATTENDANCE_CSV_PATH):
                os.remove(ATTENDANCE_CSV_PATH)
                # Recreate headers
                with open(ATTENDANCE_CSV_PATH, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Timestamp", "Name", "Role", "Class/Course"])
            
            self.lbl_reg_db_count.config(text=f"Total Registered Persons: 0")
            self.lbl_reg_status.config(text="Database cleared successfully.", fg="#2ecc71")
            
            if self.log_text:
                self.log_text.config(state=tk.NORMAL)
                self.log_text.delete(1.0, tk.END)
                self.log_text.config(state=tk.DISABLED)

    def register_from_image(self):
        name = self.entry_name.get().strip()
        role = self.combo_role.get().strip()
        course = self.entry_class.get().strip()

        if not name or not course:
            messagebox.showwarning("Incomplete Data", "Please provide Name and Class/Course before registering.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return

        self.lbl_reg_status.config(text="Processing image and extracting facial features...", fg="yellow")
        self.update()

        try:
            # First attempt with default OpenCV face detection
            try:
                # ArcFace provides higher accuracy for face recognition
                embedding_objs = DeepFace.represent(img_path=file_path, model_name="ArcFace", detector_backend="opencv", enforce_detection=True)
            except ValueError:
                # Fallback: If no face is strictly detected by Haar, force extraction on the whole image.
                # This is useful if the user provided an image that is already cropped tightly to their face.
                embedding_objs = DeepFace.represent(img_path=file_path, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
            
            if len(embedding_objs) == 0:
                self.lbl_reg_status.config(text="No facial features could be extracted.", fg="red")
                return
            elif len(embedding_objs) > 1:
                self.lbl_reg_status.config(text="Multiple faces detected. Please use a photo with only ONE person.", fg="red")
                return
                
            embedding = embedding_objs[0]["embedding"]
            
            # Create a unique key
            unique_key = f"{name}_{role}_{course}"
            
            # Save to DB
            self.registered_faces[unique_key] = embedding
            with open(FACES_DB_PATH, 'wb') as f:
                pickle.dump(self.registered_faces, f)
                
            self.lbl_reg_status.config(text=f"Successfully registered: {name} ({role})", fg="#2ecc71")
            self.lbl_reg_db_count.config(text=f"Total Registered Persons: {len(self.registered_faces)}")
            
            # Clear inputs
            self.entry_name.delete(0, tk.END)
            self.entry_class.delete(0, tk.END)
            
        except Exception as e:
            self.lbl_reg_status.config(text=f"Error during registration: {str(e)}", fg="red")


    # ================= ATTENDANCE TAB ================= #
    def setup_attendance_tab(self):
        # Top Panel for controls
        control_frame = tk.Frame(self.tab_attendance, bg="#34495e", pady=10)
        control_frame.pack(fill=tk.X, side=tk.TOP)

        btn_webcam = ttk.Button(control_frame, text="Start Live Attendance", command=self.start_attendance)
        btn_webcam.pack(side=tk.LEFT, padx=10)

        self.btn_stop = ttk.Button(control_frame, text="Stop Camera", command=self.stop_processing, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        # Main content area using PanedWindow
        paned_window = ttk.PanedWindow(self.tab_attendance, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left pane: Canvas for video
        video_frame = tk.Frame(paned_window, bg="black")
        paned_window.add(video_frame, weight=3)
        self.canvas = tk.Canvas(video_frame, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas_image_id = None
        self.canvas_width = 800
        self.canvas_height = 600
        self.canvas.bind("<Configure>", self.on_canvas_resize)

        # Right pane: Logs
        log_frame = tk.Frame(paned_window, bg="#ecf0f1", width=300)
        paned_window.add(log_frame, weight=1)
        
        log_label = tk.Label(log_frame, text="Attendance Logs", font=("Helvetica", 12, "bold"), bg="#ecf0f1")
        log_label.pack(pady=5)
        
        self.log_text = tk.Text(log_frame, state=tk.DISABLED, font=("Consolas", 10), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.recent_logs = {} # To prevent logging the same person every frame. dict of {name: last_log_time}

    def start_attendance(self):
        if len(self.registered_faces) == 0:
            messagebox.showwarning("Empty DB", "No users registered. Please register users in the Registration tab first.")
            return
            
        self.stop_processing()
        self.vid_capture = cv2.VideoCapture(0) # 0 is usually default webcam
        if not self.vid_capture.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            return
            
        self.is_running = True
        self.btn_stop.config(state=tk.NORMAL)
        self.process_video_stream()

    def process_video_stream(self):
        if not self.is_running or not self.vid_capture.isOpened():
            self.stop_processing()
            return

        ret, frame = self.vid_capture.read()
        if ret:
            # Resize frame for better processing speed
            process_frame = cv2.resize(frame, (640, 480))
            
            # We process every Nth frame, or we can process async. DeepFace can be slow on CPU.
            # For a smooth GUI, we should run detection on a separate thread or use a faster cascade to find crops first.
            # To keep dependencies simple, we'll use OpenCV Haar cascade to find ROIs, then pass those to DeepFace to verify.
            processed_display_frame = self.detect_and_recognize(process_frame)
            
            self.current_frame = processed_display_frame
            self.display_frame(processed_display_frame)
            
            # Schedule next frame - keep it responsive but allow time for processing
            self.after(10, self.process_video_stream)
        else:
            self.stop_processing()

    def cosine_distance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def detect_and_recognize(self, frame):
        # We will use Haar cascade to find faces quickly in the live feed, then run DeepFace encoding on those crops
        # This is much faster for a live webcam feed than running RetinaFace on the entire image every frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Expand crop slightly
            cy1 = max(0, y - int(h * 0.2))
            cy2 = min(frame.shape[0], y + h + int(h * 0.2))
            cx1 = max(0, x - int(w * 0.2))
            cx2 = min(frame.shape[1], x + w + int(w * 0.2))
            
            face_roi = frame[cy1:cy2, cx1:cx2]
            
            if face_roi.size == 0:
                continue
                
            # Attempt to convert this face ROI into an embedding
            identity = "Unknown"
            color = (0, 0, 255) # Red for unknown
            
            try:
                # enforce_detection=False so it doesn't fail if retinaface misses the haar crop
                objs = DeepFace.represent(img_path=face_roi, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
                if len(objs) > 0:
                    live_emb = objs[0]["embedding"]
                    
                    # Compare with DB
                    best_match = None
                    best_dist = 1.0 # High threshold
                    threshold = 0.68 # threshold for ArcFace cosine distance
                    
                    for name_key, db_emb in self.registered_faces.items():
                        dist = self.cosine_distance(db_emb, live_emb)
                        if dist < threshold and dist < best_dist:
                            best_dist = dist
                            best_match = name_key
                            
                    if best_match:
                        parts = best_match.split('_')
                        name = parts[0]
                        role = parts[1]
                        course = parts[2]
                        
                        identity = f"{name} ({role})"
                        color = (0, 255, 0) # Green for known
                        
                        self.log_attendance(name, role, course)
            except Exception as e:
                 pass # DeepFace representation failed
                 
            # Draw box and label
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 2)
            cv2.putText(frame, identity, (cx1, max(10, cy1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def log_attendance(self, name, role, course):
        now = datetime.datetime.now()
        
        # Debounce logging: only log the same person once every 30 seconds
        if name in self.recent_logs:
            if (now - self.recent_logs[name]).total_seconds() < 30:
                return # Skip logging
                
        self.recent_logs[name] = now
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            with open(ATTENDANCE_CSV_PATH, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, name, role, course])
        except Exception as e:
            print(f"Failed to write to CSV: {e}")
        
        log_str = f"[{timestamp}]\nPresent: {name}\nRole: {role}\nClass: {course}\n{'-'*20}\n"
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, log_str)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def on_canvas_resize(self, event):
        self.canvas_width = event.width
        self.canvas_height = event.height
        if self.current_frame is not None:
            self.display_frame(self.current_frame)

    def display_frame(self, frame):
        if frame is None or self.canvas_width <= 1 or self.canvas_height <= 1:
            return

        height, width = frame.shape[:2]
        scaling_factor = min(self.canvas_width / width, self.canvas_height / height)
        
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        
        if new_width <= 0 or new_height <= 0:
            return

        resized_frame = cv2.resize(frame, (new_width, new_height))
        frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # We must keep a reference to this image so Tkinter doesn't garbage collect it
        self.current_display_image = ImageTk.PhotoImage(image=img) 
        
        x_offset = (self.canvas_width - new_width) // 2
        y_offset = (self.canvas_height - new_height) // 2

        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=self.current_display_image)
        else:
            self.canvas.coords(self.canvas_image_id, x_offset, y_offset)
            self.canvas.itemconfig(self.canvas_image_id, image=self.current_display_image)

    def stop_processing(self):
        self.is_running = False
        if hasattr(self, 'btn_stop') and self.btn_stop:
            self.btn_stop.config(state=tk.DISABLED)
        if hasattr(self, 'vid_capture') and self.vid_capture:
            self.vid_capture.release()
            self.vid_capture = None

    def on_closing(self):
        self.stop_processing()
        self.destroy()

if __name__ == "__main__":
    app = AttendanceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
