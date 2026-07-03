import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import simpledialog
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image, ImageTk
import os
import sys
import csv
import datetime
import pickle
# pyrefly: ignore [missing-import]
from deepface import DeepFace
import threading
import time

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
        
        # Thread safety and tracking state for async face recognition
        self.results_lock = threading.Lock()
        self.latest_results = []
        self.is_processing = False
        self.active_faces = {}
        self._alert_reset_id = None
        
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
        
        ttk.Button(btn_frame, text="Register using Camera", command=self.register_from_camera).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Select 10+ Images to Register", command=self.register_from_image).pack(side=tk.LEFT, padx=10)
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

        file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not file_paths:
            return

        if len(file_paths) < 10:
            messagebox.showwarning("Insufficient Images", f"Please select at least 10 images for accurate registration (you selected {len(file_paths)}).")
            return

        self.lbl_reg_status.config(text="Initializing registration process...", fg="yellow")
        self.update()

        embeddings = []
        skipped_images = 0
        multiple_faces_images = 0

        for idx, file_path in enumerate(file_paths, 1):
            self.lbl_reg_status.config(text=f"Processing image {idx} of {len(file_paths)}...", fg="yellow")
            self.update()

            try:
                # First attempt with default OpenCV face detection
                try:
                    embedding_objs = DeepFace.represent(img_path=file_path, model_name="ArcFace", detector_backend="opencv", enforce_detection=True)
                except ValueError:
                    # Fallback: force extraction on the whole image (e.g. if pre-cropped)
                    embedding_objs = DeepFace.represent(img_path=file_path, model_name="ArcFace", detector_backend="skip", enforce_detection=False)

                if len(embedding_objs) == 0:
                    skipped_images += 1
                    continue
                elif len(embedding_objs) > 1:
                    multiple_faces_images += 1
                    continue

                embeddings.append(embedding_objs[0]["embedding"])

            except Exception:
                skipped_images += 1

        if len(embeddings) < 10:
            self.lbl_reg_status.config(text="Registration failed. Too few valid faces detected.", fg="red")
            messagebox.showerror(
                "Registration Failed",
                f"Successfully extracted only {len(embeddings)} face templates from {len(file_paths)} images.\n"
                f"We require at least 10 valid single-face images to guarantee accuracy.\n\n"
                f"- Skipped / Failed: {skipped_images} images\n"
                f"- Multiple faces detected: {multiple_faces_images} images"
            )
            return

        try:
            # Create a unique key
            unique_key = f"{name}_{role}_{course}"
            
            # Save to DB
            self.registered_faces[unique_key] = embeddings
            with open(FACES_DB_PATH, 'wb') as f:
                pickle.dump(self.registered_faces, f)
                
            status_text = f"Successfully registered: {name} ({role}) with {len(embeddings)} templates."
            if skipped_images > 0 or multiple_faces_images > 0:
                status_text += f" (Skipped: {skipped_images}, Multi-face: {multiple_faces_images})"
                
            self.lbl_reg_status.config(text=status_text, fg="#2ecc71")
            self.lbl_reg_db_count.config(text=f"Total Registered Persons: {len(self.registered_faces)}")
            
            # Clear inputs
            self.entry_name.delete(0, tk.END)
            self.entry_class.delete(0, tk.END)
            
        except Exception as e:
            self.lbl_reg_status.config(text=f"Error saving to database: {str(e)}", fg="red")

    def register_from_camera(self):
        name = self.entry_name.get().strip()
        role = self.combo_role.get().strip()
        course = self.entry_class.get().strip()

        if not name or not course:
            messagebox.showwarning("Incomplete Data", "Please provide Name and Class/Course before registering.")
            return

        # Release webcam if in use by live attendance feed
        self.stop_processing()

        # Launch the camera capture popup
        CameraRegisterDialog(
            self, name, role, course,
            callback=lambda embs: self.on_camera_registration_success(name, role, course, embs)
        )

    def on_camera_registration_success(self, name, role, course, embeddings):
        try:
            # Create a unique key
            unique_key = f"{name}_{role}_{course}"
            
            # Save to DB
            self.registered_faces[unique_key] = embeddings
            with open(FACES_DB_PATH, 'wb') as f:
                pickle.dump(self.registered_faces, f)
                
            self.lbl_reg_status.config(text=f"Successfully registered: {name} ({role}) using camera (10 templates).", fg="#2ecc71")
            self.lbl_reg_db_count.config(text=f"Total Registered Persons: {len(self.registered_faces)}")
            
            # Clear inputs
            self.entry_name.delete(0, tk.END)
            self.entry_class.delete(0, tk.END)
            
            messagebox.showinfo("Registration Successful", f"Successfully registered {name} ({role}) using webcam!")
            
        except Exception as e:
            self.lbl_reg_status.config(text=f"Error saving to database: {str(e)}", fg="red")


    # ================= ATTENDANCE TAB ================= #
    def setup_attendance_tab(self):
        # Top Panel for controls
        control_frame = tk.Frame(self.tab_attendance, bg="#34495e", pady=10)
        control_frame.pack(fill=tk.X, side=tk.TOP)

        btn_webcam = ttk.Button(control_frame, text="Start Live Attendance", command=self.start_attendance)
        btn_webcam.pack(side=tk.LEFT, padx=10)

        self.btn_stop = ttk.Button(control_frame, text="Stop Camera", command=self.stop_processing, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        # Alert banner for new/unknown faces
        self.lbl_alert_banner = tk.Label(control_frame, text="System Active", fg="#2ecc71", bg="#34495e", font=("Helvetica", 12, "bold"))
        self.lbl_alert_banner.pack(side=tk.RIGHT, padx=20)

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
            
            # Real-time face detection on the main thread (very fast, ~10ms)
            gray = cv2.cvtColor(process_frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            # If no faces are detected in this frame, clear results to avoid showing stale bounding boxes
            if len(faces) == 0:
                with self.results_lock:
                    self.latest_results = []
            else:
                # If background thread is idle, start processing current frame & face crops
                if not self.is_processing:
                    self.is_processing = True
                    threading.Thread(
                        target=self.recognize_faces_async,
                        args=(process_frame.copy(), list(faces)),
                        daemon=True
                    ).start()
            
            # Draw real-time bounding boxes and labels
            for (x, y, w, h) in faces:
                # Default styling for a face being processed (BGR: Orange is (0, 165, 255))
                identity = "Processing..."
                color = (0, 165, 255)
                
                # Check if we have a matched recognized identity for this face box
                matched_face = self.match_face_to_cache((x, y, w, h))
                if matched_face:
                    identity = matched_face["identity"]
                    color = matched_face["color"]
                
                # Draw box and label
                cv2.rectangle(process_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(process_frame, identity, (x, max(10, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            self.current_frame = process_frame
            self.display_frame(process_frame)
            
            # Schedule next frame (approx 30 FPS)
            self.after(30, self.process_video_stream)
        else:
            self.stop_processing()

    def cosine_distance(self, source_representation, test_representation):
        a = np.matmul(np.transpose(source_representation), test_representation)
        b = np.sum(np.multiply(source_representation, source_representation))
        c = np.sum(np.multiply(test_representation, test_representation))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

    def recognize_faces_async(self, frame, faces):
        try:
            temp_results = []
            
            for (x, y, w, h) in faces:
                # Expand crop slightly to get a better face representation (30% padding for eye alignment)
                cy1 = max(0, y - int(h * 0.3))
                cy2 = min(frame.shape[0], y + h + int(h * 0.3))
                cx1 = max(0, x - int(w * 0.3))
                cx2 = min(frame.shape[1], x + w + int(w * 0.3))
                
                face_roi = frame[cy1:cy2, cx1:cx2]
                if face_roi.size == 0:
                    continue
                
                identity = "Unknown"
                color = (0, 0, 255) # Red for unknown face (BGR: Red is (0, 0, 255))
                
                try:
                    # Run DeepFace represent with OpenCV alignment enabled
                    objs = DeepFace.represent(img_path=face_roi, model_name="ArcFace", detector_backend="opencv", enforce_detection=False)
                    if len(objs) > 0:
                        live_emb = objs[0]["embedding"]
                        
                        best_match = None
                        best_dist = 1.0
                        threshold = 0.45 # Cosine distance threshold for ArcFace
                        
                        for name_key, db_val in self.registered_faces.items():
                            # Normalize representation: check if it is a list of embeddings or a single embedding
                            if len(db_val) > 0 and isinstance(db_val[0], (list, np.ndarray)):
                                db_embs = db_val
                            else:
                                db_embs = [db_val]
                                
                            # Find the minimum distance (best match) among all registered embeddings for this person
                            for db_emb in db_embs:
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
                            color = (0, 255, 0) # Green for known face (BGR: Green is (0, 255, 0))
                            
                            # Log attendance thread-safely on main thread
                            self.after(0, self.log_attendance, name, role, course)
                except Exception:
                    pass
                
                temp_results.append({
                    "box": (x, y, w, h),
                    "identity": identity,
                    "color": color
                })
                
                # Check for "new face entered" alert
                self.check_new_face_alert(identity)
                
            with self.results_lock:
                self.latest_results = temp_results
                
        finally:
            self.is_processing = False

    def match_face_to_cache(self, current_box):
        best_match = None
        best_dist = 999999.0
        
        # Center of current box
        cx1 = current_box[0] + current_box[2] / 2
        cy1 = current_box[1] + current_box[3] / 2
        
        # Allow matching only if the distance between centers is within 1.2 times of box dimensions
        max_allowed_dist = max(current_box[2], current_box[3]) * 1.2
        
        with self.results_lock:
            for recognized in self.latest_results:
                rec_box = recognized["box"]
                cx2 = rec_box[0] + rec_box[2] / 2
                cy2 = rec_box[1] + rec_box[3] / 2
                
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if dist < best_dist and dist < max_allowed_dist:
                    best_dist = dist
                    best_match = recognized
                    
        return best_match

    def check_new_face_alert(self, identity):
        now = datetime.datetime.now()
        current_time = now.timestamp()
        
        # Debounce alerts: only alert for a specific identity once every 10 seconds
        last_seen = self.active_faces.get(identity, 0)
        if current_time - last_seen > 10.0:
            self.active_faces[identity] = current_time
            
            # Play alert sound asynchronously (non-blocking beep in background thread)
            try:
                import winsound
                winsound.Beep(1000, 250)
            except Exception:
                pass
                
            # Trigger visual and log alerts on the main thread
            self.after(0, self.trigger_gui_alert, identity, now.strftime("%H:%M:%S"))

    def trigger_gui_alert(self, identity, time_str):
        # Format the alert banner text and styling
        if identity == "Unknown":
            alert_text = f"⚠️ ALERT: Unknown face entered at {time_str}!"
            color = "#e74c3c" # Red
        else:
            alert_text = f"🔔 ALERT: {identity} entered at {time_str}!"
            color = "#3498db" # Blue
            
        if hasattr(self, 'lbl_alert_banner') and self.lbl_alert_banner:
            self.lbl_alert_banner.config(text=alert_text, fg=color)
            
        # Log to the text area
        if self.log_text:
            self.log_text.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f"[{time_str}] {alert_text}\n{'-'*30}\n")
            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)
            
        # Reset alert banner after 4 seconds
        if hasattr(self, '_alert_reset_id') and self._alert_reset_id:
            self.after_cancel(self._alert_reset_id)
        self._alert_reset_id = self.after(4000, self.reset_alert_banner)

    def reset_alert_banner(self):
        if hasattr(self, 'lbl_alert_banner') and self.lbl_alert_banner:
            self.lbl_alert_banner.config(text="System Active", fg="#2ecc71")
        self._alert_reset_id = None

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
        self.is_processing = False
        with self.results_lock:
            self.latest_results = []
        self.active_faces.clear()
        if hasattr(self, 'lbl_alert_banner') and self.lbl_alert_banner:
            self.reset_alert_banner()
        if hasattr(self, 'btn_stop') and self.btn_stop:
            self.btn_stop.config(state=tk.DISABLED)
        if hasattr(self, 'vid_capture') and self.vid_capture:
            self.vid_capture.release()
            self.vid_capture = None

    def on_closing(self):
        self.stop_processing()
        self.destroy()

class CameraRegisterDialog(tk.Toplevel):
    def __init__(self, parent, name, role, course, callback):
        super().__init__(parent)
        self.title("Camera Face Registration")
        self.geometry("700x650")
        self.configure(bg="#2c3e50")
        self.resizable(False, False)
        
        self.parent = parent
        self.name = name
        self.role = role
        self.course = course
        self.callback = callback
        
        self.embeddings = []
        self.is_running = True
        self.is_processing = False
        self.last_capture_time = 0
        self.current_display_image = None
        
        # Setup UI
        self.lbl_instruction = tk.Label(
            self, 
            text="Please look at the camera. Turn your head slightly to capture different angles.\nCapturing 10 face templates automatically...", 
            fg="white", bg="#2c3e50", font=("Helvetica", 12, "bold")
        )
        self.lbl_instruction.pack(pady=10)
        
        self.canvas = tk.Canvas(self, bg="black", width=640, height=480, highlightthickness=0)
        self.canvas.pack(pady=5)
        
        self.lbl_progress = tk.Label(
            self, 
            text="Captured: 0 / 10 templates", 
            fg="#2ecc71", bg="#2c3e50", font=("Helvetica", 14, "bold")
        )
        self.lbl_progress.pack(pady=5)
        
        self.btn_cancel = ttk.Button(self, text="Cancel Registration", command=self.close_dialog)
        self.btn_cancel.pack(pady=5)
        
        # Open Camera
        self.vid_capture = cv2.VideoCapture(0)
        if not self.vid_capture.isOpened():
            messagebox.showerror("Error", "Could not open webcam.")
            self.destroy()
            return
            
        self.protocol("WM_DELETE_WINDOW", self.close_dialog)
        
        # Start Live Feed
        self.process_stream()

    def process_stream(self):
        if not self.is_running or not self.vid_capture.isOpened():
            return

        ret, frame = self.vid_capture.read()
        if ret:
            # Resize frame for better processing
            display_frame = cv2.resize(frame, (640, 480))
            
            # Detect face in real-time on preview to assist user alignment
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            
            # Draw real-time guides
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Display live preview
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            self.current_display_image = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.current_display_image)
            
            # Trigger background extraction every 800ms if a face is detected
            now = time.time()
            if len(faces) == 1 and not self.is_processing and len(self.embeddings) < 10 and (now - self.last_capture_time > 0.8):
                self.is_processing = True
                self.last_capture_time = now
                threading.Thread(
                    target=self.process_face_async, 
                    args=(display_frame.copy(), faces[0]), 
                    daemon=True
                ).start()
                
            self.after(30, self.process_stream)
        else:
            self.close_dialog()

    def process_face_async(self, frame_copy, face_box):
        try:
            x, y, w, h = face_box
            cy1 = max(0, y - int(h * 0.3))
            cy2 = min(frame_copy.shape[0], y + h + int(h * 0.3))
            cx1 = max(0, x - int(w * 0.3))
            cx2 = min(frame_copy.shape[1], x + w + int(w * 0.3))
            
            face_roi = frame_copy[cy1:cy2, cx1:cx2]
            if face_roi.size > 0:
                # Use OpenCV detector inside DeepFace to align face crops
                objs = DeepFace.represent(img_path=face_roi, model_name="ArcFace", detector_backend="opencv", enforce_detection=True)
                if len(objs) == 1:
                    emb = objs[0]["embedding"]
                    self.after(0, self.add_embedding, emb)
        except Exception:
            pass
        finally:
            self.is_processing = False

    def add_embedding(self, emb):
        self.embeddings.append(emb)
        count = len(self.embeddings)
        self.lbl_progress.config(text=f"Captured: {count} / 10 templates")
        
        # Audio cue
        try:
            import winsound
            winsound.Beep(1200, 150)
        except Exception:
            pass
            
        if count >= 10:
            self.callback(self.embeddings)
            self.close_dialog()

    def close_dialog(self):
        self.is_running = False
        if self.vid_capture and self.vid_capture.isOpened():
            self.vid_capture.release()
        self.destroy()

if __name__ == "__main__":
    app = AttendanceApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
