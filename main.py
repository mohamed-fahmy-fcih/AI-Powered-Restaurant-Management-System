import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from keras.models import load_model
from PIL import Image, ImageTk
import threading
import time
import os
from firebase_control import *


class EmotionDetectionApp:
    def __init__(self, root, db=None):
        """Initialize the Emotion Detection App"""
        self.root = root
        self.restaurant_id = None  # Add this to store the restaurant ID
        self.db = db  # Firebase database reference
        self.emotion_model = None
        self.digit_model = None
        self.emotion_ranges = ['pos', 'neg', 'normal']
        self.source_var = None
        self.video_path = None
        self.setup_window()
        self.load_models()

        # Video dimensions
        self.video_width = 800
        self.video_height = 600

        # Processing control
        self.is_running = False
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.last_time = time.time()
        self.stats = {'pos': 0, 'neg': 0, 'normal': 0, 'total_frames': 0}
        self.detected_digit = None
        self.screenshot_taken = False  # Track if screenshot has been taken
        self.screenshot_url = None  # Store Firebase screenshot URL

        # Table ROI settings
        self.tables = {}
        self.drawing_roi = False
        self.start_x, self.start_y = 0, 0
        self.current_table_id = None
        self.table_counter = 1

        # Directory for saving images
        self.image_dir = "table_images"
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def show_alert_popup(self, neg_percent):
        """Show a popup with alert information when negative emotions reach 30%"""
        # Get current date and time
        current_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Prepare message
        message = (
            f"Alert: Negative Emotions Reached {neg_percent:.1f}%\n\n"
            f"Date: {current_date}\n"
            f"Restaurant ID: {self.restaurant_id}\n"
            f"Table Number: {self.detected_digit if self.detected_digit is not None else 'Not Detected'}\n"
            f"Screenshot Firebase Link: {self.screenshot_url if self.screenshot_url else 'Upload Failed'}"
        )

        # Schedule the popup on the main thread
        self.root.after(0, lambda: messagebox.showwarning(
            "Negative Emotion Alert",
            message
        ))
    def setup_window(self):
        """Configure the main window"""
        self.root.title("Emotion Detection App")
        self.root.geometry("1200x900")
        self.root.minsize(1000, 700)
        self.root.resizable(True, True)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

    def load_models(self):
        """Load the trained emotion and digit detection models"""
        emotion_model_path = 'Final_cnn.h5'
        digit_model_path = 'mnist_cnn_model.h5'

        if not os.path.exists(emotion_model_path):
            print(f"Emotion model file not found: {emotion_model_path}")
            messagebox.showerror("Error", f"Emotion model file not found: {emotion_model_path}")
            self.emotion_model = None
        else:
            try:
                self.emotion_model = load_model(emotion_model_path)
                print("Emotion model loaded successfully")
            except Exception as e:
                print(f"Error loading emotion model: {e}")
                self.emotion_model = None

        if not os.path.exists(digit_model_path):
            print(f"Digit model file not found: {digit_model_path}")
            messagebox.showerror("Error", f"Digit model file not found: {digit_model_path}")
            self.digit_model = None
        else:
            try:
                self.digit_model = load_model(digit_model_path)
                print("Digit model loaded successfully")
            except Exception as e:
                print(f"Error loading digit model: {e}")
                self.digit_model = None

    def start_app(self):
        """Start the application with login page"""
        self.show_login_page()
        self.root.mainloop()

    def show_login_page(self):
        """Display the login page"""
        for widget in self.root.winfo_children():
            widget.destroy()

        login_frame = tk.Frame(self.root, bg="#f0f0f0")
        login_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER, width=400, height=300)

        tk.Label(login_frame, text="Emotion Detection Login", font=("Arial", 20, "bold"), bg="#f0f0f0").pack(pady=20)
        tk.Label(login_frame, text="Username:", font=("Arial", 12), bg="#f0f0f0").pack(anchor=tk.W, padx=50)
        self.username_entry = tk.Entry(login_frame, font=("Arial", 12), width=30)
        self.username_entry.pack(pady=5, padx=50)
        tk.Label(login_frame, text="Password:", font=("Arial", 12), bg="#f0f0f0").pack(anchor=tk.W, padx=50)
        self.password_entry = tk.Entry(login_frame, font=("Arial", 12), width=30, show="*")
        self.password_entry.pack(pady=5, padx=50)
        tk.Button(
            login_frame,
            text="Login",
            font=("Arial", 12),
            bg="#4CAF50",
            fg="white",
            width=20,
            command=self.process_login
        ).pack(pady=20)

    def process_login(self):
        username = self.username_entry.get()
        password = self.password_entry.get()
        if not username or not password:
            messagebox.showerror("Error", "Please enter both username and password")
            return
        names, passwords, ids = get_all_credentials(self.db, "restaurants")  # Updated to get IDs
        if username not in names:
            messagebox.showerror("Error", "Invalid username")
            return
        if password not in passwords:
            messagebox.showerror("Error", "Invalid password")
            return

        # Find the index of the matching username and get the corresponding ID
        index = names.index(username)
        if passwords[index] == password:
            self.restaurant_id = ids[index]  # Store the restaurant ID
            messagebox.showinfo("Success", f"Welcome, {username}! (ID: {self.restaurant_id})")
            self.username = username
            self.show_dashboard(username)
        else:
            messagebox.showerror("Error", "Invalid password")
    def show_dashboard(self, username):
        """Display the main dashboard"""
        for widget in self.root.winfo_children():
            widget.destroy()

        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        main_container.columnconfigure(0, weight=1)
        main_container.rowconfigure(2, weight=1)

        self.create_header(main_container, username)
        tk.Label(main_container, text="Emotion Detection Dashboard",
                 font=("Arial", 20, "bold"), bg="#f5f5f5").pack(pady=10, fill=tk.X)

        self.create_video_player(main_container)
        self.create_status_bar(main_container)

        self.fullscreen_var = tk.BooleanVar(value=False)
        self.fullscreen_button = tk.Button(
            main_container,
            text="Toggle Fullscreen",
            command=self.toggle_fullscreen,
            bg="#555",
            fg="white"
        )
        self.fullscreen_button.place(relx=1.0, rely=0, anchor="ne", x=-80, y=10)

    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.fullscreen_var.get():
            self.root.attributes("-fullscreen", False)
            self.fullscreen_var.set(False)
            self.fullscreen_button.config(text="Toggle Fullscreen")
        else:
            self.root.attributes("-fullscreen", True)
            self.fullscreen_var.set(True)
            self.fullscreen_button.config(text="Exit Fullscreen")

    def create_header(self, parent, username):
        """Create the header with user info and logout button"""
        header_frame = tk.Frame(parent, bg="#333")
        header_frame.pack(fill=tk.X)
        tk.Label(
            header_frame,
            text=f"Logged in as: {username} (ID: {self.restaurant_id})",
            font=("Arial", 12),
            bg="#333",
            fg="white",
            padx=10,
            pady=5
        ).pack(side=tk.LEFT)
        tk.Button(
            header_frame,
            text="Logout",
            command=self.show_login_page,
            bg="#d32f2f",
            fg="white"
        ).pack(side=tk.RIGHT, padx=10, pady=5)
    def create_video_player(self, parent):
        """Create the video player UI"""
        video_container = tk.Frame(parent, bg="#f5f5f5", pady=10)
        video_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        video_container.columnconfigure(0, weight=1)
        video_container.rowconfigure(0, weight=1)

        video_frame = tk.Frame(video_container, bg="#f5f5f5")
        video_frame.pack(fill=tk.BOTH, expand=True)
        video_frame.columnconfigure(0, weight=1)
        video_frame.rowconfigure(0, weight=1)

        self.video_display_frame = tk.Frame(video_frame, width=self.video_width, height=self.video_height, bg="black")
        self.video_display_frame.grid(row=0, column=0)
        self.video_display_frame.grid_propagate(False)

        self.video_display = tk.Label(self.video_display_frame, bg="black")
        self.video_display.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events for ROI selection
        self.video_display.bind("<ButtonPress-1>", self.start_roi)
        self.video_display.bind("<B1-Motion>", self.draw_roi)
        self.video_display.bind("<ButtonRelease-1>", self.end_roi)

        self.video_placeholder = tk.Label(
            self.video_display,
            text="Video Player (800x600)",
            font=("Arial", 16),
            bg="black",
            fg="white"
        )
        self.video_placeholder.pack(fill=tk.BOTH, expand=True)

        controls_frame = tk.Frame(video_container, bg="#f5f5f5")
        controls_frame.pack(fill=tk.X, pady=10)

        tk.Button(
            controls_frame,
            text="Use Webcam",
            command=self.use_webcam,
            bg="#2196F3",
            fg="white",
            width=15
        ).pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls_frame,
            text="Select Video",
            command=self.select_video_file,
            bg="#2196F3",
            fg="white",
            width=15
        ).pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(
            controls_frame,
            text="Stop",
            command=self.stop_processing,
            bg="#f44336",
            fg="white",
            width=15,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Fixed: Separate creation and packing
        self.add_table_button = tk.Button(
            controls_frame,
            text="Add Number ROI",
            command=self.add_table_roi,
            bg="#4CAF50",
            fg="white",
            width=15
        )
        self.add_table_button.pack(side=tk.LEFT, padx=5)

        self.reset_roi_button = tk.Button(
            controls_frame,
            text="Reset Number ROI",
            command=self.reset_roi,
            bg="#FF9800",
            fg="white",
            width=15
        )
        self.reset_roi_button.pack(side=tk.LEFT, padx=5)

        self.results_frame = tk.Frame(video_container, bg="#f5f5f5", bd=2, relief=tk.GROOVE)
        self.results_frame.pack(fill=tk.X, pady=10)
        tk.Label(
            self.results_frame,
            text="Emotion Detection Results",
            font=("Arial", 12, "bold"),
            bg="#f5f5f5"
        ).pack(pady=5)
        self.results_content = tk.Label(
            self.results_frame,
            text="No results yet. Select a video or use webcam to analyze.",
            font=("Arial", 10),
            bg="#f5f5f5"
        )
        self.results_content.pack(pady=5)
    def create_status_bar(self, parent):
        """Create the status bar"""
        status_bar = tk.Frame(parent, bg="#333")
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(
            status_bar,
            text="Ready",
            font=("Arial", 10),
            bg="#333",
            fg="white",
            anchor=tk.W,
            padx=10,
            pady=5
        )
        self.status_label.pack(fill=tk.X)

    def start_processing(self):
        """Start video processing"""
        if not self.emotion_model or not self.digit_model:
            tk.messagebox.showerror("Error", "Models could not be loaded")
            return

        if self.source_var == "webcam":
            self.cap = cv2.VideoCapture(0)
            source_desc = "webcam"
        elif self.source_var == "file" and self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            source_desc = f"file: {self.video_path}"
        else:
            tk.messagebox.showerror("Error", "Invalid video source or no file selected")
            return

        if not self.cap.isOpened():
            tk.messagebox.showerror("Error", f"Could not open video source: {source_desc}")
            print(f"Failed to open {source_desc}")
            self.cap = None
            return

        print(f"Successfully opened {source_desc}")
        self.stop_button.config(state=tk.NORMAL)
        self.add_table_button.config(state=tk.NORMAL)
        self.reset_roi_button.config(state=tk.NORMAL)
        self.video_placeholder.pack_forget()
        self.stats = {'pos': 0, 'neg': 0, 'normal': 0, 'total_frames': 0}
        self.detected_digit = None
        self.screenshot_taken = False
        self.screenshot_url = None

        self.is_running = True
        self.video_thread = threading.Thread(target=self.process_video)
        self.video_thread.daemon = True
        try:
            self.video_thread.start()
            print("Video thread started")
        except Exception as e:
            print(f"Error starting video thread: {e}")
            self.is_running = False
            tk.messagebox.showerror("Error", "Failed to start video processing")

    def stop_processing(self):
        """Stop video processing"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.cap = None

        self.stop_button.config(state=tk.DISABLED)
        self.add_table_button.config(state=tk.NORMAL)
        self.reset_roi_button.config(state=tk.DISABLED)
        self.video_display.config(image='')
        self.video_placeholder.pack(fill=tk.BOTH, expand=True)
        self.status_label.config(text="Video stopped")
        self.calculate_results()

    def process_video(self):
        """Process video frames in a separate thread"""
        print("Starting video processing")
        self.frame_count = 0
        self.last_time = time.time()

        while self.is_running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    if self.source_var == "file" and self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(
                            cv2.CAP_PROP_FRAME_COUNT):
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        self.root.after(0, self.stop_processing)
                        break

                self.current_frame = frame
                visualization = frame.copy()
                emotion, face_coords = None, None

                # Process ROI for digit detection
                if self.tables:
                    table_id, (x, y, w, h) = list(self.tables.items())[0]
                    roi = frame[y:y + h, x:x + w]
                    if roi.size > 0:
                        cv2.rectangle(visualization, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        digit_info = self.detect_digit_in_roi(roi)
                        if digit_info:
                            digit, confidence = digit_info
                            self.detected_digit = digit
                            text = f"Table {digit} ({confidence:.2f})"
                        else:
                            self.detected_digit = 0
                            text = "Table 0"
                        cv2.putText(visualization, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Emotion detection
                visualization, emotion, face_coords, confidence = self.detect_emotion(visualization)
                if emotion:
                    self.stats[emotion] += 1
                    self.stats['total_frames'] += 1

                    # Check for 30% negative emotions and process
                    if self.stats['total_frames'] > 0:
                        neg_percent = (self.stats['neg'] / self.stats['total_frames']) * 100
                        if neg_percent >= 30 and not self.screenshot_taken:
                            screenshot_path = os.path.join(self.image_dir,
                                                           f"screenshot_{self.restaurant_id}_{int(time.time())}.jpg")
                            cv2.imwrite(screenshot_path, visualization)
                            if self.db:
                                try:
                                    from firebase_control import upload_file
                                    self.screenshot_url = upload_file(self.db, screenshot_path,
                                                                      f"screenshots/{self.restaurant_id}/{os.path.basename(screenshot_path)}")
                                    print(f"Screenshot uploaded to Firebase: {self.screenshot_url}")
                                except (NameError, AttributeError) as e:
                                    print(f"Firebase upload failed: {e}")
                                    self.screenshot_url = f"Local: {screenshot_path}"
                            else:
                                self.screenshot_url = f"Local: {screenshot_path}"
                                print("Firebase not available, screenshot saved locally")

                            # Prepare data and show popup
                            current_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            detection_data = {
                                "negative_percent": neg_percent,
                                "date": current_date,
                                "restaurant_id": self.restaurant_id,
                                "table_number": self.detected_digit if self.detected_digit is not None else 0,
                                "screenshot_url": self.screenshot_url if self.screenshot_url else "Upload Failed"
                            }
                            self.show_alert_popup(detection_data)
                            # Store in Firebase
                            if self.db:
                                from firebase_control import store_detection
                                store_detection(self.db, detection_data)
                            self.screenshot_taken = True

                # Update FPS
                self.frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - self.last_time
                if elapsed_time >= 1.0:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.last_time = current_time

                cv2.putText(visualization, f"FPS: {self.fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                frame_rgb = cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (self.video_width, self.video_height))
                frame_pil = Image.fromarray(frame_resized)
                frame_tk = ImageTk.PhotoImage(image=frame_pil)
                self.root.after(0, lambda: self.video_display.config(image=frame_tk))
                self.root.after(0, lambda: setattr(self.video_display, 'image', frame_tk))
                self.root.after(0, self.update_results_during_processing)

                time.sleep(0.015)
            except Exception as e:
                print(f"Error in video processing: {e}")
                self.root.after(0, self.stop_processing)
                break

    def show_alert_popup(self, detection_data):
        """Show a popup with alert information when negative emotions reach 30%"""
        message = (
            f"Alert: Negative Emotions Reached {detection_data['negative_percent']:.1f}%\n\n"
            f"Date: {detection_data['date']}\n"
            f"Restaurant ID: {detection_data['restaurant_id']}\n"
            f"Table Number: {detection_data['table_number']}\n"
            f"Screenshot Firebase Link: {detection_data['screenshot_url']}"
        )

        self.root.after(0, lambda: messagebox.showwarning(
            "Negative Emotion Alert",
            message
        ))
    def detect_digit_in_roi(self, roi):
        """Detect digits within the ROI"""
        if not self.digit_model:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        min_area = 100
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        if not valid_contours:
            return None

        largest_contour = valid_contours[0]
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit_img = gray[y:y + h, x:x + w]

        if digit_img.size == 0:
            return None

        processed_digit = self.preprocess_digit_for_recognition(digit_img)
        prediction = self.digit_model.predict(processed_digit, verbose=0)
        digit = np.argmax(prediction)
        confidence = np.max(prediction)

        if confidence > 0.9:
            return int(digit), float(confidence)
        return None

    def preprocess_digit_for_recognition(self, digit_img):
        """Preprocess digit image for recognition"""
        digit_img = cv2.resize(digit_img, (28, 28))
        if np.mean(digit_img) > 127:
            digit_img = 255 - digit_img
        _, digit_img = cv2.threshold(digit_img, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        digit_img = digit_img.astype('float32') / 255.0
        digit_img = np.expand_dims(digit_img, axis=-1)
        digit_img = np.expand_dims(digit_img, axis=0)
        return digit_img

    def preprocess_image(self, frame, x, y, w, h):
        """Preprocess face image for emotion detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = gray[y:y + h, x:x + w]
            emotion_img = cv2.resize(face, (48, 48), interpolation=cv2.INTER_AREA)
            emotion_image_array = np.array(emotion_img, dtype=np.float32) / 255.0
            emotion_input = np.expand_dims(emotion_image_array, axis=0)
            emotion_input = np.expand_dims(emotion_input, axis=-1)
            return emotion_input
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return None

    def detect_emotion(self, frame):
        """Detect emotions in the frame with confidence"""
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            detected_emotion = None
            face_coords = None
            confidence = None

            for (x, y, w, h) in faces:
                emotion_input = self.preprocess_image(frame, x, y, w, h)
                if emotion_input is None:
                    continue

                predictions = self.emotion_model.predict(emotion_input, verbose=0)
                predicted_index = np.argmax(predictions)
                confidence = float(np.max(predictions))

                if predicted_index < len(self.emotion_ranges):
                    output_emotion = self.emotion_ranges[predicted_index]
                    detected_emotion = output_emotion
                    face_coords = (x, y, w, h)
                else:
                    output_emotion = "Unknown"

                emotion_color = {
                    'pos': (0, 255, 0),
                    'neg': (0, 0, 255),
                    'normal': (255, 165, 0)
                }.get(output_emotion, (255, 255, 255))

                cv2.rectangle(frame, (x, y), (x + w, y + h), emotion_color, 2)
                # Show emotion with confidence above the face
                label = f"{output_emotion} ({confidence:.2f})"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_color, 2)

            return frame, detected_emotion, face_coords, confidence

        except Exception as e:
            print(f"Error in detection: {e}")
            return frame, None, None, None

    def add_table_roi(self):
        """Initiate ROI selection for table number"""
        if not self.is_running:
            messagebox.showinfo("Info", "Please start a video or webcam first.")
            return

        if self.tables:
            messagebox.showinfo("Info", "Only one ROI is allowed. Use Reset ROI to clear the existing one.")
            return

        self.current_table_id = "1"
        self.status_label.config(text="Draw ROI for Table 1")
        self.drawing_roi = True

    def reset_roi(self):
        """Reset the ROI"""
        self.tables.clear()
        self.detected_digit = None
        self.status_label.config(text="ROI reset")
        if self.is_running:
            self.update_results_during_processing()

    def start_roi(self, event):
        """Start drawing ROI"""
        if not self.drawing_roi or not hasattr(self, 'current_frame'):
            return
        self.start_x = int(event.x * (self.current_frame.shape[1] / self.video_width))
        self.start_y = int(event.y * (self.current_frame.shape[0] / self.video_height))

    def draw_roi(self, event):
        """Draw ROI rectangle while dragging"""
        if not self.drawing_roi or not hasattr(self, 'current_frame'):
            return

        current_x = int(event.x * (self.current_frame.shape[1] / self.video_width))
        current_y = int(event.y * (self.current_frame.shape[0] / self.video_height))

        if self.current_frame is not None:
            draw_frame = self.current_frame.copy()
            cv2.rectangle(draw_frame, (self.start_x, self.start_y), (current_x, current_y), (0, 255, 0), 2)

            frame_rgb = cv2.cvtColor(draw_frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (self.video_width, self.video_height))
            frame_pil = Image.fromarray(frame_resized)
            frame_tk = ImageTk.PhotoImage(image=frame_pil)
            self.video_display.config(image=frame_tk)
            self.video_display.image = frame_tk

    def end_roi(self, event):
        """Finish drawing ROI"""
        if not self.drawing_roi or not hasattr(self, 'current_frame'):
            return

        end_x = int(event.x * (self.current_frame.shape[1] / self.video_width))
        end_y = int(event.y * (self.current_frame.shape[0] / self.video_height))

        x = min(self.start_x, end_x)
        y = min(self.start_y, end_y)
        w = abs(end_x - self.start_x)
        h = abs(end_y - self.start_y)

        if w > 10 and h > 10:
            self.tables[self.current_table_id] = (x, y, w, h)
            self.status_label.config(text="Table 1 ROI added")
        else:
            self.status_label.config(text="ROI too small, try again")

        self.drawing_roi = False
        self.current_table_id = None

    def use_webcam(self):
        """Start webcam feed"""
        if not self.emotion_model or not self.digit_model:
            messagebox.showerror("Error", "Models not loaded!")
            return

        self.source_var = "webcam"
        self.start_processing()

    def select_video_file(self):
        """Select and play a video file"""
        if not self.emotion_model or not self.digit_model:
            messagebox.showerror("Error", "Models not loaded!")
            return

        self.video_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=[("Video Files", "*.mp4;*.avi;*.mov")]
        )
        if self.video_path:
            self.source_var = "file"
            self.start_processing()
            self.status_label.config(text=f"Playing video: {self.video_path.split('/')[-1]}")

    def update_results_during_processing(self):
        """Update results during processing"""
        if self.stats['total_frames'] > 0:
            pos_percent = (self.stats['pos'] / self.stats['total_frames']) * 100
            neg_percent = (self.stats['neg'] / self.stats['total_frames']) * 100
            normal_percent = (self.stats['normal'] / self.stats['total_frames']) * 100

            results_text = f"Live Results (processing):\n"
            results_text += f"Positive: {self.stats['pos']} frames ({pos_percent:.1f}%)\n"
            results_text += f"Negative: {self.stats['neg']} frames ({neg_percent:.1f}%)\n"
            results_text += f"Neutral: {self.stats['normal']} frames ({normal_percent:.1f}%)\n"
            if self.detected_digit is not None:
                results_text += f"Detected Table Number: {self.detected_digit}\n"
            if self.screenshot_url:
                results_text += f"Screenshot URL: {self.screenshot_url}"
            self.results_content.config(text=results_text)

    def calculate_results(self):
        """Calculate and display final results"""
        if self.stats['total_frames'] == 0:
            self.results_content.config(text="No faces detected.")
            return

        pos_percent = (self.stats['pos'] / self.stats['total_frames']) * 100
        neg_percent = (self.stats['neg'] / self.stats['total_frames']) * 100
        normal_percent = (self.stats['normal'] / self.stats['total_frames']) * 100

        emotion_stats = {k: v for k, v in self.stats.items() if k != 'total_frames'}
        dominant = max(emotion_stats, key=lambda k: emotion_stats[k])

        results_text = f"Final Results:\n"
        results_text += f"Positive: {self.stats['pos']} frames ({pos_percent:.1f}%)\n"
        results_text += f"Negative: {self.stats['neg']} frames ({neg_percent:.1f}%)\n"
        results_text += f"Neutral: {self.stats['normal']} frames ({normal_percent:.1f}%)\n"
        if self.detected_digit is not None:
            results_text += f"Detected Table Number: {self.detected_digit}\n"
        if self.screenshot_url:
            results_text += f"Screenshot URL: {self.screenshot_url}\n"
        results_text += f"\nDominant emotion: {dominant.upper()} ({self.stats[dominant]} frames)"
        self.results_content.config(text=results_text)
        self.status_label.config(text=f"Analysis complete. Dominant emotion: {dominant.upper()}")


if __name__ == "__main__":
    try:
        from firebase_control import connect_to_firebase

        db = connect_to_firebase('json_file.json')
    except Exception as e:
        print(f"Firebase connection error: {e}")
        messagebox.showerror("Error", "Could not connect to Firebase. Running in demo mode.")
        db = None

    root = tk.Tk()
    app = EmotionDetectionApp(root, db)
    app.start_app()