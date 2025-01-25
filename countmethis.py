import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class ObjectCounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Counter")
        
        # Variables
        self.image_path = None
        self.original_image = None
        self.shape_var = tk.StringVar(value="any")
        self.show_processed = tk.BooleanVar(value=False)
        self.use_erosion = tk.BooleanVar(value=False)
        self.show_contours = tk.BooleanVar(value=False)
        
        # Shape parameters
        self.circularity = tk.DoubleVar(value=0.7)
        self.solidity = tk.DoubleVar(value=0.8)
        self.rect_ratio_min = tk.DoubleVar(value=0.8)
        self.rect_ratio_max = tk.DoubleVar(value=1.2)
        self.approx_tolerance = tk.DoubleVar(value=0.04)
        
        # Main controls
        self.blur = tk.IntVar(value=5)
        self.threshold = tk.IntVar(value=60)
        self.min_area = tk.IntVar(value=100)
        
        self.shape_controls = {}
        self.setup_gui()
    
    def setup_gui(self):
        # Create frames
        self.control_frame = ttk.Frame(self.root, padding="5")
        self.control_frame.grid(row=0, column=0, sticky="nsew")
        
        self.image_frame = ttk.Frame(self.root, padding="5")
        self.image_frame.grid(row=0, column=1, sticky="nsew")
        
        current_row = 0
        
        # Basic controls
        ttk.Button(self.control_frame, text="Load Image", command=self.load_image).grid(row=current_row, column=0, pady=5)
        current_row += 1
        
        ttk.Label(self.control_frame, text="Shape Filter:").grid(row=current_row, column=0)
        current_row += 1
        shapes = ["any", "circle", "rectangle", "triangle"]
        shape_dropdown = ttk.Combobox(self.control_frame, textvariable=self.shape_var, values=shapes)
        shape_dropdown.grid(row=current_row, column=0, pady=5)
        shape_dropdown.bind('<<ComboboxSelected>>', self.on_shape_change)
        current_row += 1
        
        # Checkboxes
        ttk.Checkbutton(self.control_frame, text="Show Processed Image", 
                       variable=self.show_processed, 
                       command=self.update_image).grid(row=current_row, column=0, pady=5)
        current_row += 1
        
        ttk.Checkbutton(self.control_frame, text="Use Erosion", 
                       variable=self.use_erosion, 
                       command=self.update_image).grid(row=current_row, column=0, pady=5)
        current_row += 1
        
        ttk.Checkbutton(self.control_frame, text="Show Contours", 
                       variable=self.show_contours, 
                       command=self.update_image).grid(row=current_row, column=0, pady=5)
        current_row += 1
        
        # Main sliders
        self.add_slider("Blur", self.blur, 1, 21, current_row); current_row += 2
        self.add_slider("Threshold", self.threshold, 0, 255, current_row); current_row += 2
        self.add_slider("Min Area", self.min_area, 10, 1000, current_row); current_row += 2
        
        # Shape parameter sliders (initially hidden)
        circle_frame = ttk.LabelFrame(self.control_frame, text="Circle Parameters")
        self.shape_controls["circle"] = circle_frame
        self.add_slider("Circularity", self.circularity, 0, 1, 0, step=0.1, parent=circle_frame)
        self.add_slider("Solidity", self.solidity, 0, 1, 2, step=0.1, parent=circle_frame)
        
        rect_frame = ttk.LabelFrame(self.control_frame, text="Rectangle Parameters")
        self.shape_controls["rectangle"] = rect_frame
        self.add_slider("Min Ratio", self.rect_ratio_min, 0, 2, 0, step=0.1, parent=rect_frame)
        self.add_slider("Max Ratio", self.rect_ratio_max, 0, 2, 2, step=0.1, parent=rect_frame)
        self.add_slider("Approx Tol", self.approx_tolerance, 0.01, 0.1, 4, step=0.01, parent=rect_frame)
        
        triangle_frame = ttk.LabelFrame(self.control_frame, text="Triangle Parameters")
        self.shape_controls["triangle"] = triangle_frame
        self.add_slider("Approx Tol", self.approx_tolerance, 0.01, 0.1, 0, step=0.01, parent=triangle_frame)
        
        # Image and count labels
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.grid(row=0, column=0)
        
        self.count_label = ttk.Label(self.control_frame, text="Objects: 0")
        self.count_label.grid(row=current_row, column=0, pady=10)
        
    def add_slider(self, name, variable, from_, to_, row, step=1, parent=None):
        parent = parent or self.control_frame
        ttk.Label(parent, text=name + ":").grid(row=row, column=0)
        ttk.Scale(parent, from_=from_, to=to_, variable=variable, 
                 command=self.update_image).grid(row=row+1, column=0)
    
    def on_shape_change(self, *args):
        # Hide all shape control frames
        for frame in self.shape_controls.values():
            frame.grid_remove()
        
        # Show the relevant frame
        selected_shape = self.shape_var.get()
        if selected_shape in self.shape_controls:
            self.shape_controls[selected_shape].grid(column=0, sticky="ew", pady=10)
        
        self.update_image()
    
    def detect_shape(self, contour):
        shape = self.shape_var.get()
        if shape == "any":
            return True
            
        peri = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        if area == 0:
            return False
            
        if shape == "circle":
            circularity = 4 * np.pi * area / (peri * peri)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0
            return circularity > self.circularity.get() and solidity > self.solidity.get()
            
        elif shape == "rectangle":
            approx = cv2.approxPolyDP(contour, self.approx_tolerance.get() * peri, True)
            if len(approx) != 4:
                return False
            x, y, w, h = cv2.boundingRect(approx)
            ratio = float(w) / h if h > 0 else 0
            return self.rect_ratio_min.get() <= ratio <= self.rect_ratio_max.get()
            
        elif shape == "triangle":
            approx = cv2.approxPolyDP(contour, self.approx_tolerance.get() * peri, True)
            return len(approx) == 3
            
        return True
    
    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = cv2.imread(self.image_path)
            self.update_image()
    
    def process_image(self):
        if self.original_image is None:
            return None, None, 0
            
        blur_value = self.blur.get()
        if blur_value % 2 == 0:
            blur_value += 1
            
        gray = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
        _, thresh = cv2.threshold(blurred, self.threshold.get(), 255, cv2.THRESH_BINARY)
        
        if self.use_erosion.get():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            thresh = cv2.erode(thresh, kernel, iterations=1)
        
        processed = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_image = self.original_image.copy()
        valid_contours = []
        
        for cnt in contours:
            if cv2.contourArea(cnt) > self.min_area.get() and self.detect_shape(cnt):
                valid_contours.append(cnt)
                if self.show_contours.get():
                    cv2.drawContours(result_image, [cnt], -1, (0, 0, 255), 2)
                    cv2.drawContours(processed, [cnt], -1, (0, 0, 255), 2)
                
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    if not self.show_contours.get():
                        cv2.circle(result_image, (cx, cy), 5, (0, 255, 0), -1)
                        cv2.circle(processed, (cx, cy), 5, (0, 255, 0), -1)
                    
        return result_image, processed, len(valid_contours)
    
    def update_image(self, *args):
        if self.original_image is None:
            return
            
        result_img, processed_img, count = self.process_image()
        display_img = processed_img if self.show_processed.get() else result_img
        
        image = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        
        display_size = (800, 600)
        image.thumbnail(display_size, Image.LANCZOS)
        
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo
        
        self.count_label.configure(text=f"Objects: {count}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectCounterGUI(root)
    root.mainloop()