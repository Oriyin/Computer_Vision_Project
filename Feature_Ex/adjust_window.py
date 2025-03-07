# adjust_window.py
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from Feature_Ex.cv_pro import *

def create_adjust_window(root, img_cv, on_apply_callback):
    if img_cv is None:
        messagebox.showerror("Error", "No image loaded. Please open an image first.")
        return
        
    adjust_window = tk.Toplevel(root)
    adjust_window.title("Adjust Photo")
    adjust_window.geometry("1000x700")
    
    left_frame = tk.Frame(adjust_window)
    left_frame.pack(side="left", padx=20, pady=20)
    
    control_frame = tk.Frame(left_frame)
    control_frame.pack(fill="x", pady=10)
    
    color_frame = tk.LabelFrame(left_frame, text="Color Channels", padx=10, pady=5)
    color_frame.pack(fill="x", pady=10)
    
    preview_canvas = tk.Canvas(adjust_window, width=700, height=600)
    preview_canvas.pack(side="right", padx=20, pady=20)
    
    current_preview = {'img_pil': None}
    
    sliders_vars = {
        'brightness': tk.IntVar(value=0),
        'contrast': tk.IntVar(value=0),
        'highlights': tk.IntVar(value=0),
        'shadows': tk.IntVar(value=0),
        'saturation': tk.IntVar(value=0),
        'vibrance': tk.IntVar(value=0),
        'temperature': tk.IntVar(value=0),
        'blue': tk.IntVar(value=0),
        'green': tk.IntVar(value=0),
        'red': tk.IntVar(value=0)
    }

    def create_slider_with_entry(parent, text, variable):
        frame = tk.Frame(parent)
        frame.pack(fill="x", pady=5)
        
        label = tk.Label(frame, text=text, width=15, anchor="w")
        label.pack(side="left")
        
        entry = tk.Entry(frame, textvariable=variable, width=5)
        entry.pack(side="right")
        
        slider = tk.Scale(frame, from_=-100, to=100, orient="horizontal",
                         variable=variable, command=update_image, length=200)
        slider.pack(side="right", padx=5)
        
        def validate_entry(P):
            if P == "": return True
            try:
                value = int(P)
                return -100 <= value <= 100
            except ValueError:
                return False
        
        vcmd = (frame.register(validate_entry), '%P')
        entry.configure(validate="key", validatecommand=vcmd)
        
        def entry_update(event):
            try:
                value = int(entry.get())
                if -100 <= value <= 100:
                    variable.set(value)
                    update_image()
            except ValueError:
                pass
        
        entry.bind('<Return>', entry_update)
        entry.bind('<FocusOut>', entry_update)
        
        return frame

    def update_image(*args):
        img_adjusted = img_cv.copy()
        
        # ปรับค่าพื้นฐาน
        adjustments = {
            'brightness': adjust_brightness,
            'contrast': adjust_contrast,
            'highlights': adjust_highlights,
            'shadows': adjust_shadows,
            'saturation': adjust_saturation,
            'vibrance': adjust_vibrance,
            'temperature': adjust_temperature
        }
        
        for name, func in adjustments.items():
            if sliders_vars[name].get() != 0:
                img_adjusted = func(img_adjusted, sliders_vars[name].get())
        
        colors = {'blue': 'B', 'green': 'G', 'red': 'R'}
        for name, color in colors.items():
            if sliders_vars[name].get() != 0:
                img_adjusted = adjust_color_channel(img_adjusted, color, sliders_vars[name].get())
        
        if img_adjusted.dtype != np.uint8:
            img_adjusted = np.clip(img_adjusted, 0, 255).astype(np.uint8)
        
        img_rgb = cv2.cvtColor(img_adjusted, cv2.COLOR_BGR2RGB)
        current_preview['img_pil'] = Image.fromarray(img_rgb)
        current_preview['img_pil'].thumbnail((700, 600))
        img_tk = ImageTk.PhotoImage(current_preview['img_pil'])
        
        preview_canvas.delete("all")
        preview_canvas.create_image(350, 300, anchor=tk.CENTER, image=img_tk)
        preview_canvas.image = img_tk

    sliders_info = [
        ("Brightness", 'brightness'),
        ("Contrast", 'contrast'),
        ("Highlights", 'highlights'),
        ("Shadows", 'shadows'),
        ("Saturation", 'saturation'),
        ("Vibrance", 'vibrance'),
        ("Temperature", 'temperature')
    ]
    
    for label_text, var_name in sliders_info:
        create_slider_with_entry(control_frame, label_text, sliders_vars[var_name])

    color_sliders = [
        ("Blue", 'blue'),
        ("Green", 'green'),
        ("Red", 'red')
    ]
    
    for label_text, var_name in color_sliders:
        create_slider_with_entry(color_frame, label_text, sliders_vars[var_name])

    button_frame = tk.Frame(left_frame)
    button_frame.pack(pady=20)
    
    def apply_changes():
        update_image()
        adjusted_img = cv2.cvtColor(np.array(current_preview['img_pil']), cv2.COLOR_RGB2BGR)
        on_apply_callback(adjusted_img)
        adjust_window.destroy()
    
    apply_btn = tk.Button(button_frame, text="Apply", command=apply_changes,
                         width=10, height=2, bg="#4CAF50", fg="white")
    apply_btn.pack(side="left", padx=5)
    
    cancel_btn = tk.Button(button_frame, text="Cancel", command=adjust_window.destroy,
                          width=10, height=2, bg="#f44336", fg="white")
    cancel_btn.pack(side="left", padx=5)
    
    update_image()