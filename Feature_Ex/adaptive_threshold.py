import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

def create_adaptive_threshold_window(root, img_cv, on_apply_callback):
    threshold_window = tk.Toplevel(root)
    threshold_window.title("Black & White Adjustment")
    threshold_window.geometry("800x600")

    main_frame = tk.Frame(threshold_window)
    main_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

    preview_canvas_width = 600
    preview_canvas_height = 350
    preview_canvas = tk.Canvas(main_frame, width=preview_canvas_width, height=preview_canvas_height, bg='white')
    preview_canvas.pack(pady=10)

    control_frame = tk.Frame(main_frame)
    control_frame.pack(fill=tk.X, pady=10)

    block_size_var = tk.IntVar(value=11)
    c_value_var = tk.IntVar(value=2)
    method_var = tk.StringVar(value="Gaussian")
    
    processed_images = {
        'original': img_cv,
        'threshold': None,
        'current': None
    }

    def update_preview():
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method_var.get() == "Gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
        
        block_size = block_size_var.get()
        if block_size % 2 == 0: 
            block_size += 1
        
        threshold_img = cv2.adaptiveThreshold(
            img_gray, 
            255, 
            method, 
            cv2.THRESH_BINARY, 
            block_size, 
            c_value_var.get()
        )
        
        processed_images['threshold'] = threshold_img
        
        threshold_rgb = cv2.cvtColor(threshold_img, cv2.COLOR_GRAY2RGB)
        processed_images['current'] = threshold_rgb
        
        img_pil = Image.fromarray(threshold_rgb)
        img_width, img_height = img_pil.size
        
        scale = min(preview_canvas_width / img_width, preview_canvas_height / img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        preview_canvas.delete("all")
        preview_canvas.config(width=preview_canvas_width, height=preview_canvas_height)
        x_center = preview_canvas_width // 2
        y_center = preview_canvas_height // 2
        preview_canvas.create_image(x_center, y_center, anchor=tk.CENTER, image=img_tk)
        preview_canvas.image = img_tk

    def create_slider_with_label(parent, text, variable, from_=3, to=31, step=2):
        frame = tk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        
        label = tk.Label(frame, text=text, width=10)
        label.pack(side=tk.LEFT)
        
        slider = ttk.Scale(
            frame, 
            from_=from_, 
            to=to, 
            variable=variable, 
            orient=tk.HORIZONTAL, 
            command=lambda x: update_preview()
        )
        slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
        
        value_label = tk.Label(frame, textvariable=variable, width=3)
        value_label.pack(side=tk.RIGHT)

    create_slider_with_label(
        control_frame, 
        "Block Size", 
        block_size_var, 
        from_=3, 
        to=31, 
        step=1
    )

    create_slider_with_label(
        control_frame, 
        "Adjust", 
        c_value_var, 
        from_=0, 
        to=10, 
        step=1
    )

    method_frame = tk.Frame(control_frame)
    method_frame.pack(fill=tk.X, pady=5)
    
    tk.Label(method_frame, text="Method", width=10).pack(side=tk.LEFT)
    
    method_dropdown = ttk.Combobox(
        method_frame, 
        textvariable=method_var, 
        values=["Gaussian", "Mean"], 
        state="readonly",
        width=20
    )
    method_dropdown.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
    method_dropdown.bind("<<ComboboxSelected>>", lambda x: update_preview())
    
    def save_current_effect():
        if processed_images['current'] is not None:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
            )
            
            if save_path:
                cv2.imwrite(save_path, processed_images['current'])
                messagebox.showinfo("Success", f"Image saved as {save_path}")
    
    def apply_to_main():
        if processed_images['current'] is not None:
            on_apply_callback(processed_images['current'])
            threshold_window.destroy()
    
    action_frame = tk.Frame(main_frame)
    action_frame.pack(fill=tk.X, pady=10)
    
    apply_button = tk.Button(
        action_frame, 
        text="Apply to Image", 
        command=apply_to_main,
        width=15,
        height=2,
        bg="#4CAF50",
        fg="white"
    )
    apply_button.pack(side=tk.LEFT, padx=5)
    
    save_button = tk.Button(
        action_frame, 
        text="Save Image", 
        command=save_current_effect,
        width=15,
        height=2,
        bg="#2196F3",
        fg="white"
    )
    save_button.pack(side=tk.LEFT, padx=5)
    
    cancel_button = tk.Button(
        action_frame, 
        text="Cancel", 
        command=threshold_window.destroy,
        width=15,
        height=2,
        bg="#f44336",
        fg="white"
    )
    cancel_button.pack(side=tk.RIGHT, padx=5)

    update_preview()