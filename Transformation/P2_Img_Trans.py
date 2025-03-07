import tkinter as tk
from tkinter import messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from Transformation.Image_Tran import *

def open_transformation_window(root, img_cv,img_display, display_callback):
    if img_cv is None:
        messagebox.showerror("Error", "No image loaded. Please open an image first.")
        return
    
    window_width = 1000
    window_height = 700
    
    trans_window = tk.Toplevel(root)
    trans_window.title("Image Transformation")
    trans_window.geometry(f"{window_width}x{window_height}")
    trans_window.resizable(True, True)
    
    left_frame = tk.Frame(trans_window)
    left_frame.pack(side="left", padx=20, pady=20, fill="y")
    
    preview_frame = tk.Frame(trans_window)
    preview_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)
    
    canvas_width = window_width - 300  
    canvas_height = window_height - 40
    
    preview_canvas = tk.Canvas(preview_frame, width=canvas_width, height=canvas_height, bg="#333333")
    preview_canvas.pack(fill="both", expand=True)
    
    trans_frame = tk.LabelFrame(left_frame, text="Translation")
    trans_frame.pack(fill="x", pady=10)
    
    # Translation X
    tx_frame = tk.Frame(trans_frame)
    tx_frame.pack(fill="x", pady=5)
    tk.Label(tx_frame, text="X Translation:", width=12).pack(side="left")
    tx_var = tk.IntVar(value=0)
    tx_scale = ttk.Scale(tx_frame, from_=-100, to=100, variable=tx_var, orient="horizontal")  
    tx_scale.pack(side="left", padx=5, fill="x", expand=True)
    tx_entry = tk.Entry(tx_frame, textvariable=tx_var, width=6)
    tx_entry.pack(side="left", padx=5)
    
    # Translation Y
    ty_frame = tk.Frame(trans_frame)
    ty_frame.pack(fill="x", pady=5)
    tk.Label(ty_frame, text="Y Translation:", width=12).pack(side="left")
    ty_var = tk.IntVar(value=0)
    ty_scale = ttk.Scale(ty_frame, from_=-100, to=100, variable=ty_var, orient="horizontal")  
    ty_scale.pack(side="left", padx=5, fill="x", expand=True)
    ty_entry = tk.Entry(ty_frame, textvariable=ty_var, width=6)
    ty_entry.pack(side="left", padx=5)
    
    scale_frame = tk.LabelFrame(left_frame, text="Scaling")
    scale_frame.pack(fill="x", pady=10)
    
    # Scale X
    sx_frame = tk.Frame(scale_frame)
    sx_frame.pack(fill="x", pady=5)
    tk.Label(sx_frame, text="X Scale:", width=12).pack(side="left")
    sx_var = tk.DoubleVar(value=1.0)
    sx_scale = ttk.Scale(sx_frame, from_=0.1, to=2, variable=sx_var, orient="horizontal")
    sx_scale.pack(side="left", padx=5, fill="x", expand=True)
    sx_entry = tk.Entry(sx_frame, textvariable=sx_var, width=6)
    sx_entry.pack(side="left", padx=5)
    
    # Scale Y
    sy_frame = tk.Frame(scale_frame)
    sy_frame.pack(fill="x", pady=5)
    tk.Label(sy_frame, text="Y Scale:", width=12).pack(side="left")
    sy_var = tk.DoubleVar(value=1.0)
    sy_scale = ttk.Scale(sy_frame, from_=0.1, to=2, variable=sy_var, orient="horizontal")  
    sy_scale.pack(side="left", padx=5, fill="x", expand=True)
    sy_entry = tk.Entry(sy_frame, textvariable=sy_var, width=6)
    sy_entry.pack(side="left", padx=5)
    
    shear_frame = tk.LabelFrame(left_frame, text="Shear")
    shear_frame.pack(fill="x", pady=10)
    
    # Shear X
    shx_frame = tk.Frame(shear_frame)
    shx_frame.pack(fill="x", pady=5)
    tk.Label(shx_frame, text="X Shear:", width=12).pack(side="left")
    shx_var = tk.DoubleVar(value=0.0)
    shx_scale = ttk.Scale(shx_frame, from_=-1, to=1, variable=shx_var, orient="horizontal")  
    shx_scale.pack(side="left", padx=5, fill="x", expand=True)
    shx_entry = tk.Entry(shx_frame, textvariable=shx_var, width=6)
    shx_entry.pack(side="left", padx=5)
    
    # Shear Y
    shy_frame = tk.Frame(shear_frame)
    shy_frame.pack(fill="x", pady=5)
    tk.Label(shy_frame, text="Y Shear:", width=12).pack(side="left")
    shy_var = tk.DoubleVar(value=0.0)
    shy_scale = ttk.Scale(shy_frame, from_=-1, to=1, variable=shy_var, orient="horizontal") 
    shy_scale.pack(side="left", padx=5, fill="x", expand=True)
    shy_entry = tk.Entry(shy_frame, textvariable=shy_var, width=6)
    shy_entry.pack(side="left", padx=5)
    
    rotate_frame = tk.LabelFrame(left_frame, text="Rotation")
    rotate_frame.pack(fill="x", pady=10)
    
    angle_frame = tk.Frame(rotate_frame)
    angle_frame.pack(fill="x", pady=5)
    
    angle_var = tk.DoubleVar(value=0)
    
    def rotate_left():
        angle_var.set(angle_var.get() - 45)  
        update_preview()
        
    def rotate_right():
        angle_var.set(angle_var.get() + 45) 
        update_preview()
    
    rotate_left_btn = tk.Button(angle_frame, text="⟲ 45°", command=rotate_left,
                               width=8, bg="#4CAF50", fg="white")
    rotate_left_btn.pack(side="left", padx=5)
    
    rotate_right_btn = tk.Button(angle_frame, text="⟳ 45°", command=rotate_right,
                                width=8, bg="#4CAF50", fg="white")
    rotate_right_btn.pack(side="left", padx=5)
    
    angle_scale = ttk.Scale(angle_frame, from_=-360, to=360, variable=angle_var,
                           orient="horizontal")  
    angle_scale.pack(side="left", padx=5, fill="x", expand=True)
    angle_entry = tk.Entry(angle_frame, textvariable=angle_var, width=6)
    angle_entry.pack(side="left", padx=5)
    
    current_preview = {'transformed_img': None}
    
    def validate_entry(var, min_val, max_val, is_int=False):
        try:
            value = float(var.get())
            if is_int:
                value = int(value)
            if min_val <= value <= max_val:
                return True
            var.set(min_val if value < min_val else max_val)
        except ValueError:
            var.set(0)
        return False

    def validate_and_update():
        # แก้ไขให้ค่าที่ตรวจสอบสอดคล้องกับช่วงของสเกล
        validate_entry(tx_var, -100, 100, True)
        validate_entry(ty_var, -100, 100, True)
        validate_entry(sx_var, 0.1, 2)
        validate_entry(sy_var, 0.1, 2)
        validate_entry(shx_var, -1, 1)  # แก้จาก -0.1 เป็น -1
        validate_entry(shy_var, -1, 1)  # แก้จาก -0.1 เป็น -1
        validate_entry(angle_var, -360, 360)  # แก้จาก -50, 50 เป็น -360, 360
        update_preview()

    def update_preview(val=None):
        transformed_img = img_cv.copy()
        
        if tx_var.get() != 0 or ty_var.get() != 0:
            transformed_img = translate_image(transformed_img, tx_var.get(), ty_var.get())
        
        if sx_var.get() != 1.0 or sy_var.get() != 1.0:
            transformed_img = scale_image(transformed_img, sx_var.get(), sy_var.get())
        
        if shx_var.get() != 0.0 or shy_var.get() != 0.0:
            transformed_img = shear_image(transformed_img, shx_var.get(), shy_var.get())
        
        if angle_var.get() != 0:
            transformed_img = rotate_image(transformed_img, angle_var.get())
        
        if transformed_img.dtype != np.uint8:
            transformed_img = np.clip(transformed_img, 0, 255).astype(np.uint8)
        
        current_preview['transformed_img'] = transformed_img
        
        canvas_width = preview_canvas.winfo_width()
        canvas_height = preview_canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = window_width - 300
        if canvas_height <= 1:
            canvas_height = window_height - 40
            
        img_rgb = cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # แก้ไขการปรับขนาดภาพให้พอดีกับ canvas ทั้งระหว่างและหลังการแปลง
        img_width, img_height = img_pil.size
        scale_factor = min(canvas_width / img_width, canvas_height / img_height)
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)
        img_pil = img_pil.resize((new_width, new_height), Image.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_pil)
        
        preview_canvas.delete("all")
        preview_canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
        preview_canvas.image = img_tk
        return transformed_img
    
    def on_resize(event):
        update_preview()

    preview_canvas.bind("<Configure>", on_resize)
    
    for scale in [tx_scale, ty_scale, sx_scale, sy_scale, shx_scale, shy_scale, angle_scale]:
        scale.configure(command=update_preview)
    
    def validate_tx(name, index, mode):
        validate_and_update()
        
    def validate_ty(name, index, mode):
        validate_and_update()
        
    def validate_sx(name, index, mode):
        validate_and_update()
        
    def validate_sy(name, index, mode):
        validate_and_update()
        
    def validate_shx(name, index, mode):
        validate_and_update()
        
    def validate_shy(name, index, mode):
        validate_and_update()
        
    def validate_angle(name, index, mode):
        validate_and_update()
    
    tx_var.trace_add("write", validate_tx)
    ty_var.trace_add("write", validate_ty)
    sx_var.trace_add("write", validate_sx)
    sy_var.trace_add("write", validate_sy)
    shx_var.trace_add("write", validate_shx)
    shy_var.trace_add("write", validate_shy)
    angle_var.trace_add("write", validate_angle)
    
    button_frame = tk.Frame(left_frame)
    button_frame.pack(pady=20)
    
    def reset_transformations():
        tx_var.set(0)
        ty_var.set(0)
        sx_var.set(1.0)
        sy_var.set(1.0)
        shx_var.set(0.0)
        shy_var.set(0.0)
        angle_var.set(0)
        update_preview()
    
    def apply_changes():
        display_callback(current_preview['transformed_img'])
        trans_window.destroy()
    
    # Buttons
    reset_btn = tk.Button(button_frame, text="Reset", command=reset_transformations,
                         width=10, height=2, bg="#FFA500", fg="white")
    reset_btn.pack(side="left", padx=5)
    
    apply_btn = tk.Button(button_frame, text="Apply", command=apply_changes,
                         width=10, height=2, bg="#4CAF50", fg="white")
    apply_btn.pack(side="left", padx=5)
    
    cancel_btn = tk.Button(button_frame, text="Cancel", command=trans_window.destroy,
                          width=10, height=2, bg="#f44336", fg="white")
    cancel_btn.pack(side="left", padx=5)
    
    # Initial preview
    update_preview()