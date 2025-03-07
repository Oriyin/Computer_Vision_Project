import tkinter as tk
from tkinter import ttk, Scale, IntVar
import cv2
import numpy as np
from PIL import Image, ImageTk

def create_bg_removal_window(parent, img, callback_fn):
    if img is None:
        return
    
    # สร้างหน้าต่างย่อย
    window = tk.Toplevel(parent)
    window.title("Background Removal")
    window.geometry("800x600")
    window.minsize(800, 600)
    
    # ตัวแปรสำหรับปรับแต่งการลบพื้นหลัง
    blur_var = IntVar(value=5)
    threshold_var = IntVar(value=0)
    iterations_var = IntVar(value=5)
    method_var = tk.StringVar(value="simple")
    
    # สร้างสำเนาของภาพต้นฉบับ
    original_img = img.copy()
    current_img = img.copy()
    
    # ฟังก์ชันสำหรับแสดงผลภาพ
    def display_preview(img_to_show):
        nonlocal current_img
        current_img = img_to_show.copy()
        
        # สร้างภาพสำหรับแสดงผล
        img_rgb = cv2.cvtColor(img_to_show, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # ปรับขนาดให้พอดีกับ canvas
        canvas_width = preview_canvas.winfo_width()
        canvas_height = preview_canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 600
        if canvas_height <= 1:
            canvas_height = 400
        
        img_pil.thumbnail((canvas_width, canvas_height))
        img_tk = ImageTk.PhotoImage(img_pil)
        
        # แสดงภาพบน canvas
        preview_canvas.delete("all")
        preview_canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
        preview_canvas.image = img_tk
    
    # ฟังก์ชันสำหรับประมวลผลภาพ
    def process_image():
        from Feature_Ex.cv_pro import bgremove1, bgremove_smooth, bgremove_grabcut
        
        method = method_var.get()
        
        if method == "simple":
            result = bgremove1(original_img)
        elif method == "smooth":
            blur = blur_var.get()
            thresh_offset = threshold_var.get()
            result = bgremove_smooth(original_img, blur, thresh_offset)
        elif method == "grabcut":
            iterations = iterations_var.get()
            result = bgremove_grabcut(original_img, iterations)
        
        display_preview(result)
    
    # ฟังก์ชันเมื่อกดปุ่ม Apply
    def on_apply():
        if callback_fn:
            callback_fn(current_img)
        window.destroy()
    
    # ฟังก์ชันเมื่อกดปุ่ม Cancel
    def on_cancel():
        window.destroy()
    
    # ฟังก์ชันเมื่อเปลี่ยนวิธีการลบพื้นหลัง
    def on_method_change():
        method = method_var.get()
        
        # ซ่อนหรือแสดง frame ต่างๆ ตามวิธีที่เลือก
        if method == "simple":
            smooth_frame.pack_forget()
            grabcut_frame.pack_forget()
        elif method == "smooth":
            smooth_frame.pack(fill="x", padx=10, pady=5)
            grabcut_frame.pack_forget()
        elif method == "grabcut":
            smooth_frame.pack_forget()
            grabcut_frame.pack(fill="x", padx=10, pady=5)
        
        # ประมวลผลภาพใหม่
        process_image()
    
    # สร้าง UI
    main_frame = ttk.Frame(window, padding=10)
    main_frame.pack(fill="both", expand=True)
    
    # ส่วนควบคุมด้านซ้าย
    control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
    control_frame.pack(side="left", fill="y", padx=5, pady=5)
    
    # เลือกวิธีการลบพื้นหลัง
    method_frame = ttk.Frame(control_frame)
    method_frame.pack(fill="x", pady=10)
    
    ttk.Label(method_frame, text="Method:").pack(anchor="w")
    
    ttk.Radiobutton(method_frame, text="Simple", variable=method_var, 
                   value="simple", command=on_method_change).pack(anchor="w", padx=20)
    
    ttk.Radiobutton(method_frame, text="Smooth", variable=method_var, 
                   value="smooth", command=on_method_change).pack(anchor="w", padx=20)
    
    ttk.Radiobutton(method_frame, text="GrabCut", variable=method_var, 
                   value="grabcut", command=on_method_change).pack(anchor="w", padx=20)
    
    # ส่วนควบคุมสำหรับวิธี Smooth
    smooth_frame = ttk.LabelFrame(control_frame, text="Smooth Settings", padding=10)
    
    # Blur Amount
    blur_frame = ttk.Frame(smooth_frame)
    blur_frame.pack(fill="x", pady=5)
    
    ttk.Label(blur_frame, text="Blur:").pack(side="left")
    
    blur_scale = Scale(blur_frame, from_=3, to=21, orient="horizontal", 
                      variable=blur_var, command=lambda _: process_image())
    blur_scale.pack(side="right", fill="x", expand=True)
    
    # Threshold Offset
    threshold_frame = ttk.Frame(smooth_frame)
    threshold_frame.pack(fill="x", pady=5)
    
    ttk.Label(threshold_frame, text="Threshold:").pack(side="left")
    
    threshold_scale = Scale(threshold_frame, from_=-50, to=50, orient="horizontal", 
                           variable=threshold_var, command=lambda _: process_image())
    threshold_scale.pack(side="right", fill="x", expand=True)
    
    # ส่วนควบคุมสำหรับวิธี GrabCut
    grabcut_frame = ttk.LabelFrame(control_frame, text="GrabCut Settings", padding=10)
    
    # Iterations
    iterations_frame = ttk.Frame(grabcut_frame)
    iterations_frame.pack(fill="x", pady=5)
    
    ttk.Label(iterations_frame, text="Iterations:").pack(side="left")
    
    iterations_scale = Scale(iterations_frame, from_=1, to=10, orient="horizontal", 
                            variable=iterations_var, command=lambda _: process_image())
    iterations_scale.pack(side="right", fill="x", expand=True)
    
    # ปุ่ม Process
    process_button = ttk.Button(control_frame, text="Process", command=process_image)
    process_button.pack(fill="x", pady=10)
    
    # ปุ่ม Apply และ Cancel
    button_frame = ttk.Frame(control_frame)
    button_frame.pack(fill="x", pady=20)
    
    ttk.Button(button_frame, text="Apply", command=on_apply).pack(side="left", padx=5)
    ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side="right", padx=5)
    
    # ส่วนแสดงผลด้านขวา
    preview_frame = ttk.LabelFrame(main_frame, text="Preview", padding=10)
    preview_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)
    
    preview_canvas = tk.Canvas(preview_frame, bg="lightgray")
    preview_canvas.pack(fill="both", expand=True)
    
    # แสดงภาพต้นฉบับ
    preview_canvas.bind("<Configure>", lambda e: display_preview(current_img))
    
    # เริ่มต้นด้วยวิธี Simple
    on_method_change()
    
    # ทำให้หน้าต่างอยู่ตรงกลางของ parent
    window.update_idletasks()
    width = window.winfo_width()
    height = window.winfo_height()
    x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (width // 2)
    y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (height // 2)
    window.geometry(f"{width}x{height}+{x}+{y}")
    
    # จับโฟกัส
    window.grab_set()
    window.focus_set()
    window.wait_window()