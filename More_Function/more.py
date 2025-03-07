import tkinter as tk
from tkinter import ttk, messagebox, StringVar
from PIL import Image, ImageTk
from .morecv import *
from .document_scanner import scan_document
import cv2
import numpy as np

def open_more_window(parent, img, callback):
    more_window = tk.Toplevel(parent)
    more_window.title("More Functions")
    more_window.geometry("1000x700")
    more_window.minsize(800, 600)
    
    current_img = img.copy()
    current_preview = {'img_pil': None}
    original_img = img.copy()

    left_frame = tk.Frame(more_window, padx=10, pady=10)
    left_frame.pack(side="left", fill="y")

    preview_frame = tk.Frame(more_window, padx=10, pady=10)
    preview_frame.pack(side="right", fill="both", expand=True)

    preview_canvas = tk.Canvas(preview_frame, width=700, height=600, bg="#f0f0f0")
    preview_canvas.pack(fill="both", expand=True)

    # สร้าง Notebook สำหรับแยกแท็บแต่ละฟีเจอร์
    notebook = ttk.Notebook(left_frame)
    notebook.pack(fill="x", pady=10, expand=True)
    
    # แท็บต่างๆ
    pixel_tab = ttk.Frame(notebook, padding=10)
    sketch_tab = ttk.Frame(notebook, padding=10)
    document_tab = ttk.Frame(notebook, padding=10)
    
    notebook.add(pixel_tab, text="Pixel Art")
    notebook.add(sketch_tab, text="Sketch")
    notebook.add(document_tab, text="Document Scanner")

    def update_preview(processed_img):
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        current_preview['img_pil'] = Image.fromarray(img_rgb)
        
        canvas_width = preview_canvas.winfo_width()
        canvas_height = preview_canvas.winfo_height()
        
        if canvas_width <= 1:
            canvas_width = 700
        if canvas_height <= 1:
            canvas_height = 600
            
        img_display = current_preview['img_pil'].copy()
        img_display.thumbnail((canvas_width, canvas_height), Image.LANCZOS)
        
        img_tk = ImageTk.PhotoImage(img_display)
        preview_canvas.delete("all")
        preview_canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
        preview_canvas.image = img_tk

    def apply_pixel_art():
        nonlocal current_img
        try:
            pixel_size = int(pixel_size_var.get())
            color_levels = int(color_levels_var.get())
            if pixel_size < 1: pixel_size = 1
            if color_levels < 2: color_levels = 2
            
            # อัพเดทค่าใน Scale
            pixel_size_scale.set(pixel_size)
            color_levels_scale.set(color_levels)
            
            current_img = convert_to_pixel_art(original_img, pixel_size, color_levels)
            update_preview(current_img)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer values for Pixel Size and Color Levels.")
    
    def apply_cartoon():
        nonlocal current_img
        current_img = apply_cartoon_effect(original_img)
        update_preview(current_img)
    
    def apply_sketch():
        nonlocal current_img
        try:
            blur_size = int(sketch_blur_var.get())
            intensity = float(sketch_intensity_var.get())
            invert = sketch_invert_var.get()
            
            # Make sure blur size is odd
            if blur_size % 2 == 0:
                blur_size += 1
                sketch_blur_scale.set(blur_size)
                sketch_blur_var.set(str(blur_size))
            
            current_img = apply_sketch_effect(original_img, blur_size, intensity, invert)
            update_preview(current_img)
        except ValueError:
            messagebox.showerror("Error", "Please enter valid values for Blur Size and Intensity.")
    
    def apply_document_scan():
        nonlocal current_img
        try:
            # ใช้ฟังก์ชัน scan_document แบบอัตโนมัติ
            scanned = scan_document(original_img)
            if scanned is not None:
                current_img = scanned
                update_preview(current_img)
            else:
                messagebox.showerror("Error", "Failed to scan document. No document contour found.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def update_entry_from_scale(scale_val, var):
        var.set(str(int(float(scale_val))))
    
    def update_float_entry_from_scale(scale_val, var):
        # ปัดเลขทศนิยมให้เหลือ 1 ตำแหน่งเพื่อให้อ่านง่ายขึ้น
        rounded_val = round(float(scale_val), 1)
        var.set(str(rounded_val))
    
    def on_apply():
        callback(current_img)
        more_window.destroy()
    
    def on_cancel():
        more_window.destroy()
    
    def on_canvas_resize(event):
        if current_img is not None:
            update_preview(current_img)
    
    preview_canvas.bind("<Configure>", on_canvas_resize)
    
    # === Pixel Art Tab ===
    pixel_size_frame = tk.Frame(pixel_tab)
    pixel_size_frame.pack(fill="x", pady=5)
    
    ttk.Label(pixel_size_frame, text="Pixel Size:", width=15, anchor="w").pack(side="left")
    
    pixel_size_var = StringVar(value="8")
    pixel_size_entry = ttk.Entry(pixel_size_frame, textvariable=pixel_size_var, width=5)
    pixel_size_entry.pack(side="right")
    
    pixel_size_scale = ttk.Scale(pixel_size_frame, from_=1, to=50, orient="horizontal", length=200)
    pixel_size_scale.set(8)
    pixel_size_scale.pack(side="right", padx=5)
    
    pixel_size_scale.configure(command=lambda val: update_entry_from_scale(val, pixel_size_var))
    
    color_levels_frame = tk.Frame(pixel_tab)
    color_levels_frame.pack(fill="x", pady=5)
    
    ttk.Label(color_levels_frame, text="Color Levels:", width=15, anchor="w").pack(side="left")
    
    color_levels_var = StringVar(value="4")
    color_levels_entry = ttk.Entry(color_levels_frame, textvariable=color_levels_var, width=5)
    color_levels_entry.pack(side="right")
    
    color_levels_scale = ttk.Scale(color_levels_frame, from_=2, to=32, orient="horizontal", length=200)
    color_levels_scale.set(4)
    color_levels_scale.pack(side="right", padx=5)
    
    color_levels_scale.configure(command=lambda val: update_entry_from_scale(val, color_levels_var))
    
    btn_pixel = tk.Button(pixel_tab, text="Apply Pixel Art", command=apply_pixel_art,
                         height=2, bg="#4CAF50", fg="white")
    btn_pixel.pack(fill="x", padx=10, pady=10)
    
    btn_cartoon = tk.Button(pixel_tab, text="Apply Cartoon Effect", command=apply_cartoon,
                           height=2, bg="#2196F3", fg="white")
    btn_cartoon.pack(fill="x", padx=10, pady=5)
    
    # === Sketch Tab ===
    sketch_blur_frame = tk.Frame(sketch_tab)
    sketch_blur_frame.pack(fill="x", pady=5)
    
    ttk.Label(sketch_blur_frame, text="Blur Size:", width=15, anchor="w").pack(side="left")
    
    sketch_blur_var = StringVar(value="131")
    sketch_blur_entry = ttk.Entry(sketch_blur_frame, textvariable=sketch_blur_var, width=5)
    sketch_blur_entry.pack(side="right")
    
    sketch_blur_scale = ttk.Scale(sketch_blur_frame, from_=3, to=201, orient="horizontal", length=200)
    sketch_blur_scale.set(131)
    sketch_blur_scale.pack(side="right", padx=5)
    
    sketch_blur_scale.configure(command=lambda val: update_entry_from_scale(val, sketch_blur_var))
    
    sketch_intensity_frame = tk.Frame(sketch_tab)
    sketch_intensity_frame.pack(fill="x", pady=5)
    
    ttk.Label(sketch_intensity_frame, text="Intensity:", width=15, anchor="w").pack(side="left")
    
    sketch_intensity_var = StringVar(value="256.0")
    sketch_intensity_entry = ttk.Entry(sketch_intensity_frame, textvariable=sketch_intensity_var, width=5)
    sketch_intensity_entry.pack(side="right")
    
    sketch_intensity_scale = ttk.Scale(sketch_intensity_frame, from_=50.0, to=500.0, orient="horizontal", length=200)
    sketch_intensity_scale.set(256.0)
    sketch_intensity_scale.pack(side="right", padx=5)
    
    sketch_intensity_scale.configure(command=lambda val: update_float_entry_from_scale(val, sketch_intensity_var))
    
    sketch_invert_frame = tk.Frame(sketch_tab)
    sketch_invert_frame.pack(fill="x", pady=5)
    
    ttk.Label(sketch_invert_frame, text="Invert Effect:", width=15, anchor="w").pack(side="left")
    
    sketch_invert_var = tk.BooleanVar(value=True)
    sketch_invert_check = ttk.Checkbutton(sketch_invert_frame, variable=sketch_invert_var)
    sketch_invert_check.pack(side="left")
    
    btn_sketch = tk.Button(sketch_tab, text="Apply Sketch Effect", command=apply_sketch,
                          height=2, bg="#9C27B0", fg="white")
    btn_sketch.pack(fill="x", padx=10, pady=10)
    
    # === Document Scanner Tab ===
    doc_note = ttk.Label(document_tab, 
                         text="Document Scanner จะตรวจหาเอกสารในรูปภาพและปรับแก้มุมมอง\nพร้อมทั้งแปลงเป็นขาว-ดำเพื่อความชัดเจน", 
                         justify="center")
    doc_note.pack(pady=10)
    
    btn_scan = tk.Button(document_tab, text="Scan Document", command=apply_document_scan,
                       height=2, bg="#FF9800", fg="white")
    btn_scan.pack(fill="x", padx=10, pady=10)
    
    # เพิ่ม Event Handler สำหรับการกด Enter ใน Entry
    def entry_update(event):
        # ระบุว่าการอัพเดตมาจาก widget ไหน
        widget = event.widget
        if widget in (pixel_size_entry, color_levels_entry):
            apply_pixel_art()
        elif widget in (sketch_blur_entry, sketch_intensity_entry):
            apply_sketch()
    
    # ผูกกับทุก Entry
    pixel_size_entry.bind('<Return>', entry_update)
    color_levels_entry.bind('<Return>', entry_update)
    sketch_blur_entry.bind('<Return>', entry_update)
    sketch_intensity_entry.bind('<Return>', entry_update)
    
    # เพิ่มปุ่ม Apply และ Cancel
    button_frame = tk.Frame(left_frame)
    button_frame.pack(pady=20)
    
    apply_btn = tk.Button(button_frame, text="Apply", command=on_apply,
                         width=10, height=2, bg="#4CAF50", fg="white")
    apply_btn.pack(side="left", padx=5)
    
    cancel_btn = tk.Button(button_frame, text="Cancel", command=on_cancel,
                          width=10, height=2, bg="#f44336", fg="white")
    cancel_btn.pack(side="left", padx=5)
    
    update_preview(img)
    
    # ทำให้หน้าต่างอยู่ตรงกลางของ parent
    more_window.update_idletasks()
    width = more_window.winfo_width()
    height = more_window.winfo_height()
    x = parent.winfo_rootx() + (parent.winfo_width() // 2) - (width // 2)
    y = parent.winfo_rooty() + (parent.winfo_height() // 2) - (height // 2)
    more_window.geometry(f"{width}x{height}+{x}+{y}")