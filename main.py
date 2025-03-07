# main.py
import tkinter as tk
from tkinter import filedialog, messagebox
from ttkbootstrap import Style
import cv2
import numpy as np
from PIL import Image, ImageTk
from Feature_Ex.cv_pro import *
from Feature_Ex.adjust_window import *
from Transformation.P2_Img_Trans import *
from More_Function.more import *
from Feature_Ex.adaptive_threshold import *
from Feature_Ex.bg_removal_window import create_bg_removal_window



# Main window
root = tk.Tk()
root.title("Photo Editor CV Project")

root.geometry("1920x1080")
root.resizable(True, True)

# canvas
CANVAS_WIDTH = 1000  
CANVAS_HEIGHT = 750  

style = Style(theme='darkly')

# Global variables
file_path = None
img_original = None
img_display = None
img_cv = None

# Open Image
def open_image():
    global file_path, img_original, img_display, img_cv
    file_path = filedialog.askopenfilename(
        title="Open Image File",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")]
    )
    if file_path:
        img_original = cv2.imread(file_path)
        if img_original is None:
            messagebox.showerror("Error", f"Unable to open image at {file_path}")
            return
        img_cv = img_original.copy()
        img_display = img_original.copy()
        display_image(img_display)
    else:
        messagebox.showerror("Error", "No image selected.")

# Show Image Properties
def show_image_properties():
    global file_path
    if file_path:
        try:
            properties = check_image_properties(file_path)
            messagebox.showinfo("Image Properties", properties)
        except Exception as e:
            messagebox.showerror("Error", f"Could not get image properties: {str(e)}")
    else:
        messagebox.showerror("Error", "No image loaded. Please open an image first.")

# Display Image
def display_image(img_cv):
    global img_display
    if img_cv is None:
        messagebox.showerror("Error", "No image to display.")
        return

    if img_cv.dtype != np.uint8:
        img_cv = np.clip(img_cv, 0, 255).astype(np.uint8)
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    
    if canvas_width <= 1:
        canvas_width = CANVAS_WIDTH
    if canvas_height <= 1:
        canvas_height = CANVAS_HEIGHT
    
    img_pil.thumbnail((canvas_width, canvas_height))
    img_tk = ImageTk.PhotoImage(img_pil)
    
    canvas.delete("all")
    canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=img_tk)
    canvas.image = img_tk

# Reset Image
def reset_image():
    global img_display, img_cv
    if img_original is not None:
        img_cv = img_original.copy()
        img_display = img_original.copy()
        display_image(img_display)

# Save Image
def save_image():
    if img_cv is None:
        messagebox.showerror("Error", "No image to save. Please open an image first.")
        return
        
    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        filetypes=[
            ("JPEG files", "*.jpg;*.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
    )
    
    if file_path:
        try:
            cv2.imwrite(file_path, img_cv)
            messagebox.showinfo("Success", "Image saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {str(e)}")

def apply_grayscale_luminosity():
    global img_display, img_cv, img_original
    if img_cv is None:  
        messagebox.showerror("Error", "No image loaded.")
        return
    
    img_display = Grayscale_Luminosity(img_cv) 
    img_cv = img_display.copy()
    display_image(img_display)

def apply_black_and_white():
    global img_cv, img_display
    if img_cv is None:  
        messagebox.showerror("Error", "No image loaded.")
        return
    
    create_adaptive_threshold_window(root, img_cv, on_black_white_apply) 

def on_black_white_apply(processed_img):
    global img_cv, img_display
    img_cv = processed_img.copy()
    img_display = processed_img.copy()
    display_image(img_display)
    
def on_adjust_apply(adjusted_img):
    global img_cv, img_display
    img_cv = adjusted_img
    img_display = adjusted_img.copy()
    display_image(img_display)

def open_adjust_window():
    create_adjust_window(root, img_cv, on_adjust_apply)

def on_transform_apply(transformed_img):
    global img_cv, img_display
    img_cv = transformed_img.copy()
    img_display = transformed_img.copy()
    display_image(img_display)

def open_trans_window():
    global img_cv, img_display
    if img_cv is None:
        messagebox.showerror("Error", "No image loaded. Please open an image first.")
        return
    open_transformation_window(root, img_cv, img_display, on_transform_apply)

def open_more_functions():
    global img_cv, img_display
    if img_cv is None:
        messagebox.showerror("Error", "No image loaded. Please open an image first.")
        return
    open_more_window(root, img_cv, on_more_apply)

def on_more_apply(processed_img):
    global img_cv, img_display
    img_cv = processed_img.copy()
    img_display = processed_img.copy()
    display_image(img_display)

def on_resize(event):
    if img_display is not None:
        display_image(img_display)
        
def open_bg_removal_window():
    global img_cv, img_display
    if img_cv is None:
        messagebox.showerror("Error", "No image loaded. Please open an image first.")
        return
    create_bg_removal_window(root, img_cv, on_bg_removal_apply)

def on_bg_removal_apply(processed_img):
    global img_cv, img_display
    img_cv = processed_img.copy()
    img_display = processed_img.copy()
    display_image(img_display)

frame = tk.Frame(root)
frame.pack(side="left", padx=10, pady=20, fill="y")

canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT)
canvas.pack(side="right", padx=10, pady=10, fill="both", expand=True)

canvas.bind("<Configure>", on_resize)

open_button = tk.Button(frame, text="Open Image", command=open_image, 
                       width=20, height=2, bg="#4CAF50", fg="white")
open_button.grid(row=0, column=0, padx=10, pady=5)

save_button = tk.Button(frame, text="Save Image", command=save_image, 
                       width=20, height=2, bg="#2196F3", fg="white")
save_button.grid(row=1, column=0, padx=10, pady=5)

properties_button = tk.Button(frame, text="Image Properties", command=show_image_properties, 
                            width=20, height=2, bg="#9C27B0", fg="white")
properties_button.grid(row=2, column=0, padx=10, pady=5)

reset_button = tk.Button(frame, text="Reset Image", command=reset_image, 
                        width=20, height=2, bg="#f44336", fg="white")
reset_button.grid(row=3, column=0, padx=10, pady=5)

grayscale_lum_button = tk.Button(frame, text="Gray", 
                                command=apply_grayscale_luminosity,
                                width=20, height=2, bg="#FF9800", fg="white")
grayscale_lum_button.grid(row=4, column=0, padx=10, pady=5)

BlackAndWhite_button = tk.Button(frame, text="Black & White", 
                              command=apply_black_and_white,
                              width=20, height=2, bg="#FF9800", fg="white")
BlackAndWhite_button.grid(row=5, column=0, padx=10, pady=5)

adjust_button = tk.Button(frame, text="Adjust Photo", command=open_adjust_window, 
                         width=20, height=2, bg="#2196F3", fg="white")
adjust_button.grid(row=6, column=0, padx=10, pady=5)

transform_button = tk.Button(frame, text="Image Transformation", 
                           command=open_trans_window,
                           width=20, height=2, bg="#2196F3", fg="white")
transform_button.grid(row=7, column=0, padx=10, pady=5)

more_functions_button = tk.Button(frame, text="More Functions", 
                                command=open_more_functions,
                                width=20, height=2, bg="#673AB7", fg="white")
more_functions_button.grid(row=8, column=0, padx=10, pady=5)

bg_removal_button = tk.Button(frame, text="Remove Background", 
                           command=open_bg_removal_window,
                           width=20, height=2, bg="#E91E63", fg="white")
bg_removal_button.grid(row=9, column=0, padx=10, pady=5)

status_label = tk.Label(root, text="", padx=20, pady=10, font=("Arial", 12))
status_label.pack(side="bottom")

root.mainloop()