import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def read_image(file_path):
    img = cv2.imread(file_path) 
    return img

def Grayscale_Luminosity(img_cv):
    if img_cv.dtype != np.uint8:
        img_cv = np.clip(img_cv, 0, 255).astype(np.uint8)
        
    b, g, r = cv2.split(img_cv)
    gray_luminosity = 0.299*r + 0.587*g + 0.114*b
    gray_img = np.stack((gray_luminosity, gray_luminosity, gray_luminosity), axis=-1)  
    return gray_img

def BackAndWhite(img_cv):
    if img_cv.dtype != np.uint8:
        img_cv = np.clip(img_cv, 0, 255).astype(np.uint8)
        
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    thresh, img_black = cv2.threshold(img_gray, 135, 255, cv2.THRESH_BINARY)
    img_black_3ch = cv2.cvtColor(img_black, cv2.COLOR_GRAY2BGR)
    return img_black_3ch

def check_image_properties(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return "Error: Cannot load image."

    from PIL import Image
    try:
        pil_img = Image.open(file_path)
        dpi = pil_img.info.get('dpi', (96, 96))  # ใช้ค่า default 96 ถ้าไม่มีข้อมูล
    except Exception:
        dpi = (96, 96)  # ถ้าเกิดข้อผิดพลาด ใช้ค่า default
    
    height, width = img.shape[:2]

    properties_list = []
    
    properties_list.append(f"Dimension: {width} x {height} px")
    properties_list.append(f"Width: {width} px")
    properties_list.append(f"Height: {height} px")

    properties_list.append(f"Horizontal Resolution: {dpi[0]} dpi")
    properties_list.append(f"Vertical Resolution: {dpi[1]} dpi")

    if len(img.shape) == 2:  # Grayscale
        bit_depth = img.dtype.itemsize * 8
    elif len(img.shape) == 3:  # RGB, RGBA
        bit_depth = img.dtype.itemsize * 8 * img.shape[2]

    if bit_depth:
        properties_list.append(f"Bit Depth: {bit_depth}-bit")

    file_size = os.path.getsize(file_path)
    size_kb = file_size / 1024
    if size_kb < 1024:
        properties_list.append(f"File Size: {size_kb:.2f} KB")
    else:
        size_mb = size_kb / 1024
        properties_list.append(f"File Size: {size_mb:.2f} MB")
    
    file_format = os.path.splitext(file_path)[1].upper()[1:]
    properties_list.append(f"Format: {file_format}")
    
    return "\n".join(properties_list) if properties_list else "No valid properties found."

def adjust_brightness(img, value):
    """Adjust the brightness of the image."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v = np.clip(v, 0, 255)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(img, value):
    # แปลงค่า value เป็นค่า alpha
    # -100 -> 0.5, 0 -> 1.0, 100 -> 2.0
    alpha = 1.0 + (value / 300.0)  # ปรับช่วงให้เหมาะสมกว่าเดิม
    
    # ปรับ contrast โดยใช้จุดกึ่งกลาง 128 เป็นจุดอ้างอิง
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=128 * (1 - alpha))
    
    return adjusted

def adjust_saturation(img, value):
    # แปลงค่า value เป็น factor (-100 -> 0.0, 0 -> 1.0, 100 -> 2.0)
    factor = 1.0 + (value / 250.0)
    
    # แปลงเป็น HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # ปรับ saturation โดยใช้การคูณ (multiplicative scaling)
    s = s * factor
    s = np.clip(s, 0, 255)
    
    # รวมช่องสีและแปลงกลับ
    hsv = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_temperature(img, value):
    # แปลงค่า value เป็นเปอร์เซ็นต์ของการปรับ
    factor = value / 250.0
    
    # แยกช่องสี
    b, g, r = cv2.split(img.astype(np.float32))
    
    # ปรับสมดุลสีแบบ color temperature
    if factor > 0:  # อุ่นขึ้น (เพิ่มสีส้ม)
        r = r * (1 + factor * 0.5)  # เพิ่มสีแดงน้อยลง
        b = b * (1 - factor * 0.3)  # ลดสีน้ำเงิน
    else:  # เย็นลง (เพิ่มสีน้ำเงิน)
        r = r * (1 + factor * 0.3)  # ลดสีแดง
        b = b * (1 - factor * 0.5)  # เพิ่มสีน้ำเงินน้อยลง

    adjusted = cv2.merge([b, g, r])
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    return adjusted

def adjust_highlights(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # สร้าง weight map สำหรับ highlights แบบ gradual
    max_v = np.max(v)
    highlight_weight = np.clip((v - 127) / (max_v - 127), 0, 1)
    
    # ปรับค่า factor (-100 -> 0.5, 0 -> 1.0, 100 -> 1.5)
    factor = 1.0 + (value / 250.0)
    
    # ปรับค่าความสว่างตาม weight
    adjustment = (v * factor - v) * highlight_weight
    v = v + adjustment
    
    v = np.clip(v, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v])
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_shadows(img, value):
    # แปลงเป็น HSV และใช้ float32 สำหรับการคำนวณ
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)
    
    # สร้าง weight map สำหรับ shadows แบบ gradual
    min_v = np.min(v)
    shadow_weight = np.clip((127 - v) / (127 - min_v), 0, 1)
    
    # ปรับค่า factor (-100 -> 0.5, 0 -> 1.0, 100 -> 1.5)
    factor = 1.0 + (value / 200.0)
    
    # ปรับค่าความสว่างตาม weight
    adjustment = (v * factor - v) * shadow_weight
    v = v + adjustment
    
    # clip และแปลงกลับเป็น uint8
    v = np.clip(v, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v])
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_vibrance(img, value):
    # แปลงเป็น HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # คำนวณค่าความอิ่มตัวเฉลี่ย
    sat_mean = np.mean(s)
    
    # ปรับค่า factor ตามความอิ่มตัวปัจจุบัน
    factor = 1.0 + (value / 200.0)
    
    # ปรับแต่งความอิ่มตัวแบบไม่เท่ากัน
    adjustment = (1 - (s / 255.0)) * factor
    s = np.clip(s * (1 + adjustment), 0, 255).astype(np.uint8)
    
    # รวมช่องสีและแปลงกลับ
    adjusted = cv2.merge([h, s, v])
    return cv2.cvtColor(adjusted, cv2.COLOR_HSV2BGR)

def adjust_color_channel(img, color, value):
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    
    # ปรับค่า factor แบบไม่เป็นเชิงเส้น เพื่อให้การควบคุมละเอียดขึ้น
    if value >= 0:
        factor = 1.0 + (value / 300.0) ** 0.8
    else:
        factor = 1.0 / (1.0 + (abs(value) / 300.0) ** 0.8)
    
    if color == 'B':
        b = np.clip(b * factor, 0, 255)
        # ลดสีอื่นเล็กน้อยเพื่อรักษาสมดุล
        g = np.clip(g * (1 - abs(value) / 500), 0, 255)
        r = np.clip(r * (1 - abs(value) / 500), 0, 255)
    elif color == 'G':
        g = np.clip(g * factor, 0, 255)
        b = np.clip(b * (1 - abs(value) / 500), 0, 255)
        r = np.clip(r * (1 - abs(value) / 500), 0, 255)
    elif color == 'R':
        r = np.clip(r * factor, 0, 255)
        b = np.clip(b * (1 - abs(value) / 500), 0, 255)
        g = np.clip(g * (1 - abs(value) / 500), 0, 255)
    
    return cv2.merge([b, g, r]).astype(np.uint8)

def bgremove1(img):
    """
    ลบพื้นหลังโดยใช้ Otsu Thresholding
    
    Parameters:
    - img: ภาพต้นฉบับ (BGR)
    
    Returns:
    - finalimage: ภาพที่ลบพื้นหลังแล้ว
    """
    import cv2
    import numpy as np
    
    # สร้างสำเนาของภาพ
    myimage = img.copy()
    
    # Blur to image to reduce noise
    myimage = cv2.GaussianBlur(myimage, (5, 5), 0)
    
    # We bin the pixels. Result will be a value 1..5
    bins = np.array([0, 51, 102, 153, 204, 255])
    myimage[:, :, :] = np.digitize(myimage[:, :, :], bins, right=True) * 51
    
    # Create single channel greyscale for thresholding
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
    
    # Perform Otsu thresholding and extract the background.
    # We use Binary Threshold as we want to create an all white background
    ret, background = cv2.threshold(myimage_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    
    # Perform Otsu thresholding and extract the foreground.
    # We use TOZERO_INV as we want to keep some details of the foreground
    ret, foreground = cv2.threshold(myimage_grey, 0, 255, cv2.THRESH_TOZERO_INV + cv2.THRESH_OTSU)
    
    # Currently foreground is only a mask
    foreground = cv2.bitwise_and(myimage, myimage, mask=foreground)
    
    # Combine the background and foreground to obtain our final image
    finalimage = background + foreground
    
    return finalimage

def bgremove_smooth(img, blur_amount=5, threshold_offset=0):
    """
    ลบพื้นหลังโดยใช้ Otsu Thresholding พร้อมปรับความเรียบและค่า threshold
    
    Parameters:
    - img: ภาพต้นฉบับ (BGR)
    - blur_amount: ความแรงของการเบลอ (odd value >= 3)
    - threshold_offset: ค่าชดเชย threshold (-50 to +50)
    
    Returns:
    - finalimage: ภาพที่ลบพื้นหลังแล้ว
    """
    import cv2
    import numpy as np
    
    # สร้างสำเนาของภาพ
    myimage = img.copy()
    
    # ตรวจสอบว่า blur_amount เป็นเลขคี่
    if blur_amount % 2 == 0:
        blur_amount += 1
    
    if blur_amount < 3:
        blur_amount = 3
    
    # Blur to image to reduce noise
    myimage = cv2.GaussianBlur(myimage, (blur_amount, blur_amount), 0)
    
    # We bin the pixels. Result will be a value 1..5
    bins = np.array([0, 51, 102, 153, 204, 255])
    myimage[:, :, :] = np.digitize(myimage[:, :, :], bins, right=True) * 51
    
    # Create single channel greyscale for thresholding
    myimage_grey = cv2.cvtColor(myimage, cv2.COLOR_BGR2GRAY)
    
    # ใช้ Otsu's method เพื่อหาค่า threshold อัตโนมัติ
    otsu_thresh, _ = cv2.threshold(myimage_grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # ปรับค่า threshold ด้วย offset
    adjusted_thresh = otsu_thresh + threshold_offset
    adjusted_thresh = max(0, min(255, adjusted_thresh))
    
    # ใช้ค่า threshold ที่ปรับแล้วกับภาพเทา
    _, background = cv2.threshold(myimage_grey, adjusted_thresh, 255, cv2.THRESH_BINARY)
    
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    
    # ใช้ค่า threshold ที่ปรับแล้วกับภาพเทา (สำหรับ foreground)
    _, foreground_mask = cv2.threshold(myimage_grey, adjusted_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # Currently foreground is only a mask
    foreground = cv2.bitwise_and(myimage, myimage, mask=foreground_mask)
    
    # Combine the background and foreground to obtain our final image
    finalimage = background + foreground
    
    return finalimage

def bgremove_grabcut(img, iterations=5):
    # สร้างสำเนาของภาพ
    img_copy = img.copy()
    
    # ดึงขนาดภาพ
    height, width = img.shape[:2]
    
    # สร้างหน้ากากเริ่มต้น
    mask = np.zeros(img.shape[:2], np.uint8)
    
    # กำหนดพารามิเตอร์สำหรับ grabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # กำหนดสี่เหลี่ยมที่น่าจะมีวัตถุอยู่ (เลือกบริเวณตรงกลางภาพ)
    margin = int(min(width, height) * 0.1)  # ขอบของสี่เหลี่ยม
    rect = (margin, margin, width - margin * 2, height - margin * 2)
    
    # ดำเนินการ grabCut
    cv2.grabCut(img_copy, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    
    # แปลงค่าในหน้ากาก
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    
    # สร้างหน้ากากสำหรับพื้นหลัง
    bg_mask = 1 - mask2
    
    # สร้างพื้นหลังสีขาว
    white_bg = np.ones_like(img_copy) * 255
    
    # ใช้หน้ากากกับพื้นหลังสีขาว
    bg = white_bg * bg_mask[:, :, np.newaxis]
    
    # ใช้หน้ากากกับวัตถุด้านหน้า
    fg = img_copy * mask2[:, :, np.newaxis]
    
    # รวมวัตถุด้านหน้ากับพื้นหลังสีขาว
    result = cv2.add(bg, fg)
    
    return result