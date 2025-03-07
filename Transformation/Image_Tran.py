import cv2
import numpy as np

def shear_image(image, shx, shy):
    rows, cols = image.shape[:2]
    
    abs_shx = abs(shx)
    abs_shy = abs(shy)
    
    new_width = int(cols + abs_shy * rows)
    new_height = int(rows + abs_shx * cols)
    
    temp_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    x_offset = int((new_width - cols) / 2)
    y_offset = int((new_height - rows) / 2)
    
    temp_img[y_offset:y_offset+rows, x_offset:x_offset+cols] = image
    
    M = np.float32([
        [1, shx, 0],
        [shy, 1, 0]
    ])
    
    M[0, 2] += (new_width - cols) / 2
    M[1, 2] += (new_height - rows) / 2
    
    sheared_img = cv2.warpAffine(temp_img, M, (new_width, new_height))
    
    start_x = int((new_width - cols) / 2)
    start_y = int((new_height - rows) / 2)
    result = sheared_img[start_y:start_y+rows, start_x:start_x+cols]
    
    return result

def translate_image(image, tx, ty):

    rows, cols = image.shape[:2]
    
    # สร้างเมทริกซ์การเลื่อน
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    

    return cv2.warpAffine(image, M, (cols, rows), borderValue=(0, 0, 0))

def scale_image(image, sx, sy):
    rows, cols = image.shape[:2]
    
    # คำนวณขนาดใหม่
    new_width = int(cols * sx)
    new_height = int(rows * sy)
    
    # ปรับขนาดรูปภาพ
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
    
    # สร้างภาพผลลัพธ์ขนาดเท่าต้นฉบับ
    result = np.zeros_like(image)
    
    # คำนวณตำแหน่งที่จะวางภาพที่ปรับขนาดแล้ว
    x_offset = max(0, int((cols - new_width) / 2))
    y_offset = max(0, int((rows - new_height) / 2))
    
    if new_width > cols or new_height > rows:
        crop_x = max(0, int((new_width - cols) / 2))
        crop_y = max(0, int((new_height - rows) / 2))
        
        cropped = resized[crop_y:crop_y+min(rows, new_height), crop_x:crop_x+min(cols, new_width)]
        
        place_width = min(cols, cropped.shape[1])
        place_height = min(rows, cropped.shape[0])
        
        result[:place_height, :place_width] = cropped[:place_height, :place_width]
    else:
        result[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
    
    return result

def rotate_image(image, angle):
    rows, cols = image.shape[:2]
    center = (cols/2, rows/2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # คำนวณขนาดใหม่ของภาพหลังการหมุน
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    
    new_width = int(rows * abs_sin + cols * abs_cos)
    new_height = int(rows * abs_cos + cols * abs_sin)
    
    # ปรับเมทริกซ์เพื่อให้จุดกึ่งกลางยังอยู่ตรงกลาง
    M[0, 2] += new_width/2 - center[0]
    M[1, 2] += new_height/2 - center[1]
    
    # หมุนภาพ
    rotated = cv2.warpAffine(image, M, (new_width, new_height))
    
    # ตัดภาพให้เท่ากับขนาดเดิม
    start_x = int((new_width - cols) / 2)
    start_y = int((new_height - rows) / 2)
    
    end_x = min(start_x + cols, new_width)
    end_y = min(start_y + rows, new_height)
    
    result = rotated[start_y:end_y, start_x:end_x]
    
    if result.shape[0] != rows or result.shape[1] != cols:
        result = cv2.resize(result, (cols, rows))
    
    return result