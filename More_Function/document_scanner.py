import cv2
import numpy as np

def opencv_resize(image, ratio):
    width = int(image.shape[1] * ratio)
    height = int(image.shape[0] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def approximate_contour(contour):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)

def get_receipt_contour(contours):
    # วนลูปผ่านเส้นขอบทั้งหมด
    for c in contours:
        approx = approximate_contour(c)
        # ถ้าเส้นขอบที่ประมาณแล้วมี 4 จุด สันนิษฐานว่าเป็นสี่เหลี่ยมของเอกสาร
        if len(approx) == 4:
            return approx
    return None

def contour_to_rect(contour, resize_ratio):
    """
    แปลงเส้นขอบเป็นจุดรูปสี่เหลี่ยมที่เรียงลำดับเป็น [top-left, top-right, bottom-right, bottom-left]
    """
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # จุดบนซ้ายมีผลรวมน้อยที่สุด
    # จุดล่างขวามีผลรวมมากที่สุด
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # คำนวณความแตกต่างระหว่างจุด:
    # จุดบนขวาจะมีค่าต่างน้อยที่สุด
    # จุดล่างซ้ายจะมีค่าต่างมากที่สุด
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect / resize_ratio

def wrap_perspective(img, rect):
    # แยกจุดสี่เหลี่ยม: บนซ้าย, บนขวา, ล่างขวา, ล่างซ้าย
    (tl, tr, br, bl) = rect
    
    # คำนวณความกว้างของภาพใหม่
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    
    # คำนวณความสูงของภาพใหม่
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    # ใช้ค่าสูงสุดของความกว้างและความสูงเพื่อกำหนดขนาดภาพสุดท้าย
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    # จุดปลายทางที่จะใช้ในการแปลงมุมมอง
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # คำนวณเมทริกซ์การแปลงมุมมอง
    M = cv2.getPerspectiveTransform(rect, dst)
    
    # แปลงมุมมองภาพ
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))

def scan_document(img, resize_height=500, block_size=11, c=7):

    if img is None or img.size == 0:
        return None
    
    # สร้างสำเนาของภาพต้นฉบับ
    original = img.copy()
    
    # คำนวณอัตราส่วนการปรับขนาด
    resize_ratio = resize_height / img.shape[0]
    img = opencv_resize(img, resize_ratio)
    
    # แปลงเป็นเทาและเบลอ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # ตรวจจับขอบ
    edged = cv2.Canny(gray, 75, 200)
    
    # หาเส้นขอบ
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # เรียงเส้นขอบตามขนาด (ใหญ่ไปเล็ก)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    # หาเส้นขอบของเอกสาร
    receipt_contour = get_receipt_contour(contours)
    
    if receipt_contour is None:
        # ถ้าไม่พบเส้นขอบที่เหมาะสม ให้ใช้ภาพทั้งหมด
        height, width = original.shape[:2]
        receipt_contour = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32).reshape(4, 1, 2)
    
    # แปลงเส้นขอบเป็นสี่เหลี่ยม
    rect = contour_to_rect(receipt_contour, resize_ratio)
    
    # แปลงมุมมองภาพ
    scanned = wrap_perspective(original, rect)
    
    # ใช้ adaptive threshold
    if block_size % 2 == 0:
        block_size += 1  # ต้องเป็นเลขคี่
    
    gray = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        block_size, 
        c
    )
    
    # แปลงกลับเป็น BGR
    bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    
    return bw_bgr