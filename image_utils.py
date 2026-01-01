

import numpy as np
from PIL import Image
from scipy.signal import convolve2d

def load_image(path):
    # טעינת התמונה בעזרת PIL והפיכה למערך numpy
    img = Image.open(path)
    return np.array(img)

def edge_detection(image_array):
    # 1. המרה לגרייסקייל (ממוצע של שלושת ערוצי הצבע)
    if len(image_array.shape) == 3:
        gray_img = np.mean(image_array, axis=2)
    else:
        gray_img = image_array

    # 2. הגדרת קרנלים לזיהוי קצוות (Sobel-like)
    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])
    
    kernelX = np.array([[-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1]])

    # 3. ביצוע קונבולוציה (עם ריפוד אפסים כדי לשמור על הגודל)
    edgeY = convolve2d(gray_img, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_img, kernelX, mode='same', boundary='fill', fillvalue=0)

    # 4. חישוב המגניטודה של הקצוות
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    
    return edgeMAG
