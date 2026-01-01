
import numpy as np
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
from image_utils import load_image, edge_detection

# 1. טעינת התמונה (תוודא שיש לך קובץ בשם original_image.jpg באותה תיקייה)
original_img = load_image('original_image.jpg')

# 2. ניקוי רעשים בעזרת Median Filter
# הערה: median filter עובד הכי טוב על כל ערוץ בנפרד או על הגרייסקייל
# כאן נבצע ניקוי רעשים לפני זיהוי הקצוות
clean_image = median(original_img, ball(3))

# 3. זיהוי קצוות
edges = edge_detection(clean_image)

# 4. בינאריזציה (Thresholding)
# נבחר ערך סף - למשל 100 (ניתן לשנות את זה לפי התמונה שלך)
threshold = 100
binary_edges = (edges > threshold).astype(np.uint8) * 255

# 5. שמירת התוצאה
output_image = Image.fromarray(binary_edges)
output_image.save('my_edges.png')

print("Success! The edge-detected image has been saved as 'my_edges.png'")
