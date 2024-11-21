import cv2
import numpy as np
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import os

# masking
def apply_mask(image, center, radius):
    # 創建遮罩
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    # 將遮罩應用到圖片上
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

# contours
def draw_contours(image, center, radius):
    top_left = (center[0] - radius, center[1] - radius)
    top_right = (center[0] + radius, center[1] - radius)
    bottom_right = (center[0] + radius, center[1] + radius)
    bottom_left = (center[0] - radius, center[1] + radius)
    # 創建正方形的輪廓
    square_contour = np.array([top_left, top_right, bottom_right, bottom_left])
    # 在圖片上繪製正方形輪廓
    image_with_contours = image.copy()
    cv2.polylines(image_with_contours, [square_contour], isClosed=True, color=(0, 255, 0), thickness=2)
    return image_with_contours

# cropping
def crop_contour(image, center, radius):
    top_left = (center[0] - radius, center[1] - radius)
    # 獲取裁剪區域的座標
    x, y = max(0, top_left[0]), max(0, top_left[1])
    w, h = min(2 * radius, image.shape[1] - x), min(2 * radius, image.shape[0] - y)
    # 裁剪圖像
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# 預測
def predice(image, model, preprocess, weights):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = Image.fromarray(image_rgb)
    img_tensor = preprocess(img_tensor).unsqueeze(0)

    prediction = model(img_tensor).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]

    return category_name, score

# 將結果放置於圖片上方
def put_text_on_image(image, text):
    image_with_text = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    color = (255, 255, 255)
    thickness = 2

    # 獲取文本尺寸
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (image_with_text.shape[1] - text_size[0]) // 2  # 水平居中
    text_y = text_size[1] + 10  # 距離頂部10像素

    # 放置文本
    cv2.putText(image_with_text, text, (text_x, text_y), font, font_scale, color, thickness)
    return image_with_text

# 主處理函數
def process_image(image_path, center, radius, model, preprocess, weights, output_path):
    original_image = cv2.imread(image_path)

    # 應用遮罩
    masked_image = apply_mask(original_image, center, radius)
    contours_image = draw_contours(masked_image, center, radius)
    cropping_image = crop_contour(contours_image, center, radius)

    # 分類並將結果添加到圖像上
    for img, label in [(original_image, "original_image"),
                       (masked_image, "masked_image"),
                       (contours_image, "contours_image"),
                       (cropping_image, "cropping_image")]:
        category, score = predice(img, model, preprocess, weights)
        text = f"{category}: {100 * score:.1f}%"
        img_with_text = put_text_on_image(img, text)
        cv2.imshow(f"{label} with prediction", img_with_text)
        print(f"{label} -> {text}")

        # 儲存處理後的影像
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_image_path  = os.path.join(output_path, f"{image_name}_{label}.jpg")
        cv2.imwrite(output_image_path, img_with_text)
        cv2.waitKey(1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # ResNet50
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()
    preprocess = weights.transforms()

    # 指定儲存路徑
    output_path = r'ImageProcess\IP_Project\result'
    os.makedirs(output_path, exist_ok=True)

    # 指定圖片路徑
    image_paths = [r"ImageProcess\IP_Project\dataset\1-error_processing.jpg",
                r"ImageProcess\IP_Project\dataset\2-low_accuracy.jpg",
                r"ImageProcess\IP_Project\dataset\3-high_accuracy.jpg"]
    center_list = [(575, 305), (100, 100), (250, 385)]
    radius_list = [100, 40, 100]

    for i in range(len(image_paths)):
        print(image_paths[i])
        process_image(image_paths[i], center_list[i], radius_list[i], model, preprocess, weights, output_path)
        print("")

if __name__ == "__main__":
    main()