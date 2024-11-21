import os
import cv2
#import numpy as np
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.functional import pil_to_tensor

def Threshold(image):
    """將圖片進行Threshold處理，返回二值化後的圖片"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 轉換為灰階圖    
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)  # 高斯模糊

    # 二值化處理（反向二值化）
    _, thresh_inv = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
    # 二值化處理（正常二值化）
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

    # 使用指定閾值進行Threshold處理
    thresh, thresh_inv = Threshold(image, thresh_value=150)  # 設置為150或其他值

    return thresh, thresh_inv

def image_resnet50(folder_path):
    """對資料夾中的每張圖片進行ResNet50分類，並執行Threshold處理"""
    # 初始化ResNet50模型和權重
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # 初始化預處理轉換
    preprocess = weights.transforms()

    # 儲存所有原始圖像和處理後的Threshold圖像
    original_images = []
    threshold_images = []

    # 讀取資料夾中的所有圖片
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            # 讀取圖片並轉換為RGB
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 儲存原始圖像
            original_images.append(image)

            # 顯示原始圖像
            cv2.imshow(f"Original Image: {file_name}", image)
            cv2.waitKey(0)

            # 使用PIL轉換為張量並應用預處理
            img_tensor = Image.fromarray(image_rgb)
            img_tensor = preprocess(img_tensor).unsqueeze(0)

            # 推論原始圖像
            prediction = model(img_tensor).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            category_name = weights.meta["categories"][class_id]
            print(f"{file_name} -> {category_name}: {100 * score:.1f}%")
        except Exception as e:
            print(f"Error processing file '{file_name}': {str(e)}")

    # 進行Threshold處理並顯示結果
    for image in original_images:
        try:
            # 進行Threshold處理
            thresh, thresh_inv = Threshold(image)

            # 顯示Threshold圖像
            cv2.imshow("Threshold Binary", thresh)
            cv2.imshow("Threshold Binary Inverse", thresh_inv)
            cv2.waitKey(0)

            # 轉換二值化圖像為RGB格式
            thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(thresh_rgb)

            # 應用ResNet50模型進行推論
            img_tensor = pil_image.convert("RGB")
            img_tensor = preprocess(img_tensor).unsqueeze(0)

            # 推論二值化後的圖像
            prediction = model(img_tensor).squeeze(0).softmax(0)
            class_id = prediction.argmax().item()
            score = prediction[class_id].item()
            category_name = weights.meta["categories"][class_id]
            print(f"Threshold Image -> {category_name}: {100 * score:.1f}%")

            # 儲存處理後的Threshold圖像
            threshold_images.append(thresh)

        except Exception as e:
            print(f"Error processing thresholded image: {str(e)}")

    # 等待按鍵後關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 指定資料夾路徑
folder_path = r"D:\program\Project\ImageProcess\dataset"

# 呼叫函式來處理資料夾中的圖片
image_resnet50(folder_path)
