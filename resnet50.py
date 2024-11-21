import os
import cv2
#import numpy as np
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms.functional import pil_to_tensor

def image_resnet50(folder_path):
    """對資料夾中的每張圖片進行ResNet50分類，並執行Threshold處理"""
    # 初始化ResNet50模型和權重
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # 初始化預處理轉換
    preprocess = weights.transforms()

    # 讀取資料夾中的所有圖片
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        try:
            # 讀取圖片並轉換為RGB
            image = cv2.imread(file_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    # 等待按鍵後關閉所有視窗
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 指定資料夾路徑
folder_path = r"D:\program\Project\ImageProcess\dataset"

# 呼叫函式來處理資料夾中的圖片
image_resnet50(folder_path)
