import cv2
import numpy as np
import mediapipe as mp

class FaceWarper:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True, 
            max_num_faces=1, 
            refine_landmarks=False
        )
        
        # 目標移動點：眉毛
        self.eyebrow_idx = [65, 52, 53, 295, 282, 283]
        # 固定錨點：臉部輪廓 (防止臉部與背景崩壞)
        self.jaw_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                        397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
                        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.left_eye_idx = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_idx = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # 1. 眉毛上方的過渡區 (跟著稍微往上，模擬皮膚擠壓產生抬頭紋的效果)
        self.eyebrow_transition_idx = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
        
        # 2. 額頭中上段固定點 (當作結界，防止拉扯蔓延到頭髮與背景)
        self.forehead_fixed_idx = [71, 68, 104, 69, 108, 151, 337, 299, 333, 298, 301]
        
        # 3. 鼻樑與眉心固定點 (防止兩眼之間的皮膚跟著眉毛不自然地往上飛)
        self.nose_bridge_idx = [9, 8, 168, 6]

    def get_landmarks(self, image):
        h, w = image.shape[:2]
        results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark, h, w

    # 注意參數名稱把 eyebrow_lift_pixels 改成了 lift_ratio
    def create_expression(self, image, landmarks, h, w, lift_ratio):
        source_pts = []
        target_pts = []
        
        # --- 新增：動態測量臉部高度 ---
        # landmarks[152] 是下巴最底端，landmarks[10] 是額頭最頂端
        # Y 座標相減後乘上圖片總高度 h，就能得到這張臉的「實際像素高度」
        face_pixel_height = (landmarks[152].y - landmarks[10].y) * h
        
        # 將臉部高度乘上比例，得出這張照片真正該移動的像素量
        actual_lift_pixels = face_pixel_height * lift_ratio
        
        # 1. 圖片四個角落作為絕對錨點
        corners = [[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]]
        for c in corners:
            source_pts.append(c)
            target_pts.append(c)
            
        # 2. 加入臉部輪廓作為固定錨點
        for idx in self.jaw_idx:
            pt = [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            source_pts.append(pt)
            target_pts.append(pt)
            
        # 2.5 加入眼睛輪廓作為固定錨點 (你剛剛加上的)
        for idx in self.left_eye_idx + self.right_eye_idx:
            pt = [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            source_pts.append(pt)
            target_pts.append(pt) 

        # 2.6 加入額頭與鼻樑作為「完全固定」的錨點
        for idx in self.forehead_fixed_idx + self.nose_bridge_idx:
            pt = [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            source_pts.append(pt)
            target_pts.append(pt)  # 座標不變，完全鎖死
         
        # 3. 加入眉毛作為移動點
        for idx in self.eyebrow_idx:
            pt = [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            source_pts.append(pt)
            # 這裡改用動態計算出來的 actual_lift_pixels
            target_pts.append([pt[0], pt[1] - actual_lift_pixels])
        # 4. 新增：加入眉毛上方的過渡區 (只移動 50% 距離)
        for idx in self.eyebrow_transition_idx:
            pt = [int(landmarks[idx].x * w), int(landmarks[idx].y * h)]
            source_pts.append(pt)
            # 乘以 0.5 讓這區的皮膚產生被擠壓的感覺，而不會像塑膠一樣被死板地拉長
            target_pts.append([pt[0], pt[1] - (actual_lift_pixels * 0.5)])
            
        # (後面轉換 Numpy Array 與執行 TPS 的程式碼維持不變)
        src_arr = np.array(source_pts, dtype=np.float32).reshape(1, -1, 2)
        dst_arr = np.array(target_pts, dtype=np.float32).reshape(1, -1, 2)
        
        tps = cv2.createThinPlateSplineShapeTransformer()
        matches = [cv2.DMatch(i, i, 0) for i in range(src_arr.shape[1])]
        tps.estimateTransformation(dst_arr, src_arr, matches)
        
        return tps.warpImage(image)