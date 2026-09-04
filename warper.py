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

        # --- 眨眼用的關鍵點 ---
        # 上眼皮 → 下眼皮 的配對 (往下拉上眼皮讓眼睛閉合)
        # 每個 tuple 為 (上眼皮點, 對應的下眼皮點)，涵蓋整圈眼皮，
        # 不能有漏掉的眼皮點，否則沒被約束的點會被 TPS 亂拉。
        self.blink_pairs = [
            # 左眼 (畫面右側)
            (246, 7), (161, 163), (160, 144), (159, 145),
            (158, 153), (157, 154), (173, 155),
            # 右眼 (畫面左側)
            (466, 249), (388, 390), (387, 373), (386, 374),
            (385, 380), (384, 381), (398, 382),
        ]
        # 眼角固定點 (閉眼時眼角不應該移動)
        self.eye_corner_idx = [33, 133, 362, 263]

        # 4. 中臉固定點 (鼻子、臉頰、嘴巴)。
        #    TPS 是全域平滑內插，沒有控制點的區域會被鄰近位移「拖著跑」，
        #    中臉若不釘住，眼皮的位移會一路蔓延到嘴巴與下巴。
        self.midface_fixed_idx = [
            1, 4, 5, 19, 94, 2, 195, 197,          # 鼻樑下段與鼻頭
            98, 327, 129, 358, 115, 344,           # 鼻翼兩側
            205, 425, 50, 280, 101, 330, 36, 266,  # 臉頰
            119, 348, 117, 346, 111, 340,          # 眼睛下方顴骨
            61, 291, 0, 17, 37, 267, 84, 314,      # 嘴巴外圈
        ]

        # --- 眨眼參數 (這三個值都是實測調出來的，調大很容易破圖) ---
        # BLINK_MAX_CLOSE: 上眼皮最多只能走完開口的這個比例。
        #   實測 0.90 以上眉眼之間必定出現漩渦狀破圖，0.75 已能讀成「閉眼」。
        # BLINK_MIN_GAP_PX: 上下眼皮控制點之間硬性保留的像素距離。
        #   兩者一旦重合，TPS 會出現「同一座標要對應到兩個不同目標」的矛盾約束
        #   而解出奇異解，整張臉會融成一團。
        # BLINK_BROW_FOLLOW: 眉毛跟著眼皮下沉的比例。
        #   眉毛完全釘死的話，眼皮下移會把眉眼之間的皮膚擠壓到極限而破圖。
        self.BLINK_MAX_CLOSE = 0.75
        self.BLINK_MIN_GAP_PX = 2.5
        self.BLINK_BROW_FOLLOW = 0.4

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

    def create_blink(self, image, landmarks, h, w, blink_ratio):
        """
        產生眨眼表情 (把上眼皮往下蓋)。

        blink_ratio:
            0.0 = 眼睛完全張開 (直接回傳原圖)
            1.0 = 閉合到最大幅度 (BLINK_MAX_CLOSE)
        搭配 math.sin(progress * math.pi) 使用可做出「閉眼 -> 睜眼」的完整循環。

        重要：上眼皮永遠不會真的碰到下眼皮，詳見 BLINK_MAX_CLOSE 的說明。
        """
        # ratio 為 0 時直接回傳原圖，省去不必要的 TPS 運算
        if blink_ratio <= 0.0:
            return image

        blink_ratio = min(1.0, blink_ratio)

        source_pts = []
        target_pts = []
        seen = set()

        def add(pt, target):
            # 影像被縮小後，相鄰的 landmark 可能會落在同一個像素上。
            # 座標重複的控制點會讓 TPS 的核矩陣出現重複列而變得奇異，一律去重。
            key = (round(pt[0], 1), round(pt[1], 1))
            if key in seen:
                return
            seen.add(key)
            source_pts.append(pt)
            target_pts.append(target)

        # 1. 圖片四個角落作為絕對錨點
        for c in ([0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0]):
            add(c, list(c))

        # 2. 先算好每個上眼皮點要往下移多少 (只動 y；x 不動，避免眼角被往內擠)
        lid_moves = []
        for upper_idx, lower_idx in self.blink_pairs:
            up_x, up_y = landmarks[upper_idx].x * w, landmarks[upper_idx].y * h
            low_y = landmarks[lower_idx].y * h
            eye_gap = low_y - up_y  # 這一組眼皮的開口高度
            # 開口比保留距離還小就別動，免得算出反方向的位移
            if eye_gap <= self.BLINK_MIN_GAP_PX:
                continue
            travel = min(eye_gap * self.BLINK_MAX_CLOSE * blink_ratio,
                         eye_gap - self.BLINK_MIN_GAP_PX)
            lid_moves.append((up_x, up_y, travel))

        # 3. 臉部輪廓、額頭、鼻樑、中臉、眼角全部鎖死，把變形侷限在眼睛附近
        for idx in (self.jaw_idx + self.forehead_fixed_idx + self.nose_bridge_idx +
                    self.eye_corner_idx + self.midface_fixed_idx):
            pt = [landmarks[idx].x * w, landmarks[idx].y * h]
            add(pt, list(pt))

        # 4. 眉毛跟著同一隻眼睛的眼皮往下沉一點 (取最近的上眼皮點，自動對應左右眼)
        for idx in self.eyebrow_idx + self.eyebrow_transition_idx:
            pt = [landmarks[idx].x * w, landmarks[idx].y * h]
            if lid_moves:
                nearest = min(lid_moves,
                              key=lambda m: (m[0] - pt[0]) ** 2 + (m[1] - pt[1]) ** 2)
                add(pt, [pt[0], pt[1] + nearest[2] * self.BLINK_BROW_FOLLOW])
            else:
                add(pt, list(pt))

        # 5. 下眼皮固定不動，當作上眼皮往下移動的參考線
        for _, lower_idx in self.blink_pairs:
            pt = [landmarks[lower_idx].x * w, landmarks[lower_idx].y * h]
            add(pt, list(pt))

        # 6. 上眼皮往下移
        for up_x, up_y, travel in lid_moves:
            add([up_x, up_y], [up_x, up_y + travel])

        src_arr = np.array(source_pts, dtype=np.float32).reshape(1, -1, 2)
        dst_arr = np.array(target_pts, dtype=np.float32).reshape(1, -1, 2)

        tps = cv2.createThinPlateSplineShapeTransformer()
        matches = [cv2.DMatch(i, i, 0) for i in range(src_arr.shape[1])]
        tps.estimateTransformation(dst_arr, src_arr, matches)

        return tps.warpImage(image)