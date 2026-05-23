import cv2
import time
import torch
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from mobile_sam import build_sam_vit_t, SamPredictor


class VideoThread(QThread):
    """视频处理线程"""
    change_pixmap_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(float)
    progress_signal = pyqtSignal(int, int)  
    
    def __init__(self, model_type="yoloseg"):
        super().__init__()
        self.model_type = model_type
        self.running = False
        self.paused = False  
        self.cap = None
        self.model = None
        self.msam_predictor = None
        self.yolo_model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo_weights_path = ""
        self.msam_weights_path = ""
        self.class_filter = "all"  # all, person, car, plane
        # 减少 MSAM 调用频率
        self.frame_index = 0
        self.msam_interval = 1  # 每隔3帧调用一次 MSAM
        self.last_msam_contours = []  
        # 降低分辨率加速
        self.yolo_imgsz = 320  # YOLO 输入尺寸640,512,320
        self.msam_downscale = 0.5  # MSAM 下采样比例
        # 检测框平滑参数
        self.tracker_alpha = 0.5  # EMA 
        self.max_track_age = 10    
        self.tracks = []           
        self.msam_tracks = []      
        # 源视频帧率控制
        self.source_fps = 30.0
        self.is_video_file = False
        self.target_fps = 30.0  # 最终目标帧率30fps
        self.frame_time = 1.0 / self.target_fps  # 每帧时间间隔
        
        self.total_frames = 0
        self.current_frame_pos = 0
        self.seeking = False  
        
        self.using_tensorrt = False

    @staticmethod
    def _iou(box_a, box_b):
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter = inter_w * inter_h
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union = area_a + area_b - inter + 1e-6
        return inter / union

    def _match_and_smooth(self, boxes, classes, is_msam=False):
        """基于 IoU 的贪心匹配，并对匹配到的框做 EMA 平滑。
        返回与输入顺序对应的平滑后框列表（若无匹配则返回原框）。"""
        tracks = self.msam_tracks if is_msam else self.tracks
        alpha = float(self.tracker_alpha)

        # 将旧 track 年龄+1
        for t in tracks:
            t["age"] += 1

        # 记录匹配结果
        smoothed_boxes = []
        used_track = set()

        for i, (box, cls_idx) in enumerate(zip(boxes, classes)):

            try:
                class_name = str(int(cls_idx))
            except Exception:
                class_name = str(cls_idx)

            # 在相同类别 track 中寻找 IoU 最大者
            best_j = -1
            best_iou = 0.0
            for j, t in enumerate(tracks):
                if j in used_track:
                    continue
                if t["cls"] != class_name:
                    continue
                iou = self._iou(box, t["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_j >= 0 and best_iou > 0.1:
                t = tracks[best_j]
                ema = alpha * box + (1.0 - alpha) * t["ema"]
                t["box"] = box
                t["ema"] = ema
                t["age"] = 0
                used_track.add(best_j)
                smoothed_boxes.append(ema)
            else:
                # 新建 track
                tracks.append({
                    "box": box.copy(),
                    "ema": box.copy(),
                    "cls": class_name,
                    "age": 0
                })
                smoothed_boxes.append(box)

        # 清理不需要的track
        tracks[:] = [t for t in tracks if t["age"] <= self.max_track_age]

        if is_msam:
            self.msam_tracks = tracks
        else:
            self.tracks = tracks

        return smoothed_boxes
        
    def set_model(self, model_type, yolo_weights="", msam_weights=""):
        """设置模型"""
        self.model_type = model_type
        try:
            self.yolo_weights_path = yolo_weights or self.yolo_weights_path
            self.msam_weights_path = msam_weights or self.msam_weights_path
            self.using_tensorrt = yolo_weights.lower().endswith('.engine') if yolo_weights else False

            if model_type == "yoloseg":
                if yolo_weights:
                    if self.using_tensorrt:
                        self.model = YOLO(yolo_weights, task='segment')
                    else:
                        self.model = YOLO(yolo_weights)
                else:
                    self.model = YOLO("yolov8n-seg.pt")  
            elif model_type == "yolo+msam":
                if yolo_weights and msam_weights:
                    if self.using_tensorrt:
                        self.yolo_model = YOLO(yolo_weights, task='detect')
                    else:
                        self.yolo_model = YOLO(yolo_weights)
                    msam_model = build_sam_vit_t()
                    state_dict = torch.load(msam_weights, map_location="cpu")
                    msam_model.load_state_dict(state_dict)
                    msam_model.eval()
                    self.msam_predictor = SamPredictor(msam_model)
                else:
                    # 使用默认模型
                    self.yolo_model = YOLO("yolov11n.pt")
                    msam_model = build_sam_vit_t()
                    self.msam_predictor = SamPredictor(msam_model)

            if self.device == "cuda" and torch.cuda.is_available():
                try:
                    if self.model is not None and hasattr(self.model, "model") and isinstance(self.model.model, torch.nn.Module):
                        self.model.model.to(self.device)
                    if self.yolo_model is not None and hasattr(self.yolo_model, "model") and isinstance(self.yolo_model.model, torch.nn.Module):
                        self.yolo_model.model.to(self.device)
                    if self.msam_predictor is not None and hasattr(self.msam_predictor, "model"):
                        self.msam_predictor.model.to(self.device)
                except Exception as move_e:
                    print(f"模型移动到 {self.device} 失败: {move_e}")
        except Exception as e:
            print(f"模型加载失败: {e}")

    def set_class_filter(self, class_name: str):
        """设置类别过滤: all/person/car/plane/bike"""
        allowed = {"all", "person", "car", "plane", "bike"}
        self.class_filter = class_name if class_name in allowed else "all"

    def set_device(self, device_name: str):
        """设置推理设备: 'cpu' 或 'cuda'"""
        device_name = device_name.lower()
        if device_name == "cuda" and not torch.cuda.is_available():
            print("CUDA 不可用，已回退到 CPU")
            self.device = "cpu"
        else:
            self.device = device_name
        try:
            if self.model is not None and hasattr(self.model, "model"):
                self.model.model.to(self.device)
            if self.yolo_model is not None and hasattr(self.yolo_model, "model"):
                self.yolo_model.model.to(self.device)
            if self.msam_predictor is not None and hasattr(self.msam_predictor, "model"):
                self.msam_predictor.model.to(self.device)
        except Exception as e:
            print(f"切换设备失败: {e}")
    
    def set_video_source(self, source):
        """设置视频源"""
        if self.cap:
            self.cap.release()
        
        if source == "camera":
            self.cap = cv2.VideoCapture(0)
            self.is_video_file = False
            # 相机：尝试读取实际FPS，若无则回退30
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0 or np.isnan(fps):
                fps = 30.0
            self.source_fps = float(fps)
            self.target_fps = self.source_fps
            self.frame_time = 1.0 / max(1e-6, self.target_fps)
            self.total_frames = 0
        else:
            self.cap = cv2.VideoCapture(source)
            self.is_video_file = True
            # 从视频读取原始fps
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0 or np.isnan(fps):
                fps = 30.0
            self.source_fps = float(fps)
            # 严格按源视频FPS播放
            self.target_fps = self.source_fps
            self.frame_time = 1.0 / max(1e-6, self.target_fps)
            # 获取总帧数
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
        if not self.cap.isOpened():
            print("无法打开视频源")
            return False
        return True
    
    def seek_video(self, frame_number):
        """跳转到指定帧"""
        if self.cap and self.is_video_file:
            self.seeking = True
            
            was_paused = self.paused
            self.paused = True
            
            # 对于 TensorRT 模型，需要特殊处理
            if self.using_tensorrt:
                self.reinitialize_model()
            
            # 跳转到指定帧
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            # 重置跟踪器状态
            self.tracks = []
            self.msam_tracks = []
            self.last_msam_contours = []
            self.frame_index = 0
            
            # 恢复之前的状态
            self.paused = was_paused
            self.seeking = False
    
    def reinitialize_model(self):
        """重新初始化模型，解决 TensorRT 上下文问题"""
        if self.model_type == "yoloseg" and self.yolo_weights_path:
            try:
                self.model = None
                if self.using_tensorrt:
                    self.model = YOLO(self.yolo_weights_path, task='segment')
                else:
                    self.model = YOLO(self.yolo_weights_path)
                    
                # 重新设置设备
                if self.device == "cuda" and torch.cuda.is_available():
                    if self.model is not None and hasattr(self.model, "model") and isinstance(self.model.model, torch.nn.Module):
                        self.model.model.to(self.device)
            except Exception as e:
                print(f"重新初始化 YOLO 模型失败: {e}")
        
        elif self.model_type == "yolo+msam" and self.yolo_weights_path and self.msam_weights_path:
            try:
                self.yolo_model = None
                self.msam_predictor = None
                
                if self.using_tensorrt:
                    self.yolo_model = YOLO(self.yolo_weights_path, task='detect')
                else:
                    self.yolo_model = YOLO(self.yolo_weights_path)
                    
                msam_model = build_sam_vit_t()
                state_dict = torch.load(self.msam_weights_path, map_location="cpu")
                msam_model.load_state_dict(state_dict)
                msam_model.eval()
                self.msam_predictor = SamPredictor(msam_model)
                
                # 重新设置设备
                if self.device == "cuda" and torch.cuda.is_available():
                    if self.yolo_model is not None and hasattr(self.yolo_model, "model") and isinstance(self.yolo_model.model, torch.nn.Module):
                        self.yolo_model.model.to(self.device)
                    if self.yolo_model is not None and hasattr(self.yolo_model, "model"):
                        self.yolo_model.model.to(self.device)
                    if self.msam_predictor is not None and hasattr(self.msam_predictor, "model"):
                        self.msam_predictor.model.to(self.device)
            except Exception as e:
                print(f"重新初始化 YOLO+MSAM 模型失败: {e}")
    
    def run(self):
        """运行视频处理"""
        self.running = True
        frame_count = 0
        start_time = time.time()
        last_frame_time = start_time
        # 用于按原视频时间表输出
        scheduled_start = start_time
        frames_emitted = 0
        
        while self.running:
            if self.cap and self.cap.isOpened():
                if self.paused:
                    time.sleep(0.01)  
                    # 暂停期间重置起始基准，避免恢复时瞬间丢很多帧
                    scheduled_start = time.time() - frames_emitted * self.frame_time
                    continue
                
                if self.seeking:
                    time.sleep(0.01)
                    continue
                    
                # 如为视频文件且落后于时间表，尝试丢帧追帧（只grab不解码处理）
                if self.is_video_file:
                    now = time.time()
                    # 目标应当已经显示的帧序号
                    should_have_emitted = int((now - scheduled_start) / self.frame_time)
                    if should_have_emitted > frames_emitted:
                        skips = should_have_emitted - frames_emitted
                        # 抓取并丢弃skips-1帧（当前帧仍要处理）
                        for _ in range(max(0, skips - 1)):
                            if not self.cap.grab():
                                break
                            frames_emitted += 1

                ret, frame = self.cap.read()
                if not ret:
                    break
                
                if self.is_video_file:
                    self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.progress_signal.emit(self.current_frame_pos, self.total_frames)
                
                processed_frame = self.process_frame(frame)
                
                # 计算FPS
                frame_count += 1
                self.frame_index += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / max(1e-6, elapsed)
                    self.fps_signal.emit(fps)
                
                # 发送处理后的帧
                self.change_pixmap_signal.emit(processed_frame)
                frames_emitted += 1
                
                if self.is_video_file:
                    # 跟随原视频帧率丢帧跟随
                    scheduled_time = scheduled_start + frames_emitted * self.frame_time
                    now = time.time()
                    if now < scheduled_time:
                        time.sleep(scheduled_time - now)
                else:
                    # 相机：不超过30fps的限速
                    current_time = time.time()
                    elapsed_since_last = current_time - last_frame_time
                    if elapsed_since_last < self.frame_time:
                        time.sleep(self.frame_time - elapsed_since_last)
                    last_frame_time = time.time()
        
        if self.cap:
            self.cap.release()
    
    def process_frame(self, frame):
        """处理单帧"""
        if self.model_type == "yoloseg":
            return self.process_yoloseg(frame)
        elif self.model_type == "yolo+msam":
            return self.process_yolo_msam(frame)
        return frame
    
    def process_yoloseg(self, frame):
        """YOLO-Seg处理"""
        if self.model:
            try:
                results = self.model(frame, verbose=False, device=self.device, imgsz=self.yolo_imgsz)
            except Exception as infer_e:
                print(f"YOLOSeg 推理失败: {infer_e}")
                return frame

            annotated_frame = frame.copy()

            for result in results:
                if result.masks is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    classes = result.boxes.cls.cpu().numpy()
                    polygons = result.masks.xy

                    # 先对所有框做一次平滑
                    smoothed = self._match_and_smooth(boxes.astype(np.float32), classes, is_msam=False)

                    for i, (poly, box, cls_idx, sbox) in enumerate(zip(polygons, boxes, classes, smoothed)):
                        try:
                            class_name = result.names[int(cls_idx)] if hasattr(result, 'names') else str(int(cls_idx))
                        except Exception:
                            class_name = str(int(cls_idx))
                        if self.class_filter != "all" and class_name != self.class_filter:
                            continue

                        color = (0, 0, 255)
                        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                        cv2.polylines(annotated_frame, [pts], isClosed=True, color=color, thickness=2)

                        # 不绘制辅助矩形，仅保留mask与标签

                        # 以mask质心放置标签（位置也会更稳定，因为矩形更稳）
                        M = cv2.moments(pts)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                        else:
                            x1, y1, x2, y2 = map(int, sbox)
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.putText(annotated_frame, class_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            return annotated_frame
        return frame
    
    def process_yolo_msam(self, frame):
        """YOLO + MobileSAM处理"""
        if self.yolo_model and self.msam_predictor:
            try:
                results = self.yolo_model(frame, verbose=False, device=self.device, imgsz=self.yolo_imgsz)
            except Exception as infer_e:
                print(f"YOLO 推理失败: {infer_e}")
                return frame
            boxes = results[0].boxes.xyxy.cpu().numpy()
            scores = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            annotated_frame = frame.copy()
            
            # 隔帧调用 MSAM，其余帧复用上次的 mask
            do_msam = (self.frame_index % self.msam_interval == 0)
            current_contours = []
            if do_msam:
                # 为 MSAM 准备下采样图像以减少计算量
                scale = float(self.msam_downscale)
                if scale <= 0 or scale >= 1.0:
                    scale = 0.75
                small_frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                # 设置一次图像供 MSAM 使用（下采样后）
                self.msam_predictor.set_image(small_frame)
                for box, score, cls in zip(boxes, scores, classes):
                    x1, y1, x2, y2 = map(int, box)
                    color = (0, 0, 255)
                    try:
                        class_name = results[0].names[int(cls)] if hasattr(results[0], 'names') else str(int(cls))
                    except Exception:
                        class_name = str(int(cls))
                    if self.class_filter != "all" and class_name != self.class_filter:
                        continue
                    # 使用 YOLO 框作为提示
                    # 将原始坐标缩放到下采样尺度
                    scaled_box = np.array([x1 * scale, y1 * scale, x2 * scale, y2 * scale], dtype=np.float32).reshape(1, 4)
                    box_np = scaled_box
                    masks, scores_pred, logits = self.msam_predictor.predict(
                        box=box_np,
                        point_coords=None,
                        point_labels=None,
                        multimask_output=False
                    )
                    mask_np = masks[0].astype(np.uint8)
                    if np.any(mask_np):
                        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        # 将轮廓从下采样尺度映射回原图尺度
                        if scale != 1.0:
                            mapped_contours = []
                            inv = 1.0 / scale
                            for cnt in contours:
                                cnt = cnt.astype(np.float32)
                                cnt[:, 0, :] *= inv
                                mapped_contours.append(cnt.astype(np.int32))
                            contours = mapped_contours
                        current_contours.append((contours, class_name))
                self.last_msam_contours = current_contours
            else:
                current_contours = self.last_msam_contours

            # 为 MSAM 结果构造矩形，并做平滑
            msam_boxes = []
            msam_classes = []
            for contours, class_name in current_contours:
                if len(contours) > 0:
                    x, y, w, h = cv2.boundingRect(contours[0])
                    msam_boxes.append(np.array([x, y, x + w, y + h], dtype=np.float32))
                    msam_classes.append(class_name)

            if len(msam_boxes) > 0:
                smoothed_boxes = self._match_and_smooth(np.array(msam_boxes), np.array(msam_classes), is_msam=True)
            else:
                smoothed_boxes = []

            # 绘制mask
            box_idx = 0
            for contours, class_name in current_contours:
                color = (0, 0, 255)
                cv2.drawContours(annotated_frame, contours, -1, color, 2)
                box_idx += 1
                # 取第一个轮廓的中心点作为标签位置（简化处理）
                if len(contours) > 0:
                    cnt = contours[0]
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(annotated_frame, class_name, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return annotated_frame
        return frame

    
    def pause(self):
        """暂停处理"""
        self.paused = True
        
    def resume(self):
        """恢复处理"""
        self.paused = False
        
    def stop(self):
        """停止处理"""
        self.running = False
        self.paused = False
        self.wait()
