import torch
from ultralytics import YOLO
from mobile_sam import build_sam_vit_t, SamPredictor


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        
    def load_yolo_model(self, weights_path, task='detect'):
        """加载YOLO模型"""
        try:
            if weights_path.lower().endswith('.engine'):
                # TensorRT引擎
                model = YOLO(weights_path, task=task)
            else:
                # PyTorch模型
                model = YOLO(weights_path)
                
            # 移动到指定设备
            if self.device == "cuda" and torch.cuda.is_available():
                if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
                    model.model.to(self.device)
                    
            return model
        except Exception as e:
            print(f"加载YOLO模型失败: {e}")
            return None
            
    def load_msam_model(self, weights_path):
        """加载MobileSAM模型"""
        try:
            model = build_sam_vit_t()
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model.eval()
            
            if self.device == "cuda" and torch.cuda.is_available():
                model.to(self.device)
                
            return SamPredictor(model)
        except Exception as e:
            print(f"加载MobileSAM模型失败: {e}")
            return None
            
    def set_device(self, device_name):
        """设置推理设备"""
        device_name = device_name.lower()
        if device_name == "cuda" and not torch.cuda.is_available():
            print("CUDA 不可用，已回退到 CPU")
            self.device = "cpu"
        else:
            self.device = device_name
            
    def get_available_devices(self):
        """获取可用设备列表"""
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        return devices
