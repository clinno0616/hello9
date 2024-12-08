# utils/model_utils.py
import os
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from typing import List, Optional, Dict, Any, Tuple
import gc
import logging
import shutil

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_base_model_path() -> str:
    """獲取基礎模型路徑"""
    # 首先檢查本地是否有 model 目錄
    if not os.path.exists("model"):
        os.makedirs("model", exist_ok=True)
        
    return "model"

def get_lora_path() -> str:
    """獲取 LoRA 模型路徑"""
    # 檢查和創建 lora 目錄
    lora_path = "lora"
    if not os.path.exists(lora_path):
        os.makedirs(lora_path, exist_ok=True)
    return lora_path

def scan_lora_models() -> List[Tuple[str, float]]:
    """
    掃描 LoRA 模型目錄
    
    Returns:
        List of tuples containing (lora_name, lora_weight)
    """
    lora_path = get_lora_path()
    lora_models = []
    
    try:
        for item in os.listdir(lora_path):
            item_path = os.path.join(lora_path, item)
            if os.path.isfile(item_path):
                ext = os.path.splitext(item)[1].lower()
                if ext in ['.safetensors', '.pt', '.bin']:
                    lora_models.append((item, 1.0))  # 預設權重為 1.0
                    
        logger.info(f"找到的 LoRA 模型: {lora_models}")
        return sorted(lora_models)
        
    except Exception as e:
        logger.error(f"掃描 LoRA 模型時發生錯誤: {str(e)}")
        return []

def download_default_model():
    """下載默認模型"""
    try:
        logger.info("開始下載默認模型...")
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        # 保存到本地
        base_path = get_base_model_path()
        save_path = os.path.join(base_path, "stable-diffusion-v1-5")
        pipe.save_pretrained(save_path)
        logger.info(f"默認模型已下載至: {save_path}")
        return True
    except Exception as e:
        logger.error(f"下載默認模型失敗: {str(e)}")
        return False

def scan_models() -> List[str]:
    """
    掃描模型目錄下的所有 Stable Diffusion 模型
    """
    base_path = get_base_model_path()
    model_list = []
    
    try:
        # 如果目錄為空，下載默認模型
        if not os.listdir(base_path):
            logger.info("模型目錄為空，準備下載默認模型...")
            if download_default_model():
                logger.info("默認模型下載完成")
            else:
                logger.error("默認模型下載失敗")
                return []

        # 掃描目錄
        for item in os.listdir(base_path):
            item_path = os.path.join(base_path, item)
            
            # 檢查目錄類型模型
            if os.path.isdir(item_path):
                config_files = ["model_index.json", "config.json"]
                if any(os.path.exists(os.path.join(item_path, cf)) for cf in config_files):
                    model_list.append(item)
                    
            # 檢查檔案類型模型
            elif os.path.isfile(item_path):
                ext = os.path.splitext(item)[1].lower()
                if ext in ['.ckpt', '.safetensors', '.bin', '.pth']:
                    model_list.append(item)

        logger.info(f"找到的模型: {model_list}")
        return sorted(model_list)
        
    except Exception as e:
        logger.error(f"掃描模型時發生錯誤: {str(e)}")
        return []

def clear_gpu_memory():
    """清理 GPU 記憶體"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def load_model(model_name: str) -> Optional[StableDiffusionPipeline]:
    """
    載入 Stable Diffusion 模型
    
    Args:
        model_name: 模型名稱
        
    Returns:
        StableDiffusionPipeline 實例或 None（如果載入失敗）
    """
    try:
        logger.info(f"開始載入模型:=============> {model_name}")
        
        # 清理 GPU 記憶體
        clear_gpu_memory()
        
        # 準備模型路徑
        base_path = get_base_model_path()
        model_path = os.path.join(base_path, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型: {model_path}")
            
        # 檢查 GPU 可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            logger.info("使用 GPU 進行推理")
        else:
            logger.info("使用 CPU 進行推理")
            
        # 設置模型載入參數
        dtype = torch.float16 if device == "cuda" else torch.float32
        load_options = {
            "torch_dtype": dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
            "use_safetensors": True
        }
        
        logger.info("開始載入模型到記憶體...")
        
        # 載入模型
        if os.path.isdir(model_path):
            logger.info("載入目錄型模型...")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                **load_options
            )
        else:
            logger.info("載入單檔案模型...")
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                **load_options
            )
        
        # 設置調度器
        logger.info("配置調度器...")
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )
        
        # 移動到指定設備
        logger.info(f"將模型移動到 {device}...")
        pipe = pipe.to(device)
        
        # 啟用記憶體優化
        if device == "cuda":
            logger.info("啟用記憶體優化...")
            pipe.enable_attention_slicing("auto")
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        
        logger.info("模型載入完成")
        return pipe
        
    except Exception as e:
        logger.error(f"載入模型失敗: {str(e)}")
        clear_gpu_memory()
        return None

def get_model_info(model_name: str) -> Dict[str, Any]:
    """
    獲取模型資訊
    
    Args:
        model_name: 模型名稱
        
    Returns:
        包含模型資訊的字典
    """
    base_path = get_base_model_path()
    model_path = os.path.join(base_path, model_name)
    
    info = {
        "name": model_name,
        "path": model_path,
        "exists": os.path.exists(model_path),
        "type": None,
        "size": 0,
        "is_valid": False
    }
    
    try:
        if info["exists"]:
            if os.path.isdir(model_path):
                info["type"] = "directory"
                info["size"] = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(model_path)
                    for filename in filenames
                )
            else:
                info["type"] = "file"
                info["size"] = os.path.getsize(model_path)
            
            info["size_gb"] = round(info["size"] / (1024**3), 2)
            info["is_valid"] = True
            
    except Exception as e:
        logger.error(f"獲取模型資訊失敗: {str(e)}")
        
    return info

def validate_model(model_name: str) -> Dict[str, Any]:
    """
    驗證模型檔案
    
    Args:
        model_name: 模型名稱
        
    Returns:
        驗證結果字典
    """
    result = {
        "valid": False,
        "error": None,
        "details": {}
    }
    
    try:
        info = get_model_info(model_name)
        result["details"] = info
        
        if not info["exists"]:
            raise FileNotFoundError("模型檔案不存在")
            
        min_size = 0.5 * (1024**3)  # 最小 500MB
        if info["size"] < min_size:
            raise ValueError(f"模型檔案過小: {info['size_gb']}GB")
            
        result["valid"] = True
        
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"模型驗證失敗: {str(e)}")
        
    return result

def load_model_with_lora(model_name: str, lora_models: List[Tuple[str, float]] = None) -> Optional[StableDiffusionPipeline]:
    """
    載入 Stable Diffusion 模型並應用 LoRA
    
    Args:
        model_name: 基礎模型名稱
        lora_models: List of tuples containing (lora_name, lora_weight)
        
    Returns:
        StableDiffusionPipeline 實例或 None（如果載入失敗）
    """
    try:
        logger.info(f"開始載入模型: {model_name}")
        
        # 清理 GPU 記憶體
        clear_gpu_memory()
        
        # 準備模型路徑
        base_path = get_base_model_path()
        model_path = os.path.join(base_path, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型: {model_path}")
            
        # 檢查 GPU 可用性
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # 載入基礎模型
        load_options = {
            "torch_dtype": dtype,
            "safety_checker": None,
            "requires_safety_checker": False,
            "use_safetensors": True
        }
        
        if os.path.isdir(model_path):
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                **load_options
            )
        else:
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                **load_options
            )
        
        # 移動到指定設備
        pipe = pipe.to(device)
        
        # 載入 LoRA 模型
        if lora_models and len(lora_models) > 0:
            logger.info(f"開始載入 LoRA 模型...")
            lora_path = get_lora_path()
            
            for lora_name, lora_weight in lora_models:
                lora_path_full = os.path.join(lora_path, lora_name)
                if os.path.exists(lora_path_full):
                    try:
                        logger.info(f"載入 LoRA: {lora_name} (weight: {lora_weight})")
                        # 直接使用 load_lora_weights，不使用 adapter_name
                        pipe.load_lora_weights(
                            lora_path_full,
                            weight_name=None  # 讓它自動檢測權重名稱
                        )
                        # 直接設置 cross attention scale
                        pipe.fuse_lora(lora_scale=float(lora_weight))
                        logger.info(f"成功應用 LoRA {lora_name} 權重: {lora_weight}")
                    except Exception as e:
                        logger.error(f"載入 LoRA {lora_name} 失敗: {str(e)}")
                        continue
        
        # 設置調度器
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )
        
        # 啟用記憶體優化
        if device == "cuda":
            pipe.enable_attention_slicing("auto")
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        
        logger.info("模型載入完成")
        return pipe
        
    except Exception as e:
        logger.error(f"載入模型失敗: {str(e)}")
        clear_gpu_memory()
        return None