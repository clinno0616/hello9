# utils/image_utils.py
import io
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def save_image(image: Image.Image, filename: str, output_dir: str = "outputs") -> str:
    """
    保存生成的圖片
    
    Args:
        image: PIL Image 對象
        filename: 檔案名稱
        output_dir: 輸出目錄
        
    Returns:
        保存的檔案路徑
    """
    try:
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成檔案路徑
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{timestamp}_{filename}")
        
        # 保存圖片（使用最高質量）
        image.save(filepath, format="PNG", quality=100, optimize=False)
        logger.info(f"圖片已保存至: {filepath}")
        
        return filepath
        
    except Exception as e:
        logger.error(f"保存圖片時發生錯誤: {str(e)}")
        return ""

def get_download_link(image: Image.Image) -> Optional[bytes]:
    """
    生成圖片下載用的位元組數據
    
    Args:
        image: PIL Image 對象
        
    Returns:
        圖片的位元組數據
    """
    try:
        # 創建一個位元組緩衝區
        buf = io.BytesIO()
        
        # 保存圖片到緩衝區（使用最高質量）
        image.save(buf, format="PNG", quality=100, optimize=False)
        
        # 獲取位元組數據
        byte_data = buf.getvalue()
        
        return byte_data
        
    except Exception as e:
        logger.error(f"生成下載連結時發生錯誤: {str(e)}")
        return None

def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """
    調整圖片大小，使用高質量的重新採樣
    
    Args:
        image: PIL Image 對象
        width: 目標寬度
        height: 目標高度
        
    Returns:
        調整大小後的圖片
    """
    try:
        # 使用 LANCZOS 重採樣方法獲得最佳質量
        resized_image = image.resize((width, height), Image.LANCZOS)
        return resized_image
    except Exception as e:
        logger.error(f"調整圖片大小時發生錯誤: {str(e)}")
        return image

def get_scaled_dimensions(original_width: int, original_height: int, scale_percent: float) -> Tuple[int, int]:
    """
    計算縮放後的尺寸
    
    Args:
        original_width: 原始寬度
        original_height: 原始高度
        scale_percent: 縮放百分比 (100-200)
        
    Returns:
        (新寬度, 新高度)的元組
    """
    scale_factor = scale_percent / 100.0
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    return new_width, new_height

def add_watermark(image: Image.Image, text: str) -> Image.Image:
    """
    為圖片添加浮水印
    
    Args:
        image: PIL Image 對象
        text: 浮水印文字
        
    Returns:
        添加浮水印後的圖片
    """
    try:
        # 創建一個可以繪製的圖片副本
        img_with_watermark = image.copy()
        draw = ImageDraw.Draw(img_with_watermark)
        
        # 設置字體和大小（根據圖片大小調整）
        font_size = min(image.width, image.height) // 25
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # 獲取文字大小
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 計算文字位置（右下角）
        x = image.width - text_width - 10
        y = image.height - text_height - 10
        
        # 繪製半透明背景
        padding = 5
        draw.rectangle(
            [x-padding, y-padding, x+text_width+padding, y+text_height+padding],
            fill=(0, 0, 0, 128)
        )
        
        # 繪製文字
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
        
        return img_with_watermark
        
    except Exception as e:
        logger.error(f"添加浮水印時發生錯誤: {str(e)}")
        return image
    
def get_image_info(image: Image.Image) -> dict:
    """
    獲取圖片的基本信息
    
    Args:
        image: PIL Image 對象
        
    Returns:
        包含圖片信息的字典
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format if image.format else "未知",
        "size_kb": len(get_download_link(image)) // 1024 if image else 0
    }