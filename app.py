# app.py

import os
import sys
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
from typing import Dict, Any, Tuple, List

# 確保可以找到 utils 模組
current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.append(current_dir)

# 基礎導入
import streamlit as st
import torch
from PIL import Image

# utils 模組導入
from utils.model_utils import scan_models, load_model, get_model_info, clear_gpu_memory, scan_lora_models, load_model_with_lora
from utils.image_utils import save_image, get_download_link, add_watermark, resize_image
from utils.ui_utils import (
    create_style,
    show_system_info,
    show_error_message,
    show_success_message,
    show_image_info,
    create_prompt_templates,
    create_advanced_settings,
    create_help_section
)

def initialize_session_state():
    """初始化 session state"""
    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'current_pipe' not in st.session_state:
        st.session_state.current_pipe = None
    if 'current_loras' not in st.session_state:
        st.session_state.current_loras = []
    if 'generation_history' not in st.session_state:
        st.session_state.generation_history = []
    if 'zoom_level' not in st.session_state:
        st.session_state.zoom_level = 100
    if 'original_image' not in st.session_state:
        st.session_state.original_image = None
    if 'adjustments' not in st.session_state:
        st.session_state.adjustments = {
            'contrast': 1.0,
            'warmth': 1.0,
            'sharpness': 1.0,
            'r': 1.0,
            'g': 1.0,
            'b': 1.0,
            'saturation': 1.0,
            'highlights': 1.0,
            'shadows': 1.0
        }
    if 'selected_loras' not in st.session_state:
        st.session_state.selected_loras = []


def adjust_image(image, contrast=1.0, warmth=1.0, sharpness=1.0, rgb=(1.0, 1.0, 1.0), 
                saturation=1.0, highlights=1.0, shadows=1.0):
    """
    調整圖片的各項參數
    
    Args:
        image: PIL Image 對象
        contrast: 對比度 (0.0-2.0)
        warmth: 色溫 (0.0-2.0)
        sharpness: 銳利度 (0.0-2.0)
        rgb: RGB調整值 (r, g, b) 每個值範圍 0.0-2.0
        saturation: 飽和度 (0.0-2.0)
        highlights: 亮部 (0.0-2.0)
        shadows: 暗部 (0.0-2.0)
    
    Returns:
        調整後的圖片
    """
    try:
        # 轉換為RGB模式（如果不是的話）
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # 創建圖片副本
        adjusted = image.copy()
        
        # 調整飽和度
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(adjusted)
            adjusted = enhancer.enhance(saturation)
        
        # 調整對比度
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(adjusted)
            adjusted = enhancer.enhance(contrast)
        
        # 調整色溫
        if warmth != 1.0:
            # 將圖片轉換為numpy數組
            img_array = np.array(adjusted)
            
            # 根據色溫調整RGB通道
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            if warmth > 1.0:
                # 增加暖色調
                factor = (warmth - 1.0) * 0.5
                r = np.clip(r * (1 + factor), 0, 255)
                g = np.clip(g * (1 + factor * 0.5), 0, 255)
                b = np.clip(b * (1 - factor * 0.5), 0, 255)
            else:
                # 增加冷色調
                factor = (1.0 - warmth) * 0.5
                r = np.clip(r * (1 - factor), 0, 255)
                g = np.clip(g * (1 - factor * 0.5), 0, 255)
                b = np.clip(b * (1 + factor), 0, 255)
            
            # 合併通道
            img_array = np.stack([r, g, b], axis=2).astype(np.uint8)
            adjusted = Image.fromarray(img_array)
        
        # 調整銳利度
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(adjusted)
            adjusted = enhancer.enhance(sharpness)
        
        # 調整RGB
        if rgb != (1.0, 1.0, 1.0):
            r, g, b = adjusted.split()
            r = r.point(lambda x: np.clip(x * rgb[0], 0, 255))
            g = g.point(lambda x: np.clip(x * rgb[1], 0, 255))
            b = b.point(lambda x: np.clip(x * rgb[2], 0, 255))
            adjusted = Image.merge('RGB', (r, g, b))
            
        # 調整亮部和暗部
        if highlights != 1.0 or shadows != 1.0:
            img_array = np.array(adjusted).astype(float)
            
            # 計算亮度遮罩
            luminance = 0.2989 * img_array[:,:,0] + 0.5870 * img_array[:,:,1] + 0.1140 * img_array[:,:,2]
            
            # 創建亮部和暗部遮罩
            highlight_mask = luminance / 255.0
            shadow_mask = 1.0 - highlight_mask
            
            # 應用亮部調整
            if highlights != 1.0:
                highlight_factor = highlights - 1.0
                for i in range(3):
                    img_array[:,:,i] += highlight_factor * 255 * highlight_mask
                    
            # 應用暗部調整
            if shadows != 1.0:
                shadow_factor = shadows - 1.0
                for i in range(3):
                    img_array[:,:,i] += shadow_factor * 255 * shadow_mask
            
            # 裁剪值到有效範圍
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            adjusted = Image.fromarray(img_array)
        
        return adjusted
        
    except Exception as e:
        st.error(f"調整圖片時發生錯誤: {str(e)}")
        return image

def set_page_config():
    st.set_page_config(
        page_title="Stable Diffusion Web UI",
        page_icon="🎨",
        layout="wide",  # 使用寬屏布局
        initial_sidebar_state="collapsed"  # 預設收起側邊欄
    )

def display_image_with_zoom(image, params):
    """
    顯示帶有縮放控制的圖片，圖片會填滿視窗寬度
    
    Args:
        image: PIL Image 對象
        params: 生成參數字典
    """
    # 添加自定義 CSS 來確保圖片填滿寬度
    st.markdown("""
        <style>
        /* 確保圖片容器填滿寬度 */
        .block-container {
            max-width: 70%;
            padding-left: 0rem;
            padding-right: 0rem;
        }
        
        /* 圖片容器樣式 */
        .css-1v0mbdj {
            width: 70%;
            max-width: none;
        }
        
        /* 確保圖片填滿容器 */
        .css-1v0mbdj img {
            width: 70%;
            height: auto;
        }
        
        /* 調整滑塊容器的寬度和位置 */
        .zoom-slider {
            width: 70%;
            margin: 0 auto;
            padding: 1rem;
        }
        
        /* 調整圖片說明文字的樣式 */
        .image-caption {
            text-align: center;
            padding: 0.5rem;
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 4px;
            margin-top: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # 創建一個容器用於滑塊
    with st.container():
        # 添加縮放控制滑塊
        zoom_col1, zoom_col2, zoom_col3 = st.columns([1, 2, 1])
        with zoom_col2:
            st.slider(
                "圖片縮放 (%)", 
                min_value=100, 
                max_value=400, 
                value=st.session_state.zoom_level,
                step=20,
                key="zoom_slider",
                help="拖動滑塊來調整圖片大小"
            )
    
    # 更新 session state 中的縮放級別
    st.session_state.zoom_level = st.session_state.zoom_slider
    
    # 計算縮放後的尺寸
    zoom_factor = st.session_state.zoom_level / 100.0
    display_width = int(params['width'] * zoom_factor)
    display_height = int(params['height'] * zoom_factor)
    
    # 調整圖片大小
    if zoom_factor != 1.0:
        display_image = image.resize((display_width, display_height), Image.LANCZOS)
    else:
        display_image = image
    
    # 創建一個容器來顯示圖片，使用全寬
    image_container = st.container()
    with image_container:
        # 顯示圖片（不使用列來分割，直接使用全寬）
        st.image(
            display_image,
            caption=f"生成的圖片 (原始: {params['width']}x{params['height']}, "
                    f"顯示: {display_width}x{display_height}, "
                    f"縮放: {st.session_state.zoom_level}%)",
            output_format="PNG",
            use_container_width=True  # 改用 use_column_width 來填滿寬度
        )
    
    return display_image

def create_sidebar():
    """創建側邊欄控制項"""
    with st.sidebar:
        st.title("模型設置")
        
        # 掃描並列出可用模型
        models = scan_models()
        selected_model = st.selectbox(
            "選擇基礎模型",
            models,
            index=0 if models else None
        )
        
        # 如果選擇了模型，顯示模型信息
        if selected_model:
            model_info = get_model_info(selected_model)
            st.info(f"模型大小: {model_info['size']:.2f} GB")
        
        # LoRA 模型選擇
        st.subheader("LoRA 模型設置")
        lora_models = scan_lora_models()
        
        if lora_models:
            # 創建多選框供使用者選擇多個 LoRA
            selected_loras = []
            for lora_name, _ in lora_models:
                if st.checkbox(f"使用 {lora_name}", key=f"lora_{lora_name}"):
                    # 為每個選中的 LoRA 創建權重滑塊
                    weight = st.slider(
                        f"{lora_name} 權重",
                        min_value=0.0,
                        max_value=2.0,
                        value=1.0,
                        step=0.05,
                        key=f"weight_{lora_name}"
                    )
                    selected_loras.append((lora_name, weight))
            
            st.session_state.selected_loras = selected_loras
        else:
            st.info("未找到 LoRA 模型。請將 LoRA 模型放置於 /lora 目錄中。")
        
        # 其他參數設置
        width = st.slider("寬度", 256, 1024, 768, 64)
        height = st.slider("高度", 256, 1024, 512, 64)
        cfg_scale = st.slider("CFG Scale", 1.0, 20.0, 7.0, 0.5)
        steps = st.slider("Sampling Steps", 1, 150, 20)
        seed = st.number_input("Seed (-1 為隨機)", -1, 999999999, -1)
        
        sampling_methods = [
            "Euler a", "Euler", "LMS", "Heun", "DPM2", "DPM2 a",
            "DPM++ 2S a", "DPM++ 2M", "DPM++ SDE", "DPM fast",
            "DPM adaptive", "LMS Karras", "DPM2 Karras",
            "DPM2 a Karras", "DPM++ 2S a Karras",
            "DPM++ 2M Karras", "DPM++ SDE Karras"
        ]
        sampler = st.selectbox("Sampling Method", sampling_methods)
        
        # 顯示系統資訊
        show_system_info()
        
        # 創建進階設定
        advanced_settings = create_advanced_settings()
        
        # 創建說明文件
        create_help_section()
        
        return selected_model, {
            "width": width,
            "height": height,
            "cfg_scale": cfg_scale,
            "steps": steps,
            "seed": seed,
            "sampler": sampler,
            **advanced_settings
        }

def create_main_ui():
    """創建主要使用者介面"""
    st.title("Stable Diffusion 圖片生成器")
    
    # 設置自定義樣式
    create_style()
    
    # 提示詞模板
    template = create_prompt_templates()
    
    # 提示詞輸入
    prompt = st.text_area(
        "輸入提示詞",
        height=100,
        value=template,
        placeholder="請輸入您想要生成的圖片描述..."
    )
    
    # 添加負面提示詞
    negative_prompt = st.text_area(
        "負面提示詞（可選）",
        height=100,
        placeholder="輸入不想在圖片中出現的元素..."
    )
    
    return prompt, negative_prompt

def save_generation_history(params):
    """保存生成歷史"""
    st.session_state.generation_history.append(params)
    if len(st.session_state.generation_history) > 10:  # 只保留最近10次
        st.session_state.generation_history.pop(0)

def show_generation_history():
    """顯示生成歷史"""
    if st.session_state.generation_history:
        with st.expander("生成歷史", expanded=False):
            for i, params in enumerate(reversed(st.session_state.generation_history), 1):
                st.write(f"### 生成 #{i}")
                st.json(params)

def should_reload_model(model: str, selected_loras: List[Tuple[str, float]]) -> bool:
    """
    檢查是否需要重新載入模型
    """
    # 模型變更時需要重新載入
    if st.session_state.current_model != model:
        return True
        
    # LoRA 配置變更時需要重新載入
    current_loras = st.session_state.current_loras
    if len(current_loras) != len(selected_loras):
        return True
        
    # 檢查每個 LoRA 的設置是否有變化
    current_loras_dict = {name: weight for name, weight in current_loras}
    selected_loras_dict = {name: weight for name, weight in selected_loras}
    if current_loras_dict != selected_loras_dict:
        return True
        
    return False


def main():
    try:
        # 初始化
        initialize_session_state()
       
        # 創建側邊欄和主介面
        model, params = create_sidebar()
        if not model:  # 如果沒有找到模型，直接返回
            st.error("未找到任何模型！請確認 /model 目錄中包含有效的模型文件。")
            return
            
        prompt, negative_prompt = create_main_ui()
        
        # 生成按鈕
        col1, col2 = st.columns([1, 1])
        with col1:
            generate_button = st.button("生成圖片", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("清除結果", type="secondary", use_container_width=True)
            
        if clear_button:
            st.session_state.generated_image = None
            st.session_state.generation_history = []
            st.rerun()
            
        if generate_button:
            if not prompt:
                show_error_message("請輸入提示詞")
                return
                
            with st.spinner("正在生成圖片..."):
                try:
                    selected_loras = st.session_state.selected_loras
                    
                    # 檢查是否需要重新載入模型
                    if should_reload_model(model, selected_loras):
                        #logger.info("模型或 LoRA 設置已變更，重新載入模型...")
                        pipe = load_model_with_lora(model, selected_loras)
                        if pipe is None:
                            show_error_message("模型載入失敗")
                            return
                        st.session_state.current_model = model
                        st.session_state.current_loras = selected_loras
                        st.session_state.current_pipe = pipe
                    
                    # 設置隨機種子
                    if params["seed"] == -1:
                        params["seed"] = torch.randint(0, 1000000, (1,)).item()
                    
                    # 準備生成參數
                    generation_params = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "width": params["width"],
                        "height": params["height"],
                        "guidance_scale": params["cfg_scale"],
                        "num_inference_steps": params["steps"],
                        "generator": torch.manual_seed(params["seed"])
                    }
                    
                    # 保存生成參數時加入 LoRA 資訊
                    generation_record = {
                        "model": model,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "seed": params["seed"],
                        "width": params["width"],
                        "height": params["height"],
                        "cfg_scale": params["cfg_scale"],
                        "steps": params["steps"],
                        "sampler": params["sampler"],
                        "loras": selected_loras  # 加入 LoRA 資訊
                    }

                    # 生成圖片
                    pipe = st.session_state.current_pipe
                    image = pipe(**generation_params).images[0]
                    
                    # 添加浮水印
                    if params["enable_watermark"]:
                        image = add_watermark(image, params["watermark_text"])
                    
                    # 自動保存
                    if params["enable_auto_save"]:
                        save_image(image, f"generation_{params['seed']}.png", params["output_dir"])
                    
                    # 保存到 session state
                    st.session_state.generated_image = image
                    
                    # 保存生成參數
                    generation_record = {
                        "model": model,
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "seed": params["seed"],
                        "width": params["width"],
                        "height": params["height"],
                        "cfg_scale": params["cfg_scale"],
                        "steps": params["steps"],
                        "sampler": params["sampler"]
                    }
                    save_generation_history(generation_record)
                    
                    # 保存到 session state
                    st.session_state.generated_image = image
                    st.session_state.original_image = image.copy()  # 保存原始圖片
                    
                    # 重置調整參數
                    st.session_state.adjustments = {
                        'contrast': 1.0,
                        'warmth': 1.0,
                        'sharpness': 1.0,
                        'r': 1.0,
                        'g': 1.0,
                        'b': 1.0,
                        'saturation': 1.0,
                        'highlights': 1.0,
                        'shadows': 1.0
                    }

                    show_success_message("圖片生成成功！")
                    
                except Exception as e:
                    show_error_message(f"生成圖片時發生錯誤: {str(e)}")
                    return
                    
        # 顯示生成的圖片
        if st.session_state.generated_image:
            # 使用新的圖片顯示函數
            display_image = display_image_with_zoom(
                st.session_state.generated_image, 
                st.session_state.generation_history[-1]
            )
            #st.image(st.session_state.generated_image, 
            #        caption="生成的圖片",
            #        use_container_width=True)
            
            # 下載和參數按鈕
            col1, col2 = st.columns([1, 1])
            with col1:
                # 下載按鈕 - 使用原始圖片（非縮放版本）
                img_data = get_download_link(st.session_state.generated_image)
                st.download_button(
                    label="下載原始圖片",
                    data=img_data,
                    file_name=f"sd_generation_{params['seed']}.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col2:
                # 下載當前縮放版本
                if st.session_state.zoom_level != 100:
                    zoom_img_data = get_download_link(display_image)
                    st.download_button(
                        label=f"下載縮放版本 ({st.session_state.zoom_level}%)",
                        data=zoom_img_data,
                        file_name=f"sd_generation_{params['seed']}_scaled.png",
                        mime="image/png",
                        use_container_width=True
                    )
                else:
                    # 複製參數按鈕
                    if st.button("複製參數", use_container_width=True):
                        st.code(str(st.session_state.generation_history[-1]))
            
            # 顯示生成參數
            show_image_info(st.session_state.generation_history[-1])
            
        # 顯示生成歷史
        show_generation_history()
        
        # 顯示生成的圖片和調整面板
        if st.session_state.generated_image:
            # 使用 container 來組織布局
            main_container = st.container()
            with main_container:
                # 圖片調整面板
                st.markdown("### 圖片調整")
                col1, col2, col3 = st.columns([1,1,1])
                with col1:
                    # 基本調整
                    st.session_state.adjustments['contrast'] = st.slider(
                        "對比度", 0.0, 2.0, 
                        st.session_state.adjustments['contrast'], 0.01,
                        help="調整圖片的對比度"
                    )
                    st.session_state.adjustments['warmth'] = st.slider(
                        "色溫", 0.0, 2.0, 
                        st.session_state.adjustments['warmth'], 0.01,
                        help="調整圖片的色溫（<1.0 冷色調，>1.0 暖色調）"
                    )
                    st.session_state.adjustments['sharpness'] = st.slider(
                        "銳利度", 0.0, 2.0, 
                        st.session_state.adjustments['sharpness'], 0.01,
                        help="調整圖片的銳利度"
                    )
                with col2:
                    # RGB 調整
                    st.session_state.adjustments['r'] = st.slider(
                        "紅色", 0.0, 2.0, 
                        st.session_state.adjustments['r'], 0.01,
                        help="調整紅色通道"
                    )
                    st.session_state.adjustments['g'] = st.slider(
                        "綠色", 0.0, 2.0, 
                        st.session_state.adjustments['g'], 0.01,
                        help="調整綠色通道"
                    )
                    st.session_state.adjustments['b'] = st.slider(
                        "藍色", 0.0, 2.0, 
                        st.session_state.adjustments['b'], 0.01,
                        help="調整藍色通道"
                    )
                with col3:
                    # 新增控制項
                    st.session_state.adjustments['saturation'] = st.slider(
                    "飽和度", 0.0, 2.0, 1.0, 0.01,
                    help="調整圖片的飽和度"
                    )
                    st.session_state.adjustments['highlights'] = st.slider(
                    "亮部", 0.0, 2.0, 1.0, 0.01,
                    help="調整圖片的亮部區域"
                    )
                    st.session_state.adjustments['shadows'] = st.slider(
                    "暗部", 0.0, 2.0, 1.0, 0.01,
                    help="調整圖片的暗部區域"
                    )
                    
                # 重置按鈕
                if st.button("重置調整", use_container_width=True):
                    st.session_state.adjustments = {
                        'contrast': 1.0,
                        'warmth': 1.0,
                        'sharpness': 1.0,
                        'r': 1.0,
                        'g': 1.0,
                        'b': 1.0,
                        'saturation': 1.0,
                        'highlights': 1.0,
                        'shadows': 1.0
                    }
                    #st.rerun()
                # 應用調整並顯示圖片
                adjusted_image = adjust_image(
                    st.session_state.original_image,
                    contrast=st.session_state.adjustments['contrast'],
                    warmth=st.session_state.adjustments['warmth'],
                    sharpness=st.session_state.adjustments['sharpness'],
                    rgb=(
                        st.session_state.adjustments['r'],
                        st.session_state.adjustments['g'],
                        st.session_state.adjustments['b']
                    ),
                    saturation=st.session_state.adjustments['saturation'],
                    highlights=st.session_state.adjustments['highlights'],
                    shadows=st.session_state.adjustments['shadows']
                )
                
                # 顯示調整後的圖片
                _, image_col, _ = st.columns([1, 8, 1])
                with image_col:
                    st.image(
                        adjusted_image,
                        caption=f"調整後的圖片 ({params['width']}x{params['height']})",
                        use_container_width=True
                    )
                    
                    # 下載按鈕
                    col1, col2 = st.columns(2)
                    with col1:
                        # 下載原始圖片
                        img_data = get_download_link(st.session_state.original_image)
                        st.download_button(
                            label="下載原始圖片",
                            data=img_data,
                            file_name=f"original_{params['seed']}.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    with col2:
                        # 下載調整後的圖片
                        adjusted_data = get_download_link(adjusted_image)
                        st.download_button(
                            label="下載調整後的圖片",
                            data=adjusted_data,
                            file_name=f"adjusted_{params['seed']}.png",
                            mime="image/png",
                            use_container_width=True
                        )
    except Exception as e:
        show_error_message(f"程式執行時發生錯誤: {str(e)}")
        
if __name__ == "__main__":
    main()