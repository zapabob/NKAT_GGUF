#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Kobold.cpp Backend Selector
ユーザー提供のチューニングレシピに基づく最適化エンジン
"""

import os
import json
import logging
from pathlib import Path

class NKATBackendSelector:
    def __init__(self):
        self.config = {
            "nkat_star_gemm": True,
            "cuda_optimization": True,
            "rtx_3080_optimization": True,
            "rope_scaling": "low",
            "mirostat_2": True,
            "default_settings": {
                "threads": 12,
                "parallel": 4,
                "context": 4096,
                "gpu_layers": 35,  # Q4_K_M 7B用
                "cuda_f16": True,
                "mirostat": 2,
                "mirostat_lr": 0.6
            }
        }
    
    def get_optimal_gpu_layers(self, model_size, quantization="Q4_K_M"):
        """モデルサイズと量子化に基づく最適GPU層数"""
        optimization_table = {
            "7B": {"Q4_K_M": 35, "Q6_K": 30, "Q8_0": 25, "Q4_0": 40},
            "13B": {"Q4_K_M": 28, "Q6_K": 25, "Q8_0": 20, "Q4_0": 32},
            "30B": {"Q4_K_M": 15, "Q6_K": 12, "Q8_0": 10, "Q4_0": 18},
            "70B": {"Q4_K_M": 8, "Q6_K": 6, "Q8_0": 5, "Q4_0": 10}
        }
        
        return optimization_table.get(model_size, {}).get(quantization, 30)
    
    def get_rtx_3080_optimization(self):
        """RTX 3080専用最適化設定"""
        return {
            "cuda_architectures": "86",
            "tensor_cores": True,
            "memory_optimization": True,
            "max_vram_usage": "9.5GB",  # 10GB中9.5GB使用
            "batch_size_optimization": True
        }
    
    def generate_kobold_command(self, model_path, custom_settings=None):
        """最適化されたKobold.cppコマンド生成"""
        settings = self.config["default_settings"].copy()
        if custom_settings:
            settings.update(custom_settings)
        
        # モデルサイズ自動検出（簡易版）
        model_name = Path(model_path).name.lower()
        if "7b" in model_name:
            settings["gpu_layers"] = self.get_optimal_gpu_layers("7B")
        elif "13b" in model_name:
            settings["gpu_layers"] = self.get_optimal_gpu_layers("13B")
        
        command_parts = [
            "python koboldcpp.py",
            f"--model \"{model_path}\"",
            f"--threads {settings['threads']}",
            f"--parallel {settings['parallel']}",
            f"--context {settings['context']}",
            f"--gpu-layers {settings['gpu_layers']}",
            "--rope-scaling low",
            "--cuda-f16",
            f"--mirostat {settings['mirostat']}",
            f"--mirostat-lr {settings['mirostat_lr']}"
        ]
        
        # NKAT拡張パラメータ（実装済みの場合）
        if self.config.get("nkat_star_gemm", False):
            command_parts.extend([
                "--nkat-theta-path theta_rank4.bin",
                "--nkat-decay 0.97"
            ])
        
        return " ".join(command_parts)
    
    def save_config(self, filepath="nkat_kobold_config.json"):
        """設定をJSONファイルに保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    selector = NKATBackendSelector()
    
    # 設定保存
    selector.save_config()
    
    # サンプルコマンド生成
    sample_command = selector.generate_kobold_command(
        "models/llama-7b-q4_k_m.gguf"
    )
    
    print("🔥 NKAT-Kobold.cpp最適化コマンド:")
    print(sample_command)
    print()
    print("📋 RTX 3080最適化設定:")
    rtx_settings = selector.get_rtx_3080_optimization()
    for key, value in rtx_settings.items():
        print(f"   {key}: {value}")
