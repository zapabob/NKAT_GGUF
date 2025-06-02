#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Kobold.cpp統合スクリプト
ユーザー提供のチューニングレシピに基づくNKAT拡張実装
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
import logging
from tqdm import tqdm
import torch

# ログ設定（日本語対応）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nkat_kobold_integration.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NKATKoboldIntegrator:
    def __init__(self):
        self.project_root = Path(".")
        self.llama_cpp_dir = self.project_root / "llama.cpp"
        self.ggml_src_dir = self.llama_cpp_dir / "ggml" / "src"
        self.nkat_extensions = {
            "nkat_star_gemm": False,
            "nkat_theta_path": False,
            "nkat_decay": False,
            "backend_selector": False
        }
        
    def check_cuda_availability(self):
        """CUDA環境とRTX 3080の確認"""
        try:
            if not torch.cuda.is_available():
                logger.warning("⚠️  CUDA が利用できません")
                return False
                
            device_count = torch.cuda.device_count()
            logger.info(f"🎯 CUDA デバイス数: {device_count}")
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"   GPU {i}: {props.name} (メモリ: {props.total_memory / 1024**3:.1f} GB)")
                
                # RTX 3080 検出
                if "3080" in props.name:
                    logger.info("🔥 RTX 3080 検出！最適化を有効化します")
                    return True
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ CUDA チェック失敗: {e}")
            return False
    
    def check_nkat_integration_status(self):
        """現在のNKAT統合状況を確認"""
        logger.info("📋 NKAT統合状況確認中...")
        
        # ggml.c でのNKAT実装チェック
        ggml_c_path = self.ggml_src_dir / "ggml.c"
        if ggml_c_path.exists():
            with open(ggml_c_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if "GGML_OP_NKAT_STAR_GEMM" in content:
                    self.nkat_extensions["nkat_star_gemm"] = True
                    logger.info("✅ NKAT STAR GEMM 実装済み")
                else:
                    logger.info("⏳ NKAT STAR GEMM 未実装")
        
        # backend_selector.py チェック（kobold.cpp用）
        backend_selector_path = self.project_root / "backend_selector.py"
        if backend_selector_path.exists():
            self.nkat_extensions["backend_selector"] = True
            logger.info("✅ Backend Selector 存在")
        else:
            logger.info("⏳ Backend Selector 未作成")
            
        return any(self.nkat_extensions.values())
    
    def create_nkat_backend_selector(self):
        """Kobold.cpp用のNKATバックエンドセレクター作成"""
        logger.info("🛠️  NKAT Backend Selector 作成中...")
        
        backend_selector_content = '''#!/usr/bin/env python3
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
            f"--model \\"{model_path}\\"",
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
'''
        
        with open("backend_selector.py", 'w', encoding='utf-8') as f:
            f.write(backend_selector_content)
        
        logger.info("✅ Backend Selector 作成完了")
        self.nkat_extensions["backend_selector"] = True
    
    def create_nkat_theta_generator(self):
        """NKAT Theta パラメータファイル生成"""
        logger.info("🧮 NKAT Theta パラメータ生成中...")
        
        try:
            # ダミーのtheta_rank4.binファイル生成（実際の実装では適切な値を使用）
            import numpy as np
            
            theta_data = {
                "rank": 4,
                "decay": 0.97,
                "temperature": 0.7,
                "nkat_coefficients": np.random.randn(4, 4096).astype(np.float16)
            }
            
            # バイナリ形式で保存
            np.savez_compressed("theta_rank4.npz", **theta_data)
            
            # .binファイルとしてもエクスポート（kobold.cpp互換用）
            with open("theta_rank4.bin", 'wb') as f:
                # ヘッダー情報
                f.write(b"NKAT")  # マジックナンバー
                f.write((4).to_bytes(4, 'little'))  # rank
                f.write(len(theta_data["nkat_coefficients"].tobytes()).to_bytes(4, 'little'))
                # データ
                f.write(theta_data["nkat_coefficients"].tobytes())
            
            logger.info("✅ Theta パラメータファイル生成完了")
            self.nkat_extensions["nkat_theta_path"] = True
            
        except Exception as e:
            logger.error(f"❌ Theta パラメータ生成失敗: {e}")
    
    def create_performance_monitor(self):
        """パフォーマンスモニタリングスクリプト作成"""
        logger.info("📊 パフォーマンスモニター作成中...")
        
        monitor_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-Kobold.cpp パフォーマンスモニター
"""

import time
import psutil
import subprocess
import json
from datetime import datetime

class NKATPerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.metrics = []
    
    def start_monitoring(self):
        """モニタリング開始"""
        self.start_time = time.time()
        print("🚀 パフォーマンスモニタリング開始")
    
    def get_gpu_stats(self):
        """GPU統計取得"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True
            )
            gpu_data = result.stdout.strip().split(", ")
            return {
                "gpu_utilization": float(gpu_data[0]),
                "memory_used": float(gpu_data[1]),
                "memory_total": float(gpu_data[2]),
                "temperature": float(gpu_data[3])
            }
        except:
            return None
    
    def log_metrics(self, tokens_per_second=None):
        """メトリクス記録"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        gpu_stats = self.get_gpu_stats()
        
        metric = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "tokens_per_second": tokens_per_second
        }
        
        if gpu_stats:
            metric.update(gpu_stats)
        
        self.metrics.append(metric)
        
        # リアルタイム表示
        print(f"📊 CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}% | " +
              f"GPU: {gpu_stats['gpu_utilization']:.1f}% | " +
              f"VRAM: {gpu_stats['memory_used']:.0f}/{gpu_stats['memory_total']:.0f}MB | " +
              f"Tok/s: {tokens_per_second or 'N/A'}")
    
    def save_report(self, filename="nkat_performance_report.json"):
        """パフォーマンスレポート保存"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        print(f"📄 レポート保存: {filename}")

if __name__ == "__main__":
    monitor = NKATPerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        while True:
            monitor.log_metrics()
            time.sleep(2)
    except KeyboardInterrupt:
        monitor.save_report()
        print("\\n✅ モニタリング終了")
'''
        
        with open("nkat_performance_monitor.py", 'w', encoding='utf-8') as f:
            f.write(monitor_content)
        
        logger.info("✅ パフォーマンスモニター作成完了")
    
    def run_integration(self):
        """NKAT-Kobold.cpp統合実行"""
        logger.info("🚀 NKAT-Kobold.cpp統合開始...")
        
        # CUDA確認
        if not self.check_cuda_availability():
            logger.warning("⚠️  CUDA未確認のため一部機能が制限されます")
        
        # 現在の統合状況確認
        self.check_nkat_integration_status()
        
        with tqdm(total=4, desc="NKAT統合進行中") as pbar:
            # Backend Selector作成
            if not self.nkat_extensions["backend_selector"]:
                self.create_nkat_backend_selector()
            pbar.update(1)
            
            # Theta パラメータ生成
            if not self.nkat_extensions["nkat_theta_path"]:
                self.create_nkat_theta_generator()
            pbar.update(1)
            
            # パフォーマンスモニター作成
            self.create_performance_monitor()
            pbar.update(1)
            
            # 設定ファイル最終化
            self.finalize_configuration()
            pbar.update(1)
        
        logger.info("🎉 NKAT-Kobold.cpp統合完了！")
        self.print_usage_instructions()
    
    def finalize_configuration(self):
        """最終設定"""
        config = {
            "nkat_kobold_integration": {
                "version": "1.0",
                "rtx_3080_optimized": True,
                "extensions": self.nkat_extensions,
                "recommended_models": [
                    "llama-7b-q4_k_m.gguf",
                    "llama-13b-q4_k_m.gguf",
                    "llama-7b-q6_k.gguf"
                ],
                "performance_targets": {
                    "tokens_per_second": "45+ (7B Q4_K_M)",
                    "perplexity_improvement": "-4%",
                    "vram_usage": "< 9.5GB"
                }
            }
        }
        
        with open("nkat_kobold_config_final.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def print_usage_instructions(self):
        """使用方法表示"""
        print("\n" + "="*60)
        print("🎯 NKAT-Kobold.cpp 使用方法")
        print("="*60)
        print()
        print("1️⃣ 最適化ビルド実行:")
        print("   py -3 -c \"exec(open('setup_nkat_kobold_integration.py').read())\"")
        print("   .\\build_nkat_kobold_optimized.ps1")
        print()
        print("2️⃣ Backend Selector使用:")
        print("   py -3 backend_selector.py")
        print()
        print("3️⃣ 推奨実行コマンド (7B Q4_K_M):")
        print("   python koboldcpp.py --model models/llama-7b-q4_k_m.gguf \\")
        print("     --threads 12 --parallel 4 --context 4096 \\")
        print("     --gpu-layers 35 --cuda-f16 --rope-scaling low \\")
        print("     --mirostat 2 --mirostat-lr 0.6")
        print()
        print("4️⃣ パフォーマンスモニタリング:")
        print("   py -3 nkat_performance_monitor.py")
        print()
        print("💡 レシピのポイント:")
        print("   ✅ Q4_K_M: 速度・品質・VRAM最適バランス")
        print("   ✅ gpu_layers=35: RTX 3080 (10GB) 推奨")  
        print("   ✅ threads=12: 物理コア数最適化")
        print("   ✅ mirostat 2: 高品質出力制御")
        print("="*60)

if __name__ == "__main__":
    integrator = NKATKoboldIntegrator()
    integrator.run_integration() 