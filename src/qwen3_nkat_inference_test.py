#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-8B-ERP NKAT推論テストスクリプト
RTX3080 CUDA最適化対応
"""

import os
import sys
import torch
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
import logging

# 日本語表示設定
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'  # 英語フォント使用

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qwen3_nkat_test.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class Qwen3NKATInference:
    """Qwen3専用NKAT推論エンジン"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        
        logger.info(f"🚀 Qwen3-NKAT Inference Engine")
        logger.info(f"   📱 Device: {self.device}")
        logger.info(f"   🎮 GPU: {self.gpu_name}")
        logger.info(f"   📁 Model: {Path(model_path).name}")
        
        # CUDA最適化設定
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info(f"   ⚡ CUDA optimization enabled")
    
    def check_model_compatibility(self) -> bool:
        """モデル互換性チェック"""
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found: {self.model_path}")
                return False
            
            file_size = os.path.getsize(self.model_path) / (1024**3)
            logger.info(f"📊 Model size: {file_size:.2f} GB")
            
            # GGUF形式確認（簡易）
            with open(self.model_path, 'rb') as f:
                header = f.read(8)
                if b"GGUF" in header or b"GGML" in header:
                    logger.info(f"✅ GGUF format detected")
                    return True
                else:
                    logger.warning(f"⚠️ Unknown format, proceeding anyway")
                    return True
                    
        except Exception as e:
            logger.error(f"❌ Compatibility check failed: {e}")
            return False
    
    def estimate_vram_usage(self) -> dict:
        """VRAM使用量推定"""
        if not torch.cuda.is_available():
            return {"status": "CPU mode"}
        
        # RTX3080のスペック (10GB VRAM)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        free_vram = torch.cuda.memory_reserved(0) / (1024**3)
        
        model_size_gb = os.path.getsize(self.model_path) / (1024**3)
        
        # NKAT推論時の追加メモリ使用量推定
        nkat_overhead = model_size_gb * 0.15  # θテンソル + 中間計算
        total_estimated = model_size_gb + nkat_overhead + 1.0  # バッファ
        
        estimation = {
            "gpu_model": self.gpu_name,
            "total_vram_gb": total_vram,
            "model_size_gb": model_size_gb,
            "nkat_overhead_gb": nkat_overhead,
            "estimated_usage_gb": total_estimated,
            "compatibility": "OK" if total_estimated < total_vram * 0.9 else "TIGHT"
        }
        
        logger.info(f"🔧 VRAM Estimation:")
        logger.info(f"   📱 {estimation['gpu_model']}: {estimation['total_vram_gb']:.1f}GB")
        logger.info(f"   📂 Model: {estimation['model_size_gb']:.2f}GB")
        logger.info(f"   ⚙️ NKAT overhead: {estimation['nkat_overhead_gb']:.2f}GB")
        logger.info(f"   📊 Total estimated: {estimation['estimated_usage_gb']:.2f}GB")
        logger.info(f"   ✅ Status: {estimation['compatibility']}")
        
        return estimation
    
    def synthetic_inference_test(self, seq_length: int = 512, batch_size: int = 1) -> dict:
        """合成データによる推論性能テスト"""
        logger.info(f"🧪 Synthetic inference test starting...")
        logger.info(f"   📏 Sequence length: {seq_length}")
        logger.info(f"   📦 Batch size: {batch_size}")
        
        try:
            # 合成入力データ生成
            vocab_size = 32000  # Qwen3想定
            hidden_size = 4096  # 8B model想定
            
            # トークンシーケンス生成
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length), device=self.device)
            
            # 埋め込みベクトル生成（合成）
            embeddings = torch.randn(batch_size, seq_length, hidden_size, device=self.device, dtype=torch.float16)
            
            # NKAT演算シミュレーション
            times = []
            throughputs = []
            
            logger.info(f"   ⏱️ Running {10} iterations...")
            
            for i in tqdm(range(10), desc="NKAT inference"):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                # スター積演算シミュレーション
                # 1. 線形変換
                W = torch.randn(hidden_size, hidden_size, device=self.device, dtype=torch.float16)
                y_linear = torch.matmul(embeddings, W.T)
                
                # 2. NKAT θ項（反対称行列）
                theta = torch.randn(hidden_size, hidden_size, device=self.device, dtype=torch.float16)
                theta = theta - theta.T  # 反対称化
                y_phase = 0.5 * 0.97 * torch.matmul(embeddings, theta.T)
                
                # 3. スター積結合
                y_star = y_linear + y_phase
                
                # 4. アクティベーション
                output = torch.nn.functional.gelu(y_star)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                iteration_time = end_time - start_time
                tokens_per_sec = (batch_size * seq_length) / iteration_time
                
                times.append(iteration_time)
                throughputs.append(tokens_per_sec)
            
            # 統計計算
            avg_time = np.mean(times)
            avg_throughput = np.mean(throughputs)
            std_throughput = np.std(throughputs)
            
            # VRAM使用量チェック
            if torch.cuda.is_available():
                memory_used = torch.cuda.max_memory_allocated(0) / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
            else:
                memory_used = memory_reserved = 0
            
            results = {
                "model": "Qwen3-8B-ERP (synthetic)",
                "seq_length": seq_length,
                "batch_size": batch_size,
                "device": str(self.device),
                "avg_time_ms": avg_time * 1000,
                "avg_throughput_tokens_per_sec": avg_throughput,
                "throughput_std": std_throughput,
                "memory_used_gb": memory_used,
                "memory_reserved_gb": memory_reserved,
                "nkat_enabled": True
            }
            
            logger.info(f"📊 Test Results:")
            logger.info(f"   ⏱️ Average time: {results['avg_time_ms']:.2f}ms")
            logger.info(f"   🚀 Throughput: {results['avg_throughput_tokens_per_sec']:.1f} tokens/sec")
            logger.info(f"   📱 VRAM used: {results['memory_used_gb']:.2f}GB")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Synthetic test failed: {e}")
            return {"error": str(e)}
    
    def save_results(self, results: dict, filename: str = "qwen3_nkat_test_results.json"):
        """結果保存"""
        try:
            output_dir = Path("output/qwen3_nkat_testing")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / filename
            
            # タイムスタンプ追加
            results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
            results["python_version"] = sys.version
            results["pytorch_version"] = torch.__version__
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Results saved: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

def main():
    """メイン実行関数"""
    print("🔥 Qwen3-8B-ERP NKAT Inference Test")
    print("=" * 50)
    
    model_path = "models/integrated/Qwen3-8B-ERP-v0.1.i1-Q6_K.gguf"
    
    # 推論エンジン初期化
    engine = Qwen3NKATInference(model_path)
    
    # モデル互換性チェック
    if not engine.check_model_compatibility():
        print("❌ Model compatibility check failed")
        return
    
    # VRAM推定
    vram_estimation = engine.estimate_vram_usage()
    
    # 合成データテスト
    test_configs = [
        {"seq_length": 256, "batch_size": 1},
        {"seq_length": 512, "batch_size": 1},
        {"seq_length": 1024, "batch_size": 1},
    ]
    
    all_results = {
        "model_info": {
            "path": model_path,
            "size_gb": os.path.getsize(model_path) / (1024**3)
        },
        "vram_estimation": vram_estimation,
        "test_results": []
    }
    
    for config in test_configs:
        print(f"\n🧪 Testing: {config}")
        result = engine.synthetic_inference_test(**config)
        all_results["test_results"].append(result)
        
        # 中間結果表示
        if "error" not in result:
            print(f"   ✅ {result['avg_throughput_tokens_per_sec']:.1f} tokens/sec")
        else:
            print(f"   ❌ Error: {result['error']}")
    
    # 結果保存
    engine.save_results(all_results)
    
    print("\n🎉 Qwen3-NKAT inference test completed!")
    print(f"📁 Check output/qwen3_nkat_testing/ for detailed results")

if __name__ == "__main__":
    main() 