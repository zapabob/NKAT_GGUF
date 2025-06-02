#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Inference Engine
スター積GEMM演算による非可換推論エンジン実装
"""

import os
import sys
import json
import struct
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import time

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nkat_inference.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NKATStarGEMM:
    """NKAT スター積GEMM演算器"""
    
    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        logger.info(f"🔥 NKAT Star-GEMM initialized: device={self.device}")
    
    def star_multiply(self, A: torch.Tensor, x: torch.Tensor, 
                     theta: torch.Tensor, scale_theta: float,
                     gamma: float = 0.97) -> torch.Tensor:
        """
        スター積演算: (A ⋆ x) = Ax + 0.5 * γ * (θ ⋆ x)
        
        Args:
            A: 量子化重み行列
            x: 入力ベクトル
            theta: θテンソル (反対称)
            scale_theta: θの量子化スケール
            gamma: 減衰係数
        
        Returns:
            スター積結果
        """
        # 1. 標準的な行列積 (量子化)
        if A.dtype == torch.int8 and x.dtype == torch.float32:
            # 量子化行列とFP32ベクトルの積
            y_linear = self._quantized_matmul(A, x)
        else:
            y_linear = torch.matmul(A.float(), x)
        
        # 2. θ⋆x 位相項計算（サイズ調整付き）
        if theta is not None and scale_theta > 0:
            # θを復元
            theta_fp = theta.float() * scale_theta
            
            # サイズ調整: θのサイズと入力ベクトルのサイズを合わせる
            theta_size = theta_fp.shape[0]
            x_size = x.shape[0] if x.dim() == 1 else x.shape[-1]
            
            if theta_size != x_size:
                if x_size > theta_size:
                    # 入力を切り取り
                    x_adjusted = x[:theta_size] if x.dim() == 1 else x[..., :theta_size]
                    phase_term = torch.matmul(theta_fp, x_adjusted)
                    # 結果をゼロパディングで元のサイズに復元
                    if y_linear.shape[0] > theta_size:
                        phase_term_padded = torch.zeros_like(y_linear)
                        phase_term_padded[:theta_size] = phase_term
                        phase_term = phase_term_padded
                else:
                    # θを切り取り
                    theta_adjusted = theta_fp[:x_size, :x_size]
                    phase_term = torch.matmul(theta_adjusted, x)
            else:
                # サイズが一致している場合
                phase_term = torch.matmul(theta_fp, x)
            
            # スター積結合
            y_star = y_linear + 0.5 * gamma * phase_term
        else:
            y_star = y_linear
        
        return y_star
    
    def _quantized_matmul(self, A_q8: torch.Tensor, x_fp32: torch.Tensor) -> torch.Tensor:
        """最適化された量子化行列積"""
        if self.use_cuda:
            return self._cuda_quantized_matmul(A_q8, x_fp32)
        else:
            return self._cpu_quantized_matmul(A_q8, x_fp32)
    
    def _cpu_quantized_matmul(self, A_q8: torch.Tensor, x_fp32: torch.Tensor) -> torch.Tensor:
        """CPU版量子化行列積（符号XOR最適化）"""
        # 簡易実装（実際にはAVX2/NEON最適化が必要）
        return torch.matmul(A_q8.float(), x_fp32)
    
    def _cuda_quantized_matmul(self, A_q8: torch.Tensor, x_fp32: torch.Tensor) -> torch.Tensor:
        """CUDA版量子化行列積（TensorCore活用）"""
        # cuBLAS/cuDNNライブラリ使用想定
        A_fp16 = A_q8.to(torch.float16)
        x_fp16 = x_fp32.to(torch.float16)
        return torch.matmul(A_fp16, x_fp16).float()

class NKATModelLoader:
    """NKAT-GGUF モデルローダー"""
    
    def __init__(self):
        self.tensors = {}
        self.theta_tensors = {}
        self.metadata = {}
        
    def load_nkat_gguf(self, model_path: str) -> bool:
        """NKAT-GGUF ファイル読み込み"""
        try:
            logger.info(f"📂 NKAT-GGUF読み込み: {model_path}")
            
            if not os.path.exists(model_path):
                logger.error(f"❌ ファイルが見つかりません: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                # ヘッダー確認
                magic = f.read(4)
                if magic != b"NKAT":
                    logger.error(f"❌ 無効なNKAT-GGUFファイル")
                    return False
                
                # θテンソル数読み込み
                theta_count = struct.unpack('<I', f.read(4))[0]
                logger.info(f"   📊 θテンソル数: {theta_count}")
                
                # メタデータ読み込み
                metadata_size = struct.unpack('<I', f.read(4))[0]
                metadata_bytes = f.read(metadata_size)
                self.metadata = json.loads(metadata_bytes.decode('utf-8'))
                logger.info(f"   📄 メタデータ: NKAT v{self.metadata.get('nkat_version', 'unknown')}")
                
                # θテンソル読み込み
                for i in range(theta_count):
                    # テンソル名
                    name_size = struct.unpack('<I', f.read(4))[0]
                    tensor_name = f.read(name_size).decode('utf-8')
                    
                    # θテンソル形状
                    shape = struct.unpack('<II', f.read(8))
                    
                    # θテンソルデータ
                    data_size = shape[0] * shape[1]
                    theta_data = np.frombuffer(f.read(data_size), dtype=np.int8).reshape(shape)
                    
                    # スケール
                    scale_theta = struct.unpack('<f', f.read(4))[0]
                    
                    self.theta_tensors[tensor_name] = {
                        "data": torch.from_numpy(theta_data),
                        "scale": scale_theta,
                        "shape": shape
                    }
                    
                    logger.info(f"   ✅ {tensor_name}: {shape}, scale={scale_theta:.6f}")
            
            logger.info(f"🎉 NKAT-GGUF読み込み完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ NKAT-GGUF読み込み失敗: {e}")
            return False
    
    def get_theta_tensor(self, layer_name: str) -> Optional[Tuple[torch.Tensor, float]]:
        """指定レイヤーのθテンソル取得"""
        theta_name = layer_name.replace(".weight", ".theta.weight")
        if theta_name in self.theta_tensors:
            theta_info = self.theta_tensors[theta_name]
            return theta_info["data"], theta_info["scale"]
        return None, 0.0

class NKATInferenceEngine:
    """NKAT推論エンジン"""
    
    def __init__(self, model_path: str, use_cuda: bool = True):
        self.model_loader = NKATModelLoader()
        self.star_gemm = NKATStarGEMM(use_cuda)
        self.model_path = model_path
        self.layers = {}
        self.config = {}
        
        # デフォルト設定
        self.config = {
            "theta_gamma": 0.97,
            "theta_enabled": True,
            "layer_decay": True,
            "max_seq_len": 4096
        }
    
    def load_model(self) -> bool:
        """モデル読み込み"""
        success = self.model_loader.load_nkat_gguf(self.model_path)
        if success:
            # メタデータから設定更新
            metadata = self.model_loader.metadata
            self.config.update({
                "theta_gamma": metadata.get("theta_gamma", 0.97),
                "theta_rank": metadata.get("theta_rank", 4)
            })
            logger.info(f"⚙️  NKAT設定: γ={self.config['theta_gamma']}, rank={self.config['theta_rank']}")
        return success
    
    def forward_layer(self, x: torch.Tensor, layer_name: str, 
                     weight: torch.Tensor, layer_idx: int = 0) -> torch.Tensor:
        """単一レイヤー順伝播（NKAT拡張）"""
        # θテンソル取得
        theta, scale_theta = self.model_loader.get_theta_tensor(layer_name)
        
        # レイヤー深度による減衰
        gamma = self.config["theta_gamma"]
        if self.config["layer_decay"]:
            gamma = gamma ** layer_idx
        
        # スター積GEMM実行
        if theta is not None and self.config["theta_enabled"]:
            y = self.star_gemm.star_multiply(
                A=weight, 
                x=x, 
                theta=theta, 
                scale_theta=scale_theta,
                gamma=gamma
            )
            logger.debug(f"   🌟 スター積適用: {layer_name} (γ={gamma:.3f})")
        else:
            # フォールバック：標準行列積
            y = torch.matmul(weight.float(), x)
            logger.debug(f"   📐 標準行列積: {layer_name}")
        
        return y
    
    def benchmark_inference(self, sequence_length: int = 512, 
                          num_iterations: int = 100) -> Dict[str, float]:
        """推論性能ベンチマーク"""
        logger.info(f"🏁 NKAT推論ベンチマーク開始")
        logger.info(f"   📏 シーケンス長: {sequence_length}")
        logger.info(f"   🔄 反復回数: {num_iterations}")
        
        # ダミー入力生成
        batch_size = 1
        hidden_size = 4096
        x = torch.randn(batch_size, sequence_length, hidden_size)
        
        if self.star_gemm.use_cuda:
            x = x.cuda()
        
        # ダミー重み・θテンソル
        weight = torch.randn(hidden_size, hidden_size).to(torch.int8)
        theta = torch.randint(-127, 128, (512, 512), dtype=torch.int8)
        scale_theta = 0.01
        
        if self.star_gemm.use_cuda:
            weight = weight.cuda()
            theta = theta.cuda()
        
        # ウォームアップ
        for _ in range(10):
            _ = self.star_gemm.star_multiply(weight, x[0, 0], theta, scale_theta)
        
        if self.star_gemm.use_cuda:
            torch.cuda.synchronize()
        
        # ベンチマーク実行
        start_time = time.time()
        
        for i in tqdm(range(num_iterations), desc="推論ベンチマーク"):
            for seq_idx in range(sequence_length):
                y = self.star_gemm.star_multiply(
                    weight, x[0, seq_idx], theta, scale_theta, 
                    gamma=self.config["theta_gamma"]
                )
        
        if self.star_gemm.use_cuda:
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # 結果計算
        total_time = end_time - start_time
        total_operations = num_iterations * sequence_length
        ops_per_second = total_operations / total_time
        tokens_per_second = ops_per_second  # 簡易計算
        
        results = {
            "total_time": total_time,
            "tokens_per_second": tokens_per_second,
            "operations_per_second": ops_per_second,
            "avg_latency_ms": (total_time / total_operations) * 1000,
            "device": str(self.star_gemm.device),
            "theta_enabled": self.config["theta_enabled"]
        }
        
        # 結果表示
        logger.info(f"📊 ベンチマーク結果:")
        logger.info(f"   ⚡ tok/s: {tokens_per_second:.1f}")
        logger.info(f"   ⏱️  レイテンシ: {results['avg_latency_ms']:.2f} ms")
        logger.info(f"   🖥️  デバイス: {results['device']}")
        logger.info(f"   🌟 NKAT有効: {results['theta_enabled']}")
        
        return results
    
    def compare_with_baseline(self, sequence_length: int = 512) -> Dict[str, float]:
        """ベースライン（標準GEMM）との比較"""
        logger.info(f"🔍 ベースライン比較開始")
        
        # NKAT有効での測定
        self.config["theta_enabled"] = True
        nkat_results = self.benchmark_inference(sequence_length, 50)
        
        # NKAT無効での測定
        self.config["theta_enabled"] = False
        baseline_results = self.benchmark_inference(sequence_length, 50)
        
        # 比較結果
        speedup_ratio = baseline_results["tokens_per_second"] / nkat_results["tokens_per_second"]
        overhead_percentage = (speedup_ratio - 1.0) * 100
        
        comparison = {
            "nkat_tokens_per_second": nkat_results["tokens_per_second"],
            "baseline_tokens_per_second": baseline_results["tokens_per_second"],
            "speedup_ratio": speedup_ratio,
            "overhead_percentage": overhead_percentage,
            "nkat_latency_ms": nkat_results["avg_latency_ms"],
            "baseline_latency_ms": baseline_results["avg_latency_ms"]
        }
        
        logger.info(f"📈 比較結果:")
        logger.info(f"   🔥 NKAT: {comparison['nkat_tokens_per_second']:.1f} tok/s")
        logger.info(f"   📐 ベースライン: {comparison['baseline_tokens_per_second']:.1f} tok/s") 
        logger.info(f"   📊 オーバーヘッド: {overhead_percentage:+.1f}%")
        
        # 設定復元
        self.config["theta_enabled"] = True
        
        return comparison

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NKAT Inference Engine")
    parser.add_argument("--model", "-m", required=True, help="NKAT-GGUFモデルファイル")
    parser.add_argument("--benchmark", action="store_true", help="ベンチマークモード")
    parser.add_argument("--compare", action="store_true", help="ベースライン比較")
    parser.add_argument("--seq-len", type=int, default=512, help="シーケンス長")
    parser.add_argument("--iterations", type=int, default=100, help="反復回数")
    parser.add_argument("--no-cuda", action="store_true", help="CUDA無効")
    parser.add_argument("--theta-gamma", type=float, default=0.97, help="θ減衰率")
    
    args = parser.parse_args()
    
    # 推論エンジン初期化
    engine = NKATInferenceEngine(args.model, use_cuda=not args.no_cuda)
    engine.config["theta_gamma"] = args.theta_gamma
    
    # モデル読み込み
    if not engine.load_model():
        logger.error("❌ モデル読み込み失敗")
        sys.exit(1)
    
    if args.benchmark:
        results = engine.benchmark_inference(args.seq_len, args.iterations)
        
        # 結果をJSONで保存
        output_file = f"nkat_benchmark_{args.seq_len}_{args.iterations}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 結果保存: {output_file}")
    
    if args.compare:
        comparison = engine.compare_with_baseline(args.seq_len)
        
        # 比較結果をJSONで保存
        output_file = f"nkat_comparison_{args.seq_len}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        logger.info(f"📄 比較結果保存: {output_file}")

if __name__ == "__main__":
    main() 