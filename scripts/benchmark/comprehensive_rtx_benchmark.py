#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-GGUF Comprehensive Benchmark for RTX30/RTX40 Series
RTX30/40シリーズ向け包括的ベンチマーク
"""

import os
import sys
import json
import time
import torch
import subprocess
import psutil
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_rtx_benchmark.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RTXBenchmarkSuite:
    """RTX30/40シリーズ向け包括的ベンチマーク"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gpu_info = self.detect_rtx_gpu()
        self.config = self.load_rtx_config()
        self.results = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "gpu_info": self.gpu_info,
                "system_info": self.get_system_info()
            },
            "test_results": []
        }
        
        logger.info(f"🔥 RTX Comprehensive Benchmark Suite")
        logger.info(f"   🎮 GPU: {self.gpu_info.get('name', 'Unknown')}")
        logger.info(f"   💾 VRAM: {self.gpu_info.get('memory_gb', 0):.1f} GB")
    
    def detect_rtx_gpu(self) -> Dict:
        """RTX GPU検出"""
        try:
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=name,memory.total', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                line = result.stdout.strip()
                if line:
                    name, memory_mb = line.split(', ')
                    memory_gb = float(memory_mb) / 1024
                    
                    # RTXシリーズ判定
                    rtx_series = None
                    gpu_model = None
                    
                    if "RTX 30" in name or any(model in name for model in ["RTX 3060", "RTX 3070", "RTX 3080", "RTX 3090"]):
                        rtx_series = "RTX30"
                        if "3060" in name:
                            gpu_model = "RTX3060"
                        elif "3070" in name:
                            gpu_model = "RTX3070"
                        elif "3080" in name:
                            gpu_model = "RTX3080"
                        elif "3090" in name:
                            gpu_model = "RTX3090"
                    elif "RTX 40" in name or any(model in name for model in ["RTX 4060", "RTX 4070", "RTX 4080", "RTX 4090"]):
                        rtx_series = "RTX40"
                        if "4060" in name:
                            gpu_model = "RTX4060"
                        elif "4070" in name:
                            gpu_model = "RTX4070"
                        elif "4080" in name:
                            gpu_model = "RTX4080"
                        elif "4090" in name:
                            gpu_model = "RTX4090"
                    
                    return {
                        "name": name,
                        "memory_gb": memory_gb,
                        "rtx_series": rtx_series,
                        "gpu_model": gpu_model,
                        "supported": rtx_series is not None
                    }
        except Exception as e:
            logger.warning(f"Failed to detect GPU: {e}")
        
        return {"name": "Unknown", "memory_gb": 0, "supported": False}
    
    def load_rtx_config(self) -> Dict:
        """RTX設定読み込み"""
        if not self.gpu_info.get("supported"):
            logger.warning("Unsupported GPU, using default config")
            return self.get_default_config()
        
        series = self.gpu_info["rtx_series"].lower()
        config_path = Path(f"configs/{series}/default_config.json")
        
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.info(f"✅ Loaded {series.upper()} configuration")
                return config
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")
        
        return self.get_default_config()
    
    def get_default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            "rtx_series": "default",
            "cuda_architectures": ["86"],
            "recommended_settings": {
                "default": {
                    "vram_gb": 8,
                    "max_context": 4096,
                    "gpu_layers": 35,
                    "batch_size": 512,
                    "threads": 8
                }
            },
            "nkat_parameters": {
                "rank": 6,
                "gamma": 0.95,
                "optimization_target": "balanced"
            }
        }
    
    def get_system_info(self) -> Dict:
        """システム情報取得"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "platform": sys.platform,
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
        }
    
    def run_gpu_memory_benchmark(self) -> Dict:
        """GPU メモリベンチマーク"""
        logger.info("🧠 Running GPU memory benchmark...")
        
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        results = {
            "test_name": "GPU Memory Benchmark",
            "allocated_memory_tests": [],
            "peak_memory_gb": 0,
            "memory_efficiency": 0
        }
        
        try:
            # メモリ使用量テスト
            memory_sizes = [1, 2, 4, 6, 8]  # GB
            
            for size_gb in tqdm(memory_sizes, desc="Memory tests"):
                if size_gb > self.gpu_info["memory_gb"] * 0.9:  # 90%制限
                    continue
                
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
                try:
                    # テンソル作成（float16で効率化）
                    elements = int(size_gb * 1024**3 / 2)  # float16 = 2 bytes
                    tensor = torch.randn(elements, dtype=torch.float16, device=self.device)
                    
                    # 簡単な演算実行
                    start_time = time.time()
                    result = torch.sum(tensor)
                    torch.cuda.synchronize()
                    end_time = time.time()
                    
                    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
                    results["peak_memory_gb"] = max(results["peak_memory_gb"], peak_memory)
                    
                    results["allocated_memory_tests"].append({
                        "target_gb": size_gb,
                        "actual_peak_gb": peak_memory,
                        "compute_time_ms": (end_time - start_time) * 1000,
                        "success": True
                    })
                    
                    # メモリクリア
                    del tensor, result
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    results["allocated_memory_tests"].append({
                        "target_gb": size_gb,
                        "error": str(e),
                        "success": False
                    })
            
            # メモリ効率計算
            if results["allocated_memory_tests"]:
                successful_tests = [t for t in results["allocated_memory_tests"] if t["success"]]
                if successful_tests:
                    max_successful = max(t["target_gb"] for t in successful_tests)
                    results["memory_efficiency"] = max_successful / self.gpu_info["memory_gb"]
            
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def run_nkat_performance_test(self) -> Dict:
        """NKAT性能テスト"""
        logger.info("⚡ Running NKAT performance test...")
        
        results = {
            "test_name": "NKAT Performance Test",
            "configurations": [],
            "best_config": None,
            "throughput_range": {}
        }
        
        # NKATパラメータ設定
        nkat_configs = [
            {"rank": 4, "gamma": 0.95},
            {"rank": 6, "gamma": 0.97},
            {"rank": 8, "gamma": 0.98},
        ]
        
        best_throughput = 0
        
        for config in tqdm(nkat_configs, desc="NKAT configs"):
            try:
                # NKATレイヤーシミュレーション
                result = self.simulate_nkat_inference(config["rank"], config["gamma"])
                
                results["configurations"].append({
                    "rank": config["rank"],
                    "gamma": config["gamma"],
                    "throughput_tokens_per_sec": result["throughput"],
                    "memory_usage_gb": result["memory_gb"],
                    "latency_ms": result["latency_ms"],
                    "success": True
                })
                
                if result["throughput"] > best_throughput:
                    best_throughput = result["throughput"]
                    results["best_config"] = config
                
            except Exception as e:
                results["configurations"].append({
                    "rank": config["rank"],
                    "gamma": config["gamma"],
                    "error": str(e),
                    "success": False
                })
        
        # スループット範囲
        successful_configs = [c for c in results["configurations"] if c["success"]]
        if successful_configs:
            throughputs = [c["throughput_tokens_per_sec"] for c in successful_configs]
            results["throughput_range"] = {
                "min": min(throughputs),
                "max": max(throughputs),
                "avg": sum(throughputs) / len(throughputs)
            }
        
        return results
    
    def simulate_nkat_inference(self, rank: int, gamma: float) -> Dict:
        """NKAT推論シミュレーション"""
        # 実際のNKAT実装に基づくシミュレーション
        hidden_size = 4096
        seq_length = 1024
        batch_size = 1
        
        # GPU設定に基づく調整
        gpu_model = self.gpu_info.get("gpu_model", "default")
        if gpu_model in self.config["recommended_settings"]:
            settings = self.config["recommended_settings"][gpu_model]
            seq_length = min(seq_length, settings["max_context"] // 4)
        
        try:
            torch.cuda.empty_cache()
            start_time = time.time()
            
            # 合成テンソル作成
            x = torch.randn(batch_size, seq_length, hidden_size, 
                          device=self.device, dtype=torch.float16)
            
            # NKAT演算シミュレーション
            iterations = 10
            for _ in range(iterations):
                # 線形変換
                linear_out = torch.matmul(x, torch.randn(hidden_size, hidden_size, device=self.device, dtype=torch.float16))
                
                # θテンソル（低ランク近似）
                u = torch.randn(hidden_size, rank, device=self.device, dtype=torch.float16)
                v = torch.randn(rank, hidden_size, device=self.device, dtype=torch.float16)
                theta = torch.matmul(u, v)
                
                # 反対称化
                theta_antisymm = 0.5 * (theta - theta.T)
                
                # NKAT演算
                nkat_out = gamma * torch.matmul(x, theta_antisymm.T)
                
                # 最終出力
                output = linear_out + nkat_out
                x = output  # 次の反復用
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            # メトリクス計算
            total_time = end_time - start_time
            tokens_processed = batch_size * seq_length * iterations
            throughput = tokens_processed / total_time
            
            memory_used = torch.cuda.max_memory_allocated() / (1024**3)
            latency_ms = (total_time / iterations) * 1000
            
            return {
                "throughput": throughput,
                "memory_gb": memory_used,
                "latency_ms": latency_ms
            }
            
        except Exception as e:
            logger.error(f"NKAT simulation failed: {e}")
            return {"throughput": 0, "memory_gb": 0, "latency_ms": float('inf')}
    
    def run_context_length_scaling_test(self) -> Dict:
        """コンテキスト長スケーリングテスト"""
        logger.info("📏 Running context length scaling test...")
        
        results = {
            "test_name": "Context Length Scaling Test",
            "scaling_results": [],
            "recommended_context": 0
        }
        
        # GPU別推奨最大コンテキスト長
        gpu_model = self.gpu_info.get("gpu_model", "default")
        if gpu_model in self.config["recommended_settings"]:
            max_context = self.config["recommended_settings"][gpu_model]["max_context"]
        else:
            max_context = 4096
        
        context_lengths = [512, 1024, 2048, 4096, 8192, 16384]
        context_lengths = [c for c in context_lengths if c <= max_context]
        
        for ctx_len in tqdm(context_lengths, desc="Context lengths"):
            try:
                start_time = time.time()
                
                # コンテキスト長テスト
                batch_size = 1
                hidden_size = 4096
                
                torch.cuda.empty_cache()
                
                # 入力テンソル作成
                input_tensor = torch.randn(batch_size, ctx_len, hidden_size, 
                                         device=self.device, dtype=torch.float16)
                
                # 簡単なTransformer風の演算
                attention_weights = torch.randn(ctx_len, ctx_len, device=self.device, dtype=torch.float16)
                output = torch.matmul(attention_weights, input_tensor)
                
                torch.cuda.synchronize()
                end_time = time.time()
                
                process_time = end_time - start_time
                memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                tokens_per_sec = ctx_len / process_time
                
                results["scaling_results"].append({
                    "context_length": ctx_len,
                    "process_time_ms": process_time * 1000,
                    "memory_usage_gb": memory_used,
                    "tokens_per_sec": tokens_per_sec,
                    "success": True
                })
                
                # 推奨コンテキスト長更新
                if memory_used < self.gpu_info["memory_gb"] * 0.8:  # 80%以下
                    results["recommended_context"] = ctx_len
                
            except Exception as e:
                results["scaling_results"].append({
                    "context_length": ctx_len,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def run_thermal_stability_test(self) -> Dict:
        """熱安定性テスト"""
        logger.info("🌡️ Running thermal stability test...")
        
        results = {
            "test_name": "Thermal Stability Test",
            "temperature_readings": [],
            "performance_degradation": 0,
            "thermal_throttling_detected": False
        }
        
        try:
            # 長時間負荷テスト（5分間）
            test_duration = 300  # 5分
            start_time = time.time()
            
            initial_throughput = None
            
            while time.time() - start_time < test_duration:
                # GPU温度取得
                temp_result = subprocess.run([
                    'nvidia-smi', '--query-gpu=temperature.gpu',
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)
                
                if temp_result.returncode == 0:
                    temperature = int(temp_result.stdout.strip())
                else:
                    temperature = 0
                
                # 性能テスト
                current_throughput = self.quick_performance_test()
                
                if initial_throughput is None:
                    initial_throughput = current_throughput
                
                results["temperature_readings"].append({
                    "timestamp": time.time() - start_time,
                    "temperature_c": temperature,
                    "throughput": current_throughput
                })
                
                # サーマルスロットリング検出（温度83度以上で性能10%以上低下）
                if temperature > 83 and current_throughput < initial_throughput * 0.9:
                    results["thermal_throttling_detected"] = True
                
                time.sleep(30)  # 30秒間隔
            
            # 性能劣化計算
            if len(results["temperature_readings"]) > 1:
                final_throughput = results["temperature_readings"][-1]["throughput"]
                results["performance_degradation"] = (initial_throughput - final_throughput) / initial_throughput
        
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def quick_performance_test(self) -> float:
        """クイック性能テスト"""
        try:
            # 簡単な性能測定
            size = 2048
            x = torch.randn(size, size, device=self.device, dtype=torch.float16)
            
            start_time = time.time()
            for _ in range(10):
                y = torch.matmul(x, x)
            torch.cuda.synchronize()
            end_time = time.time()
            
            return 1.0 / (end_time - start_time)  # 操作/秒
            
        except Exception:
            return 0.0
    
    def create_benchmark_visualization(self) -> None:
        """ベンチマーク結果可視化"""
        logger.info("📊 Creating benchmark visualizations...")
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'RTX Benchmark Results - {self.gpu_info.get("name", "Unknown")}', 
                        fontsize=16, fontweight='bold')
            
            # 1. メモリ使用量
            memory_test = next((r for r in self.results["test_results"] if r["test_name"] == "GPU Memory Benchmark"), None)
            if memory_test and "allocated_memory_tests" in memory_test:
                successful_tests = [t for t in memory_test["allocated_memory_tests"] if t["success"]]
                if successful_tests:
                    sizes = [t["target_gb"] for t in successful_tests]
                    times = [t["compute_time_ms"] for t in successful_tests]
                    
                    ax1.plot(sizes, times, 'b-o', linewidth=2, markersize=6)
                    ax1.set_xlabel('Memory Size (GB)')
                    ax1.set_ylabel('Compute Time (ms)')
                    ax1.set_title('Memory Usage vs Compute Time')
                    ax1.grid(True, alpha=0.3)
            
            # 2. NKAT性能比較
            nkat_test = next((r for r in self.results["test_results"] if r["test_name"] == "NKAT Performance Test"), None)
            if nkat_test and "configurations" in nkat_test:
                successful_configs = [c for c in nkat_test["configurations"] if c["success"]]
                if successful_configs:
                    ranks = [c["rank"] for c in successful_configs]
                    throughputs = [c["throughput_tokens_per_sec"] for c in successful_configs]
                    
                    bars = ax2.bar(range(len(ranks)), throughputs, color='skyblue', alpha=0.7)
                    ax2.set_xlabel('NKAT Rank')
                    ax2.set_ylabel('Throughput (tokens/sec)')
                    ax2.set_title('NKAT Performance by Rank')
                    ax2.set_xticks(range(len(ranks)))
                    ax2.set_xticklabels([f"Rank {r}" for r in ranks])
                    ax2.grid(True, alpha=0.3)
                    
                    # 値をバーに表示
                    for bar, throughput in zip(bars, throughputs):
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughputs)*0.01,
                                f'{throughput:.0f}', ha='center', va='bottom')
            
            # 3. コンテキスト長スケーリング
            context_test = next((r for r in self.results["test_results"] if r["test_name"] == "Context Length Scaling Test"), None)
            if context_test and "scaling_results" in context_test:
                successful_scaling = [s for s in context_test["scaling_results"] if s["success"]]
                if successful_scaling:
                    contexts = [s["context_length"] for s in successful_scaling]
                    memories = [s["memory_usage_gb"] for s in successful_scaling]
                    
                    ax3.plot(contexts, memories, 'g-s', linewidth=2, markersize=6)
                    ax3.set_xlabel('Context Length (tokens)')
                    ax3.set_ylabel('Memory Usage (GB)')
                    ax3.set_title('Context Length vs Memory Usage')
                    ax3.grid(True, alpha=0.3)
                    
                    # VRAM制限線
                    ax3.axhline(y=self.gpu_info["memory_gb"], color='red', linestyle='--', 
                               label=f'VRAM Limit ({self.gpu_info["memory_gb"]:.1f} GB)')
                    ax3.legend()
            
            # 4. 熱安定性
            thermal_test = next((r for r in self.results["test_results"] if r["test_name"] == "Thermal Stability Test"), None)
            if thermal_test and "temperature_readings" in thermal_test:
                readings = thermal_test["temperature_readings"]
                if readings:
                    times = [r["timestamp"] / 60 for r in readings]  # 分単位
                    temps = [r["temperature_c"] for r in readings]
                    
                    ax4.plot(times, temps, 'r-', linewidth=2)
                    ax4.set_xlabel('Time (minutes)')
                    ax4.set_ylabel('Temperature (°C)')
                    ax4.set_title('GPU Temperature Over Time')
                    ax4.grid(True, alpha=0.3)
                    
                    # 危険温度線
                    ax4.axhline(y=83, color='orange', linestyle='--', label='Throttle Warning (83°C)')
                    ax4.axhline(y=90, color='red', linestyle='--', label='Critical (90°C)')
                    ax4.legend()
            
            plt.tight_layout()
            
            # 保存
            output_dir = Path("output/benchmarks")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            chart_path = output_dir / f"rtx_benchmark_{self.gpu_info.get('gpu_model', 'unknown')}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            
            logger.info(f"📊 Benchmark charts saved: {chart_path}")
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
    
    def run_comprehensive_benchmark(self) -> None:
        """包括的ベンチマーク実行"""
        logger.info("🚀 Starting comprehensive RTX benchmark suite")
        
        # 1. GPU メモリベンチマーク
        memory_result = self.run_gpu_memory_benchmark()
        self.results["test_results"].append(memory_result)
        
        # 2. NKAT性能テスト
        nkat_result = self.run_nkat_performance_test()
        self.results["test_results"].append(nkat_result)
        
        # 3. コンテキスト長スケーリング
        context_result = self.run_context_length_scaling_test()
        self.results["test_results"].append(context_result)
        
        # 4. 熱安定性テスト（短縮版）
        thermal_result = self.run_thermal_stability_test()
        self.results["test_results"].append(thermal_result)
        
        # 5. 結果保存
        self.save_results()
        
        # 6. 可視化
        self.create_benchmark_visualization()
        
        # 7. サマリー表示
        self.print_benchmark_summary()
    
    def save_results(self) -> None:
        """結果保存"""
        try:
            output_dir = Path("output/benchmarks")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gpu_model = self.gpu_info.get("gpu_model", "unknown")
            filename = f"rtx_benchmark_{gpu_model}_{timestamp}.json"
            
            output_path = output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Benchmark results saved: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
    
    def print_benchmark_summary(self) -> None:
        """ベンチマークサマリー表示"""
        print("\n" + "="*60)
        print("🔥 RTX Comprehensive Benchmark Summary")
        print("="*60)
        
        print(f"🎮 GPU: {self.gpu_info.get('name', 'Unknown')}")
        print(f"💾 VRAM: {self.gpu_info.get('memory_gb', 0):.1f} GB")
        print(f"📊 Tests Completed: {len(self.results['test_results'])}")
        
        # 各テスト結果サマリー
        for result in self.results["test_results"]:
            test_name = result["test_name"]
            print(f"\n📋 {test_name}:")
            
            if test_name == "GPU Memory Benchmark":
                if "memory_efficiency" in result:
                    print(f"   Memory Efficiency: {result['memory_efficiency']:.1%}")
                    print(f"   Peak Memory: {result['peak_memory_gb']:.1f} GB")
            
            elif test_name == "NKAT Performance Test":
                if "best_config" in result and result["best_config"]:
                    print(f"   Best Config: Rank {result['best_config']['rank']}, Gamma {result['best_config']['gamma']}")
                if "throughput_range" in result:
                    tr = result["throughput_range"]
                    print(f"   Throughput Range: {tr.get('min', 0):.0f} - {tr.get('max', 0):.0f} tokens/sec")
            
            elif test_name == "Context Length Scaling Test":
                if "recommended_context" in result:
                    print(f"   Recommended Context: {result['recommended_context']} tokens")
            
            elif test_name == "Thermal Stability Test":
                if "thermal_throttling_detected" in result:
                    throttling = "Yes" if result["thermal_throttling_detected"] else "No"
                    print(f"   Thermal Throttling: {throttling}")
        
        print("\n📁 Results saved in: output/benchmarks/")
        print("🎉 Comprehensive benchmark completed!")

def main():
    """メイン実行"""
    print("🔥 RTX30/RTX40 Series Comprehensive Benchmark")
    print("=" * 60)
    
    benchmark = RTXBenchmarkSuite()
    
    if not benchmark.gpu_info.get("supported"):
        print("⚠️ Warning: GPU not explicitly supported")
        print("   This benchmark is optimized for RTX 30/40 series")
        
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Benchmark cancelled.")
            return
    
    benchmark.run_comprehensive_benchmark()

if __name__ == "__main__":
    main() 