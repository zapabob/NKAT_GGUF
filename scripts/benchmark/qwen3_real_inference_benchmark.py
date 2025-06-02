#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-8B-ERP 実測推論ベンチマーク
llama.cpp実行ファイルベースの正確な性能測定
"""

import os
import sys
import subprocess
import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import psutil
import threading
import queue

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qwen3_real_benchmark.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class GPUMonitor:
    """GPU使用率モニタリング"""
    
    def __init__(self):
        self.monitoring = False
        self.gpu_stats = []
        
    def start_monitoring(self):
        """GPU監視開始"""
        self.monitoring = True
        self.gpu_stats = []
        
        def monitor():
            while self.monitoring:
                try:
                    # nvidia-smi query
                    result = subprocess.run([
                        'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                        '--format=csv,noheader,nounits'
                    ], capture_output=True, text=True, timeout=2)
                    
                    if result.returncode == 0:
                        line = result.stdout.strip()
                        if line:
                            gpu_util, mem_used, mem_total, temp = line.split(', ')
                            self.gpu_stats.append({
                                'timestamp': time.time(),
                                'gpu_utilization': float(gpu_util),
                                'memory_used_mb': float(mem_used),
                                'memory_total_mb': float(mem_total),
                                'temperature_c': float(temp)
                            })
                except Exception as e:
                    logger.warning(f"GPU monitoring error: {e}")
                
                time.sleep(0.5)  # 500ms間隔
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict:
        """GPU監視停止と統計取得"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
        
        if not self.gpu_stats:
            return {}
        
        gpu_utils = [s['gpu_utilization'] for s in self.gpu_stats]
        mem_usage = [s['memory_used_mb'] for s in self.gpu_stats]
        temps = [s['temperature_c'] for s in self.gpu_stats]
        
        return {
            'avg_gpu_utilization': sum(gpu_utils) / len(gpu_utils),
            'max_gpu_utilization': max(gpu_utils),
            'avg_memory_used_mb': sum(mem_usage) / len(mem_usage),
            'max_memory_used_mb': max(mem_usage),
            'avg_temperature_c': sum(temps) / len(temps),
            'max_temperature_c': max(temps),
            'sample_count': len(self.gpu_stats)
        }

class Qwen3RealBenchmark:
    """Qwen3実測ベンチマーク"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.gpu_monitor = GPUMonitor()
        
        # llama.cpp実行ファイル検索
        self.main_exe = self.find_llama_executable()
        
        logger.info(f"🔥 Qwen3 Real Inference Benchmark")
        logger.info(f"   📁 Model: {Path(model_path).name}")
        logger.info(f"   ⚙️ Executable: {self.main_exe}")
    
    def find_llama_executable(self) -> Optional[str]:
        """llama.cpp実行ファイル検索"""
        search_paths = [
            "llama.cpp/build/Release/main.exe",
            "llama.cpp/build_cuda_en/Release/main.exe", 
            "llama.cpp/build_nkat_direct/Release/main.exe",
            "llama.cpp/build_simple/Release/main.exe"
        ]
        
        for path in search_paths:
            if os.path.exists(path):
                logger.info(f"✅ Found executable: {path}")
                return path
        
        logger.error("❌ No llama.cpp executable found")
        return None
    
    def run_inference_test(self, prompt: str, max_tokens: int = 100, 
                          ctx_size: int = 2048, threads: int = 12, 
                          gpu_layers: int = 40) -> Dict:
        """推論テスト実行"""
        
        if not self.main_exe:
            return {"error": "No executable found"}
        
        # コマンド構築
        cmd = [
            self.main_exe,
            "-m", self.model_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "-c", str(ctx_size),
            "-t", str(threads),
            "-ngl", str(gpu_layers),
            "--temp", "0.7",
            "--top-p", "0.9",
            "--repeat-penalty", "1.1",
            "--color",
            "--timing"
        ]
        
        logger.info(f"🚀 Starting inference test...")
        logger.info(f"   📝 Prompt length: {len(prompt)} chars")
        logger.info(f"   🎯 Max tokens: {max_tokens}")
        logger.info(f"   🖥️ Context: {ctx_size}, Threads: {threads}, GPU layers: {gpu_layers}")
        
        # GPU監視開始
        self.gpu_monitor.start_monitoring()
        
        try:
            start_time = time.time()
            
            # プロセス実行
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120,  # 2分タイムアウト
                encoding='utf-8',
                errors='replace'
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # GPU監視停止
            gpu_stats = self.gpu_monitor.stop_monitoring()
            
            # 結果解析
            if result.returncode == 0:
                return self.parse_llama_output(result.stdout, result.stderr, total_time, gpu_stats)
            else:
                logger.error(f"❌ Inference failed: {result.stderr}")
                return {"error": result.stderr, "total_time": total_time}
                
        except subprocess.TimeoutExpired:
            self.gpu_monitor.stop_monitoring()
            logger.error("❌ Inference timeout")
            return {"error": "timeout"}
        except Exception as e:
            self.gpu_monitor.stop_monitoring()
            logger.error(f"❌ Inference error: {e}")
            return {"error": str(e)}
    
    def parse_llama_output(self, stdout: str, stderr: str, total_time: float, gpu_stats: Dict) -> Dict:
        """llama.cpp出力解析"""
        
        # トークン数抽出
        prompt_tokens = 0
        generated_tokens = 0
        
        # llamaの統計情報抽出
        timing_pattern = r'eval time\s*=\s*([\d.]+)\s*ms.*?(\d+)\s*tokens.*?([\d.]+)\s*t/s'
        timing_match = re.search(timing_pattern, stderr)
        
        load_time_pattern = r'load time\s*=\s*([\d.]+)\s*ms'
        load_time_match = re.search(load_time_pattern, stderr)
        
        prompt_eval_pattern = r'prompt eval time\s*=\s*([\d.]+)\s*ms.*?(\d+)\s*tokens.*?([\d.]+)\s*t/s'
        prompt_eval_match = re.search(prompt_eval_pattern, stderr)
        
        results = {
            "success": True,
            "total_wall_time": total_time,
            "model_path": self.model_path,
            "gpu_stats": gpu_stats
        }
        
        # ロード時間
        if load_time_match:
            results["load_time_ms"] = float(load_time_match.group(1))
        
        # プロンプト評価
        if prompt_eval_match:
            results["prompt_eval_time_ms"] = float(prompt_eval_match.group(1))
            results["prompt_tokens"] = int(prompt_eval_match.group(2))
            results["prompt_tokens_per_sec"] = float(prompt_eval_match.group(3))
        
        # 生成評価
        if timing_match:
            results["eval_time_ms"] = float(timing_match.group(1))
            results["generated_tokens"] = int(timing_match.group(2))
            results["generation_tokens_per_sec"] = float(timing_match.group(3))
        
        # 生成テキスト抽出（簡易）
        if stdout:
            # プロンプト後の生成部分を取得
            lines = stdout.split('\n')
            generated_text = '\n'.join(lines[1:]) if len(lines) > 1 else ""
            results["generated_text"] = generated_text[:500]  # 最初の500文字
            results["generated_text_length"] = len(generated_text)
        
        logger.info(f"📊 Parse Results:")
        if "generation_tokens_per_sec" in results:
            logger.info(f"   🚀 Generation: {results['generation_tokens_per_sec']:.1f} tokens/sec")
        if "prompt_tokens_per_sec" in results:
            logger.info(f"   📝 Prompt eval: {results['prompt_tokens_per_sec']:.1f} tokens/sec")
        if "gpu_stats" in results and results["gpu_stats"]:
            logger.info(f"   🎮 GPU util: {results['gpu_stats'].get('avg_gpu_utilization', 0):.1f}%")
        
        return results
    
    def benchmark_suite(self) -> Dict:
        """ベンチマークスイート実行"""
        
        test_cases = [
            {
                "name": "Short Response",
                "prompt": "What is the capital of Japan?",
                "max_tokens": 50,
                "ctx_size": 1024
            },
            {
                "name": "Medium Generation", 
                "prompt": "Write a short story about a robot learning to paint.",
                "max_tokens": 200,
                "ctx_size": 2048
            },
            {
                "name": "Long Context",
                "prompt": "Explain the concept of quantum computing in detail, including its advantages and challenges.",
                "max_tokens": 500,
                "ctx_size": 4096
            },
            {
                "name": "Code Generation",
                "prompt": "def fibonacci(n):\n    # Generate fibonacci sequence up to n terms\n",
                "max_tokens": 150,
                "ctx_size": 2048
            }
        ]
        
        all_results = {
            "model_info": {
                "path": self.model_path,
                "size_gb": os.path.getsize(self.model_path) / (1024**3)
            },
            "test_results": []
        }
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"\n🧪 Test {i+1}/{len(test_cases)}: {test_case['name']}")
            
            result = self.run_inference_test(
                prompt=test_case["prompt"],
                max_tokens=test_case["max_tokens"],
                ctx_size=test_case["ctx_size"]
            )
            
            result.update({
                "test_name": test_case["name"],
                "test_prompt": test_case["prompt"]
            })
            
            all_results["test_results"].append(result)
            
            # 中間結果表示
            if "generation_tokens_per_sec" in result:
                logger.info(f"   ✅ {result['generation_tokens_per_sec']:.1f} tokens/sec")
            else:
                logger.info(f"   ❌ Failed: {result.get('error', 'unknown')}")
            
            # テスト間で小休止
            time.sleep(2)
        
        return all_results
    
    def save_results(self, results: Dict, filename: str = "qwen3_real_benchmark_results.json"):
        """結果保存"""
        try:
            output_dir = Path("output/qwen3_nkat_testing")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_path = output_dir / filename
            
            # タイムスタンプと環境情報追加
            results["benchmark_info"] = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "python_version": sys.version,
                "executable_path": self.main_exe,
                "system_cpu_count": psutil.cpu_count(),
                "system_memory_gb": psutil.virtual_memory().total / (1024**3)
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Results saved: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

def main():
    """メイン実行"""
    print("🔥 Qwen3-8B-ERP Real Inference Benchmark")
    print("=" * 60)
    
    model_path = "models/integrated/Qwen3-8B-ERP-v0.1.i1-Q6_K.gguf"
    
    # ベンチマーク実行
    benchmark = Qwen3RealBenchmark(model_path)
    
    if not benchmark.main_exe:
        print("❌ llama.cpp executable not found. Please build first.")
        return
    
    # ベンチマークスイート実行
    results = benchmark.benchmark_suite()
    
    # 結果保存
    benchmark.save_results(results)
    
    # サマリー表示
    print("\n📊 Benchmark Summary:")
    print("-" * 40)
    
    for result in results["test_results"]:
        test_name = result.get("test_name", "Unknown")
        if "generation_tokens_per_sec" in result:
            tokens_per_sec = result["generation_tokens_per_sec"]
            gpu_util = result.get("gpu_stats", {}).get("avg_gpu_utilization", 0)
            print(f"   {test_name:15s}: {tokens_per_sec:6.1f} t/s (GPU: {gpu_util:4.1f}%)")
        else:
            error = result.get("error", "unknown")
            print(f"   {test_name:15s}: FAILED ({error})")
    
    print("\n🎉 Real benchmark completed!")
    print("📁 Check output/qwen3_nkat_testing/ for detailed results")

if __name__ == "__main__":
    main() 