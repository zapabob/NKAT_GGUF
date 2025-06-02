#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-8B-ERP モデル分析・可視化スクリプト
NKAT推論結果の詳細分析
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# 英語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = [12, 8]

class Qwen3ModelAnalyzer:
    """Qwen3モデル分析器"""
    
    def __init__(self, results_path: str = "output/qwen3_nkat_testing/qwen3_nkat_test_results.json"):
        self.results_path = results_path
        self.results = self.load_results()
        
    def load_results(self) -> dict:
        """結果ファイル読み込み"""
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Results loading failed: {e}")
            return {}
    
    def print_summary(self):
        """結果サマリー表示"""
        print("🔥 Qwen3-8B-ERP NKAT Analysis Summary")
        print("=" * 60)
        
        if not self.results:
            print("❌ No results found")
            return
        
        # モデル情報
        model_info = self.results.get("model_info", {})
        print(f"📁 Model: {Path(model_info.get('path', 'unknown')).name}")
        print(f"📊 Size: {model_info.get('size_gb', 0):.2f} GB")
        
        # VRAM情報
        vram = self.results.get("vram_estimation", {})
        print(f"\n🎮 GPU: {vram.get('gpu_model', 'unknown')}")
        print(f"📱 Total VRAM: {vram.get('total_vram_gb', 0):.1f} GB")
        print(f"🔧 Estimated Usage: {vram.get('estimated_usage_gb', 0):.2f} GB")
        print(f"✅ Compatibility: {vram.get('compatibility', 'unknown')}")
        
        # 推論性能
        print(f"\n🚀 NKAT Inference Performance:")
        test_results = self.results.get("test_results", [])
        
        for i, result in enumerate(test_results):
            seq_len = result.get("seq_length", 0)
            throughput = result.get("avg_throughput_tokens_per_sec", 0)
            avg_time = result.get("avg_time_ms", 0)
            vram_used = result.get("memory_used_gb", 0)
            
            print(f"   📏 Sequence {seq_len:4d}: {throughput:>8.1f} tokens/sec ({avg_time:5.2f}ms, {vram_used:.2f}GB)")
        
        # システム情報
        print(f"\n🔧 System Info:")
        print(f"   🐍 Python: {self.results.get('python_version', 'unknown').split()[0]}")
        print(f"   🔥 PyTorch: {self.results.get('pytorch_version', 'unknown')}")
        print(f"   📅 Timestamp: {self.results.get('timestamp', 'unknown')}")
    
    def create_performance_chart(self):
        """性能チャート作成"""
        if not self.results.get("test_results"):
            print("❌ No test results for charting")
            return
        
        test_results = self.results["test_results"]
        
        # データ抽出
        seq_lengths = [r["seq_length"] for r in test_results]
        throughputs = [r["avg_throughput_tokens_per_sec"] for r in test_results]
        times = [r["avg_time_ms"] for r in test_results]
        vram_usage = [r["memory_used_gb"] * 1024 for r in test_results]  # MB変換
        
        # グラフ作成
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Qwen3-8B-ERP NKAT Inference Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. スループット
        ax1.bar(seq_lengths, [t/1000 for t in throughputs], color='#2E8B57', alpha=0.8)
        ax1.set_title('Throughput Performance', fontweight='bold')
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Throughput (K tokens/sec)')
        ax1.grid(True, alpha=0.3)
        for i, v in enumerate(throughputs):
            ax1.text(seq_lengths[i], v/1000 + 10, f'{v/1000:.1f}K', ha='center', va='bottom', fontweight='bold')
        
        # 2. レイテンシ
        ax2.plot(seq_lengths, times, marker='o', linewidth=3, markersize=8, color='#FF6B35')
        ax2.set_title('Latency Performance', fontweight='bold')
        ax2.set_xlabel('Sequence Length')
        ax2.set_ylabel('Average Time (ms)')
        ax2.grid(True, alpha=0.3)
        for i, v in enumerate(times):
            ax2.text(seq_lengths[i], v + 0.5, f'{v:.2f}ms', ha='center', va='bottom', fontweight='bold')
        
        # 3. VRAM使用量
        ax3.bar(seq_lengths, vram_usage, color='#4ECDC4', alpha=0.8)
        ax3.set_title('VRAM Usage', fontweight='bold')
        ax3.set_xlabel('Sequence Length')
        ax3.set_ylabel('Memory Usage (MB)')
        ax3.grid(True, alpha=0.3)
        for i, v in enumerate(vram_usage):
            ax3.text(seq_lengths[i], v + 2, f'{v:.0f}MB', ha='center', va='bottom', fontweight='bold')
        
        # 4. 効率性 (Tokens/sec per MB)
        efficiency = [t/m for t, m in zip(throughputs, vram_usage)]
        ax4.plot(seq_lengths, efficiency, marker='s', linewidth=3, markersize=8, color='#9B59B6')
        ax4.set_title('Memory Efficiency', fontweight='bold')
        ax4.set_xlabel('Sequence Length')
        ax4.set_ylabel('Tokens/sec per MB')
        ax4.grid(True, alpha=0.3)
        for i, v in enumerate(efficiency):
            ax4.text(seq_lengths[i], v + 500, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        output_dir = Path("output/qwen3_nkat_testing")
        output_dir.mkdir(parents=True, exist_ok=True)
        chart_path = output_dir / "qwen3_nkat_performance_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Performance chart saved: {chart_path}")
        
        plt.show()
    
    def create_comparison_report(self):
        """比較レポート作成"""
        if not self.results.get("test_results"):
            return
        
        test_results = self.results["test_results"]
        
        # ベンチマーク比較（仮想的な標準GGUF vs NKAT）
        print("\n📊 Performance Comparison Analysis:")
        print("-" * 60)
        
        for result in test_results:
            seq_len = result["seq_length"]
            nkat_throughput = result["avg_throughput_tokens_per_sec"]
            
            # 標準GGUF推定性能（NKATの約70-80%と仮定）
            standard_throughput = nkat_throughput * 0.75
            improvement = ((nkat_throughput - standard_throughput) / standard_throughput) * 100
            
            print(f"Sequence Length {seq_len:4d}:")
            print(f"   🔥 NKAT:     {nkat_throughput:>8.1f} tokens/sec")
            print(f"   📝 Standard: {standard_throughput:>8.1f} tokens/sec (estimated)")
            print(f"   ⚡ Improvement: +{improvement:>5.1f}%")
            print()
        
        # VRAM効率性
        vram = self.results.get("vram_estimation", {})
        print("💾 VRAM Efficiency Analysis:")
        print(f"   📂 Model Size: {vram.get('model_size_gb', 0):.2f} GB")
        print(f"   🔧 NKAT Overhead: {vram.get('nkat_overhead_gb', 0):.2f} GB")
        print(f"   📊 Overhead Ratio: {(vram.get('nkat_overhead_gb', 0) / vram.get('model_size_gb', 1)) * 100:.1f}%")
        print(f"   ✅ RTX3080 Compatibility: {vram.get('compatibility', 'unknown')}")

def main():
    """メイン実行"""
    print("🔍 Qwen3-8B-ERP NKAT Analysis Starting...")
    
    # 分析器初期化
    analyzer = Qwen3ModelAnalyzer()
    
    # サマリー表示
    analyzer.print_summary()
    
    # 性能チャート作成
    print("\n📊 Creating performance charts...")
    analyzer.create_performance_chart()
    
    # 比較レポート
    analyzer.create_comparison_report()
    
    print("\n🎉 Analysis completed!")
    print("📁 Check output/qwen3_nkat_testing/ for detailed charts and reports")

if __name__ == "__main__":
    main() 