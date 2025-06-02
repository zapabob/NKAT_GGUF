#!/usr/bin/env python3
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
        print("\n✅ モニタリング終了")
