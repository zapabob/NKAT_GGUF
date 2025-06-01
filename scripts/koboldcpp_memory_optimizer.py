#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 KoboldCPP メモリ最適化・エラー解決システム
KoboldCPP Memory Optimization & Error Resolution System

対応エラー:
- bad_alloc error while reading value for key 'tokenizer.ggml.tokens'
- access violation reading 0x0000000000000008
- メモリ不足エラー
"""

import os
import sys
import json
import psutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class KoboldCPPConfig:
    """KoboldCPP最適化設定"""
    # メモリ設定
    max_memory_usage_percent: float = 80.0
    enable_memory_mapping: bool = False  # nommap=True推奨
    enable_memory_lock: bool = False     # usemlock=False推奨
    
    # BLAS設定
    blas_batch_size: int = 128           # 512から128に削減
    blas_threads: int = 4                # スレッド数削減
    
    # GPU設定
    gpu_layers: int = 0                  # GPU層数制限
    context_size: int = 2048             # コンテキストサイズ削減
    
    # 安全性設定
    enable_noavx2: bool = True           # AVX2無効化
    enable_failsafe: bool = True         # フェイルセーフ有効
    
    # ポート設定
    port: int = 5001

class MemoryAnalyzer:
    """メモリ分析器"""
    
    def __init__(self):
        self.total_memory = psutil.virtual_memory().total / 1024**3
        self.available_memory = psutil.virtual_memory().available / 1024**3
        
    def analyze_system_memory(self) -> Dict[str, Any]:
        """システムメモリ分析"""
        memory_info = psutil.virtual_memory()
        
        analysis = {
            'total_gb': self.total_memory,
            'available_gb': self.available_memory,
            'used_gb': memory_info.used / 1024**3,
            'usage_percent': memory_info.percent,
            'safe_for_large_models': self.available_memory > 12.0,
            'recommended_context_size': self._recommend_context_size(),
            'recommended_gpu_layers': self._recommend_gpu_layers()
        }
        
        return analysis
    
    def _recommend_context_size(self) -> int:
        """推奨コンテキストサイズ"""
        if self.available_memory >= 16.0:
            return 4096
        elif self.available_memory >= 8.0:
            return 2048
        elif self.available_memory >= 4.0:
            return 1024
        else:
            return 512
    
    def _recommend_gpu_layers(self) -> int:
        """推奨GPU層数"""
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                if vram_gb >= 8.0:
                    return min(33, int(vram_gb * 3))  # VRAMの3倍程度
                else:
                    return 0
            else:
                return 0
        except ImportError:
            return 0

class KoboldCPPOptimizer:
    """KoboldCPP最適化器"""
    
    def __init__(self, config: KoboldCPPConfig):
        self.config = config
        self.memory_analyzer = MemoryAnalyzer()
        
    def optimize_for_model(self, model_path: str) -> KoboldCPPConfig:
        """モデル用最適化設定生成"""
        print(f"🔧 KoboldCPP最適化設定生成: {Path(model_path).name}")
        
        # ファイルサイズ分析
        model_size_gb = os.path.getsize(model_path) / 1024**3
        
        # メモリ分析
        memory_analysis = self.memory_analyzer.analyze_system_memory()
        
        # 最適化設定生成
        optimized_config = self._generate_optimized_config(model_size_gb, memory_analysis)
        
        print(f"  📊 モデルサイズ: {model_size_gb:.1f}GB")
        print(f"  💾 利用可能メモリ: {memory_analysis['available_gb']:.1f}GB")
        print(f"  ⚙️ 推奨設定:")
        print(f"    - コンテキストサイズ: {optimized_config.context_size}")
        print(f"    - GPU層数: {optimized_config.gpu_layers}")
        print(f"    - BLAS バッチサイズ: {optimized_config.blas_batch_size}")
        
        return optimized_config
    
    def _generate_optimized_config(self, model_size_gb: float, memory_analysis: Dict[str, Any]) -> KoboldCPPConfig:
        """最適化設定生成"""
        config = KoboldCPPConfig()
        
        # メモリベースの調整
        available_gb = memory_analysis['available_gb']
        
        # コンテキストサイズ調整
        if model_size_gb > 8.0:  # 大型モデル
            config.context_size = min(2048, memory_analysis['recommended_context_size'])
            config.blas_batch_size = 64
            config.gpu_layers = 0  # CPU推奨
        elif model_size_gb > 4.0:  # 中型モデル
            config.context_size = min(4096, memory_analysis['recommended_context_size'])
            config.blas_batch_size = 128
            config.gpu_layers = min(16, memory_analysis['recommended_gpu_layers'])
        else:  # 小型モデル
            config.context_size = 4096
            config.blas_batch_size = 256
            config.gpu_layers = memory_analysis['recommended_gpu_layers']
        
        # メモリ不足時の安全設定
        if available_gb < model_size_gb * 2:
            config.enable_memory_mapping = False  # nommap=True
            config.enable_memory_lock = False     # usemlock=False
            config.context_size = min(config.context_size, 1024)
            config.blas_batch_size = min(config.blas_batch_size, 64)
            config.gpu_layers = 0
        
        return config
    
    def generate_launch_command(self, model_path: str, optimized_config: KoboldCPPConfig) -> str:
        """起動コマンド生成"""
        cmd_parts = [
            "python koboldcpp.py",
            f"--model \"{model_path}\"",
            f"--contextsize {optimized_config.context_size}",
            f"--blasbatchsize {optimized_config.blas_batch_size}",
            f"--blasthreads {optimized_config.blas_threads}",
            f"--port {optimized_config.port}",
            "--skiplauncher"
        ]
        
        # GPU設定
        if optimized_config.gpu_layers > 0:
            cmd_parts.append(f"--gpulayers {optimized_config.gpu_layers}")
            cmd_parts.append("--usecublas normal 0")
        else:
            cmd_parts.append("--gpulayers 0")
        
        # メモリ設定
        if not optimized_config.enable_memory_mapping:
            cmd_parts.append("--nommap")
        
        if not optimized_config.enable_memory_lock:
            cmd_parts.append("--usemlock False")
        
        # 安全性設定
        if optimized_config.enable_noavx2:
            cmd_parts.append("--noavx2")
        
        if optimized_config.enable_failsafe:
            cmd_parts.append("--failsafe")
        
        return " ".join(cmd_parts)
    
    def create_batch_file(self, model_path: str, output_path: str = None) -> str:
        """最適化バッチファイル作成"""
        if not output_path:
            model_name = Path(model_path).stem
            output_path = f"run_{model_name}_optimized.bat"
        
        optimized_config = self.optimize_for_model(model_path)
        launch_command = self.generate_launch_command(model_path, optimized_config)
        
        batch_content = f"""@echo off
REM KoboldCPP最適化起動スクリプト
REM モデル: {Path(model_path).name}
REM 生成日時: {Path().cwd()}

echo 🚀 KoboldCPP最適化起動
echo モデル: {Path(model_path).name}
echo.

REM メモリ監視開始
echo 📊 システム情報:
systeminfo | findstr "Total Physical Memory"
echo.

REM KoboldCPP起動
echo 🔧 KoboldCPP起動中...
{launch_command}

pause
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        print(f"📝 バッチファイル作成: {output_path}")
        return output_path

class ErrorResolver:
    """エラー解決器"""
    
    @staticmethod
    def resolve_bad_alloc_error(model_path: str) -> List[str]:
        """bad_allocエラー解決策"""
        solutions = [
            "🔧 解決策 1: メモリマッピング無効化",
            "  → --nommap フラグを追加",
            "",
            "🔧 解決策 2: コンテキストサイズ削減",
            "  → --contextsize 1024 または 512",
            "",
            "🔧 解決策 3: GPU層数削減",
            "  → --gpulayers 0 (CPU使用)",
            "",
            "🔧 解決策 4: BLASバッチサイズ削減",
            "  → --blasbatchsize 64 または 32",
            "",
            "🔧 解決策 5: AVX2無効化",
            "  → --noavx2 フラグを追加",
            "",
            "🔧 解決策 6: メモリロック無効化",
            "  → --usemlock False",
            "",
            "⚠️ 最終手段: 破損ファイル修復",
            "  → NKAT-LoRA蒸留システムで修復"
        ]
        return solutions
    
    @staticmethod
    def resolve_access_violation_error() -> List[str]:
        """アクセス違反エラー解決策"""
        solutions = [
            "🔧 解決策 1: CLBlast設定調整",
            "  → --useclblast 0 0 --gpulayers 0",
            "",
            "🔧 解決策 2: メモリ保護",
            "  → --failsafe フラグを追加",
            "",
            "🔧 解決策 3: 単一GPU使用",
            "  → 複数GPU環境では1つのGPUのみ使用",
            "",
            "🔧 解決策 4: テンソル分割調整",
            "  → --tensor_split で明示的に分割",
            "",
            "🔧 解決策 5: CPU専用実行",
            "  → 全てのGPU機能を無効化"
        ]
        return solutions

def main():
    """メイン関数"""
    print("🔧 KoboldCPP メモリ最適化・エラー解決システム v1.0")
    print("=" * 60)
    
    if len(sys.argv) < 2:
        print("使用法:")
        print("  python koboldcpp_memory_optimizer.py <model_path> [action]")
        print("")
        print("アクション:")
        print("  optimize  - 最適化設定生成（デフォルト）")
        print("  batch     - 最適化バッチファイル作成")
        print("  analyze   - メモリ分析のみ")
        print("  resolve   - エラー解決策表示")
        print("")
        print("例:")
        print("  python koboldcpp_memory_optimizer.py model.gguf optimize")
        print("  python koboldcpp_memory_optimizer.py model.gguf batch")
        return
    
    model_path = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "optimize"
    
    if not os.path.exists(model_path):
        print(f"❌ ファイルが見つかりません: {model_path}")
        return
    
    config = KoboldCPPConfig()
    optimizer = KoboldCPPOptimizer(config)
    
    if action == "analyze":
        # メモリ分析のみ
        analyzer = MemoryAnalyzer()
        analysis = analyzer.analyze_system_memory()
        
        print("📊 システムメモリ分析:")
        print(f"  総メモリ: {analysis['total_gb']:.1f}GB")
        print(f"  利用可能: {analysis['available_gb']:.1f}GB")
        print(f"  使用中: {analysis['used_gb']:.1f}GB")
        print(f"  使用率: {analysis['usage_percent']:.1f}%")
        print(f"  大型モデル対応: {'はい' if analysis['safe_for_large_models'] else 'いいえ'}")
        print(f"  推奨コンテキスト: {analysis['recommended_context_size']}")
        print(f"  推奨GPU層数: {analysis['recommended_gpu_layers']}")
        
    elif action == "batch":
        # バッチファイル作成
        batch_file = optimizer.create_batch_file(model_path)
        print(f"✅ バッチファイル作成完了: {batch_file}")
        
    elif action == "resolve":
        # エラー解決策表示
        print("🩺 KoboldCPPエラー解決策:")
        print("")
        print("🔴 bad_alloc エラー (tokenizer.ggml.tokens)")
        solutions = ErrorResolver.resolve_bad_alloc_error(model_path)
        for solution in solutions:
            print(solution)
        
        print("")
        print("🔴 access violation エラー")
        solutions = ErrorResolver.resolve_access_violation_error()
        for solution in solutions:
            print(solution)
        
    else:
        # 最適化設定生成
        optimized_config = optimizer.optimize_for_model(model_path)
        launch_command = optimizer.generate_launch_command(model_path, optimized_config)
        
        print("")
        print("🚀 最適化起動コマンド:")
        print(launch_command)
        print("")
        print("📝 設定ファイル保存...")
        
        # 設定をJSONで保存
        config_dict = {
            'model_path': model_path,
            'optimized_config': {
                'max_memory_usage_percent': optimized_config.max_memory_usage_percent,
                'enable_memory_mapping': optimized_config.enable_memory_mapping,
                'enable_memory_lock': optimized_config.enable_memory_lock,
                'blas_batch_size': optimized_config.blas_batch_size,
                'blas_threads': optimized_config.blas_threads,
                'gpu_layers': optimized_config.gpu_layers,
                'context_size': optimized_config.context_size,
                'enable_noavx2': optimized_config.enable_noavx2,
                'enable_failsafe': optimized_config.enable_failsafe,
                'port': optimized_config.port
            },
            'launch_command': launch_command
        }
        
        config_file = f"{Path(model_path).stem}_koboldcpp_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        print(f"💾 設定保存: {config_file}")

if __name__ == "__main__":
    main() 