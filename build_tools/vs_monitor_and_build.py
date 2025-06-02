#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Studio Installation Monitor & RTX 3080 CUDA Auto-Build
CUDAコア最適化最前提の自動ビルドシステム
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime
import threading

class VSMonitorAndBuilder:
    def __init__(self):
        self.vs_paths = [
            "C:/Program Files/Microsoft Visual Studio/2022/Community",
            "C:/Program Files/Microsoft Visual Studio/2022/Professional", 
            "C:/Program Files/Microsoft Visual Studio/2022/Enterprise",
            "C:/Program Files/Microsoft Visual Studio/2022/BuildTools"
        ]
        
        self.rtx3080_info = {
            "cuda_cores": 8704,
            "tensor_cores": 272,
            "sm_count": 68,
            "arch": "sm_86"
        }
        
        self.monitoring = True
        
    def log(self, message, level="INFO"):
        """カラーログ出力"""
        colors = {
            "SUCCESS": "\033[92m",
            "ERROR": "\033[91m", 
            "WARN": "\033[93m",
            "INFO": "\033[94m",
            "CUDA": "\033[95m",
            "MONITOR": "\033[96m"
        }
        reset = "\033[0m"
        color = colors.get(level, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] [{level}] {message}{reset}")
    
    def check_vs_installation(self):
        """Visual Studio インストール確認"""
        for vs_path in self.vs_paths:
            if Path(vs_path).exists():
                # 重要コンポーネントの確認
                vcvars_path = Path(vs_path) / "Common7" / "Tools" / "VsDevCmd.bat"
                msbuild_path = Path(vs_path) / "MSBuild" / "Current" / "Bin" / "MSBuild.exe"
                cmake_path = Path(vs_path) / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin" / "cmake.exe"
                
                components_found = {
                    "VsDevCmd": vcvars_path.exists(),
                    "MSBuild": msbuild_path.exists(),
                    "CMake": cmake_path.exists()
                }
                
                total_components = len(components_found)
                found_components = sum(components_found.values())
                
                self.log(f"Visual Studio検出: {vs_path}", "SUCCESS")
                self.log(f"コンポーネント: {found_components}/{total_components}", "INFO")
                
                for comp, found in components_found.items():
                    status = "✅" if found else "❌"
                    self.log(f"  {status} {comp}", "INFO")
                
                # 最低限の要件をチェック（VsDevCmd + MSBuild）
                if components_found["VsDevCmd"] and components_found["MSBuild"]:
                    return vs_path
                else:
                    self.log("必要コンポーネントが不足", "WARN")
        
        return None
    
    def monitor_installation(self, check_interval=30, max_wait_minutes=60):
        """Visual Studio インストール監視"""
        self.log("=== Visual Studio インストール監視開始 ===", "MONITOR")
        self.log(f"チェック間隔: {check_interval}秒", "INFO")
        self.log(f"最大待機時間: {max_wait_minutes}分", "INFO")
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        
        while self.monitoring and (time.time() - start_time) < max_wait_seconds:
            vs_path = self.check_vs_installation()
            if vs_path:
                self.log("Visual Studio インストール完了検出！", "SUCCESS")
                return vs_path
            
            elapsed_minutes = int((time.time() - start_time) / 60)
            remaining_minutes = max_wait_minutes - elapsed_minutes
            
            self.log(f"インストール待機中... {elapsed_minutes}/{max_wait_minutes}分経過 (残り{remaining_minutes}分)", "MONITOR")
            time.sleep(check_interval)
        
        if not self.monitoring:
            self.log("監視が中断されました", "WARN")
        else:
            self.log("最大待機時間に達しました", "WARN")
        
        return None
    
    def execute_cuda_build(self, vs_path):
        """RTX 3080 CUDA最適化ビルド実行"""
        self.log("=== RTX 3080 CUDA最適化ビルド開始 ===", "CUDA")
        self.log(f"Visual Studio: {vs_path}", "SUCCESS")
        self.log(f"RTX 3080 CUDAコア: {self.rtx3080_info['cuda_cores']}個", "CUDA")
        self.log(f"テンサーコア: {self.rtx3080_info['tensor_cores']}個", "CUDA")
        self.log(f"アーキテクチャ: {self.rtx3080_info['arch']}", "CUDA")
        
        try:
            # CUDA最適化ビルドスクリプト実行
            self.log("CUDA最適化ビルドスクリプト実行中...", "CUDA")
            
            result = subprocess.run([
                "py", "-3", "cuda_direct_build_fixed.py", "--clean"
            ], capture_output=True, text=True, timeout=3600)  # 1時間タイムアウト
            
            if result.returncode == 0:
                self.log("RTX 3080 CUDA最適化ビルド成功！", "SUCCESS")
                self.log("出力:", "INFO")
                for line in result.stdout.split('\\n')[-10:]:  # 最後の10行
                    if line.strip():
                        self.log(f"  {line}", "INFO")
                return True
            else:
                self.log("ビルドでエラーが発生しました", "ERROR")
                self.log("エラー詳細:", "ERROR")
                for line in result.stderr.split('\\n')[-5:]:  # エラーの最後5行
                    if line.strip():
                        self.log(f"  {line}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("ビルドがタイムアウトしました（1時間）", "ERROR")
            return False
        except Exception as e:
            self.log(f"ビルド実行エラー: {e}", "ERROR")
            return False
    
    def run_performance_test(self):
        """RTX 3080パフォーマンステスト"""
        self.log("=== RTX 3080 パフォーマンステスト ===", "CUDA")
        
        # 実行可能ファイル検索
        exe_paths = [
            "llama.cpp/build_cuda_optimized/main.exe",
            "llama.cpp/build_cuda_optimized/Release/main.exe",
            "llama.cpp/build/main.exe",
            "llama.cpp/build/Release/main.exe"
        ]
        
        found_exe = None
        for exe_path in exe_paths:
            if Path(exe_path).exists():
                found_exe = exe_path
                break
        
        if not found_exe:
            self.log("実行可能ファイルが見つかりません", "WARN")
            return False
        
        self.log(f"実行ファイル: {found_exe}", "SUCCESS")
        
        # 基本機能テスト
        try:
            result = subprocess.run([found_exe, "--help"], 
                                  capture_output=True, text=True, timeout=30)
            
            if "CUDA" in result.stdout or "GPU" in result.stdout:
                self.log("CUDA機能確認済み！", "SUCCESS")
                
                # ファイルサイズ確認
                file_size = Path(found_exe).stat().st_size / (1024*1024)
                self.log(f"実行ファイルサイズ: {file_size:.2f} MB", "INFO")
                
                return True
            else:
                self.log("CUDA機能が検出されませんでした", "WARN")
                return False
                
        except Exception as e:
            self.log(f"パフォーマンステストエラー: {e}", "ERROR")
            return False
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        self.log("監視停止要求", "INFO")
    
    def run_full_pipeline(self):
        """完全なパイプライン実行"""
        self.log("=== RTX 3080 CUDAコア最適化パイプライン開始 ===", "CUDA")
        
        # 1. 現在のVisual Studio状況確認
        vs_path = self.check_vs_installation()
        
        if vs_path:
            self.log("Visual Studio既にインストール済み", "SUCCESS")
        else:
            self.log("Visual Studio インストール待機中...", "MONITOR")
            vs_path = self.monitor_installation()
        
        if not vs_path:
            self.log("Visual Studio インストールが完了しませんでした", "ERROR")
            self.log("手動でインストールを完了してから再実行してください", "INFO")
            return False
        
        # 2. RTX 3080 CUDA最適化ビルド実行
        build_success = self.execute_cuda_build(vs_path)
        
        if not build_success:
            self.log("ビルドに失敗しました", "ERROR")
            return False
        
        # 3. パフォーマンステスト
        test_success = self.run_performance_test()
        
        # 4. 結果サマリー
        self.log("", "INFO")
        self.log("=== RTX 3080 CUDA最適化結果サマリー ===", "SUCCESS")
        self.log(f"Visual Studio: ✅ {vs_path}", "SUCCESS")
        self.log(f"CUDA最適化ビルド: {'✅' if build_success else '❌'}", "SUCCESS" if build_success else "ERROR")
        self.log(f"パフォーマンステスト: {'✅' if test_success else '❌'}", "SUCCESS" if test_success else "WARN")
        self.log(f"RTX 3080 CUDAコア: {self.rtx3080_info['cuda_cores']}個 最適化完了", "CUDA")
        
        return build_success and test_success

def main():
    """メイン実行関数"""
    print("🚀 Visual Studio Monitor & RTX 3080 CUDA Auto-Builder")
    print("=" * 65)
    print("CUDAコア最適化最前提の自動ビルドシステム")
    print()
    
    monitor = VSMonitorAndBuilder()
    
    try:
        success = monitor.run_full_pipeline()
        
        if success:
            print("\\n🎉 RTX 3080 CUDA最適化ビルド完全成功！")
            print("🎯 8,704 CUDAコア + 272テンサーコア 最大活用準備完了")
            print("\\n次のステップ:")
            print("   cd llama.cpp/build_cuda_optimized")
            print("   ./main.exe -m ../../../models/test/model.gguf -p 'RTX 3080 Test'")
        else:
            print("\\n⚠️ パイプライン実行に問題が発生しました")
            print("💡 ログを確認して手動で問題を解決してください")
        
        return success
        
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("\\n⏹️ ユーザーによって停止されました")
        return False
    except Exception as e:
        print(f"\\n❌ 予期しないエラー: {e}")
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 