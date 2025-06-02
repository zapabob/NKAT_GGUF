#!/usr/bin/env python3
"""
NKAT-llama.cpp 統合スクリプト
非可換コルモゴロフ-アーノルド表現理論をllama.cppに統合
"""

import os
import shutil
import logging
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nkat_llama_integration.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NKATLlamaCppIntegrator:
    def __init__(self):
        self.base_dir = Path(".")
        self.llama_cpp_dir = self.base_dir / "llama.cpp"
        self.output_dir = self.base_dir / "output" / "cuda_kernels"
        self.cuda_dir = self.llama_cpp_dir / "ggml" / "src" / "ggml-cuda"
        
        # バックアップディレクトリ
        self.backup_dir = self.base_dir / "emergency_backups" / f"nkat_integration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("🌟 NKAT-llama.cpp 統合システム v1.0")
        print("=" * 50)
        logger.info("🌟 NKAT-llama.cpp統合開始")
        
    def create_backup(self):
        """重要ファイルのバックアップ作成"""
        try:
            print("💾 バックアップ作成中...")
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # llama.cppの重要ファイルをバックアップ
            backup_files = [
                "CMakeLists.txt",
                "ggml/src/ggml-cuda/common.cuh",
                "ggml/src/ggml.c",
                "src/llama.cpp"
            ]
            
            for file_path in tqdm(backup_files, desc="バックアップ"):
                src = self.llama_cpp_dir / file_path
                if src.exists():
                    dst = self.backup_dir / file_path
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    
            logger.info(f"✅ バックアップ完了: {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ バックアップ失敗: {str(e)}")
            return False
    
    def copy_nkat_files(self):
        """NKATファイルをllama.cppにコピー"""
        try:
            print("📁 NKATファイルコピー中...")
            
            # CUDAディレクトリが存在することを確認
            if not self.cuda_dir.exists():
                logger.error(f"❌ CUDAディレクトリが見つかりません: {self.cuda_dir}")
                return False
                
            file_mappings = {
                "nkat_star_gemm_kernels.cu": self.cuda_dir / "nkat_star_gemm_kernels.cu",
                "nkat_cuda_interface.cpp": self.cuda_dir / "nkat_cuda_interface.cpp", 
                "nkat_cuda.h": self.cuda_dir / "nkat_cuda.h"
            }
            
            for src_name, dst_path in tqdm(file_mappings.items(), desc="ファイルコピー"):
                src_path = self.output_dir / src_name
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    logger.info(f"📄 コピー完了: {src_name} → {dst_path}")
                else:
                    logger.warning(f"⚠️ ソースファイル未発見: {src_path}")
                    
            return True
            
        except Exception as e:
            logger.error(f"❌ ファイルコピー失敗: {str(e)}")
            return False
    
    def modify_cmake_lists(self):
        """CMakeLists.txtにNKAT設定を追加"""
        try:
            print("⚙️ CMakeLists.txt更新中...")
            
            cmake_file = self.llama_cpp_dir / "CMakeLists.txt"
            if not cmake_file.exists():
                logger.error(f"❌ CMakeLists.txtが見つかりません: {cmake_file}")
                return False
                
            # 現在の内容を読み込み
            with open(cmake_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # NKAT設定を追加（GGML_CUDA使用）
            nkat_cmake_addition = """
# NKAT (Non-commutative Kolmogorov-Arnold Theory) Integration
option(LLAMA_NKAT "Enable NKAT support" ON)

if(LLAMA_NKAT AND GGML_CUDA)
    message(STATUS "🌟 NKAT enabled with CUDA support")
    add_compile_definitions(GGML_USE_NKAT)
    
    # NKAT specific CUDA flags
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DGGML_CUDA_NKAT_ENABLED")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --maxrregcount=64")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_86")
    
    # Add NKAT source files to ggml-cuda
    target_sources(ggml-cuda PRIVATE
        ggml/src/ggml-cuda/nkat_star_gemm_kernels.cu
        ggml/src/ggml-cuda/nkat_cuda_interface.cpp
    )
endif()
"""
            
            # CUDA設定の後に追加
            if "GGML_CUDA" in content and "NKAT" not in content:
                # 適切な位置を見つけて追加
                lines = content.split('\n')
                insert_pos = len(lines)
                
                for i, line in enumerate(lines):
                    if "endif()" in line and "GGML_CUDA" in lines[max(0, i-20):i]:
                        insert_pos = i + 1
                        break
                        
                lines.insert(insert_pos, nkat_cmake_addition)
                content = '\n'.join(lines)
                
                # ファイルに書き戻し
                with open(cmake_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                logger.info("✅ CMakeLists.txt更新完了")
                return True
            else:
                logger.info("ℹ️ NKAT設定は既に存在するか、CUDA設定が見つかりません")
                return True
                
        except Exception as e:
            logger.error(f"❌ CMakeLists.txt更新失敗: {str(e)}")
            return False
    
    def modify_ggml_cuda_common(self):
        """ggml-cuda/common.cuhにNKATヘッダーを追加"""
        try:
            print("🔧 CUDA共通ヘッダー更新中...")
            
            common_file = self.cuda_dir / "common.cuh"
            if not common_file.exists():
                logger.warning(f"⚠️ common.cuhが見つかりません: {common_file}")
                return True  # 致命的ではない
                
            with open(common_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # NKATヘッダーを追加
            if "nkat_cuda.h" not in content:
                nkat_include = '\n#ifdef GGML_CUDA_NKAT_ENABLED\n#include "nkat_cuda.h"\n#endif\n'
                
                # 適切な位置にインクルードを追加
                if "#pragma once" in content:
                    content = content.replace("#pragma once", "#pragma once" + nkat_include)
                else:
                    content = nkat_include + content
                    
                with open(common_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
                logger.info("✅ CUDA共通ヘッダー更新完了")
            else:
                logger.info("ℹ️ NKATヘッダーは既に追加済み")
                
            return True
            
        except Exception as e:
            logger.error(f"❌ CUDA共通ヘッダー更新失敗: {str(e)}")
            return False
    
    def build_with_nkat(self):
        """NKAT有効でビルド実行"""
        try:
            print("🔨 NKAT統合ビルド開始...")
            
            build_dir = self.llama_cpp_dir / "build_nkat"
            build_dir.mkdir(exist_ok=True)
            
            os.chdir(build_dir)
            
            # CMake設定（NKAT有効、GGML_CUDA使用）
            cmake_cmd = [
                "cmake", "..",
                "-G", "Visual Studio 16 2019",
                "-A", "x64",
                "-DGGML_CUDA=ON",
                "-DLLAMA_NKAT=ON",
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCUDA_ARCHITECTURES=86",
                "-DCUDA_TOOLKIT_ROOT_DIR=C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.8"
            ]
            
            logger.info(f"🔧 CMake設定実行: {' '.join(cmake_cmd)}")
            
            import subprocess
            result = subprocess.run(cmake_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ CMake設定失敗:\n{result.stderr}")
                return False
                
            # ビルド実行
            build_cmd = ["cmake", "--build", ".", "--config", "Release", "--target", "llama-cli", "--parallel", "4"]
            logger.info(f"🔨 ビルド実行: {' '.join(build_cmd)}")
            
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"❌ ビルド失敗:\n{result.stderr}")
                return False
                
            logger.info("✅ NKAT統合ビルド完了")
            return True
            
        except Exception as e:
            logger.error(f"❌ ビルド失敗: {str(e)}")
            return False
        finally:
            os.chdir(self.base_dir)
    
    def verify_integration(self):
        """統合の検証"""
        try:
            print("🔍 統合検証中...")
            
            # ビルドされた実行ファイルを確認（新しいターゲット名）
            main_exe = self.llama_cpp_dir / "build_nkat" / "bin" / "Release" / "llama-cli.exe"
            if not main_exe.exists():
                main_exe = self.llama_cpp_dir / "build_nkat" / "Release" / "llama-cli.exe"
                
            if main_exe.exists():
                logger.info(f"✅ 実行ファイル確認: {main_exe}")
                
                # ファイルサイズ確認
                file_size = main_exe.stat().st_size / (1024 * 1024)
                logger.info(f"📏 ファイルサイズ: {file_size:.2f} MB")
                
                return True
            else:
                logger.error("❌ 実行ファイルが見つかりません")
                return False
                
        except Exception as e:
            logger.error(f"❌ 統合検証失敗: {str(e)}")
            return False
    
    def generate_integration_report(self, success):
        """統合レポート生成"""
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "integration_success": success,
                "nkat_version": "0.2",
                "cuda_architecture": "sm_86",
                "backup_location": str(self.backup_dir),
                "files_integrated": [
                    "nkat_star_gemm_kernels.cu",
                    "nkat_cuda_interface.cpp", 
                    "nkat_cuda.h"
                ],
                "build_configuration": {
                    "ggml_cuda": True,
                    "llama_nkat": True,
                    "cmake_generator": "Visual Studio 16 2019",
                    "cuda_toolkit": "v12.8"
                }
            }
            
            report_file = self.base_dir / "nkat_integration_report.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            logger.info(f"📊 統合レポート生成: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ レポート生成失敗: {str(e)}")
    
    def run_integration(self):
        """統合プロセス実行"""
        success = False
        
        try:
            # ステップ実行
            steps = [
                ("バックアップ作成", self.create_backup),
                ("NKATファイルコピー", self.copy_nkat_files),
                ("CMakeLists.txt更新", self.modify_cmake_lists),
                ("CUDA共通ヘッダー更新", self.modify_ggml_cuda_common),
                ("NKAT統合ビルド", self.build_with_nkat),
                ("統合検証", self.verify_integration)
            ]
            
            for step_name, step_func in steps:
                print(f"\n🔄 {step_name}...")
                if not step_func():
                    logger.error(f"❌ {step_name}で失敗しました")
                    break
            else:
                success = True
                print("\n🎉 NKAT-llama.cpp統合完了!")
                print("=" * 50)
                print("📋 次のステップ:")
                print("   1. cd llama.cpp/build_nkat/Release")
                print("   2. ./llama-cli.exe -m ../../../models/test/sample.gguf --nkat-enable")
                print("   3. Moyal star product による非可換推論を体験")
                
        except Exception as e:
            logger.error(f"❌ 統合プロセス失敗: {str(e)}")
        finally:
            self.generate_integration_report(success)
            
        return success

def main():
    integrator = NKATLlamaCppIntegrator()
    return integrator.run_integration()

if __name__ == "__main__":
    main() 