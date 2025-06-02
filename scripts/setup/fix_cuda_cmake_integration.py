#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA-CMake統合修正スクリプト
NKAT-llama.cpp統合でのCUDA toolset問題を解決
"""

import os
import sys
import subprocess
import logging
import shutil
from pathlib import Path
import time

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cuda_cmake_fix.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class CUDACMakeIntegrationFixer:
    def __init__(self, nkat_dir: str = ".", llama_dir: str = "llama.cpp"):
        self.nkat_dir = Path(nkat_dir).resolve()
        self.llama_dir = Path(llama_dir).resolve()
        self.build_dir = self.llama_dir / "build"
        
        logger.info("🔧 CUDA-CMake統合修正ツール初期化")
        logger.info(f"   NKATディレクトリ: {self.nkat_dir}")
        logger.info(f"   llama.cppディレクトリ: {self.llama_dir}")
        
    def check_cuda_environment(self):
        """CUDA環境の詳細確認"""
        logger.info("🐍 CUDA環境確認中...")
        
        # CUDA Toolkit確認
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\Program Files (x86)\NVIDIA GPU Computing Toolkit\CUDA"
        ]
        
        cuda_version = None
        for cuda_path in cuda_paths:
            if os.path.exists(cuda_path):
                versions = [d for d in os.listdir(cuda_path) if d.startswith('v')]
                if versions:
                    cuda_version = versions[-1]  # 最新バージョン
                    logger.info(f"✅ CUDA {cuda_version} が見つかりました: {cuda_path}")
                    break
        
        if not cuda_version:
            logger.error("❌ CUDA Toolkitが見つかりません")
            return False
            
        # Visual Studio確認
        vs_paths = [
            r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat",
            r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
        ]
        
        vs_found = None
        for vs_path in vs_paths:
            if os.path.exists(vs_path):
                vs_found = vs_path
                logger.info(f"✅ Visual Studio環境が見つかりました: {vs_path}")
                break
                
        if not vs_found:
            logger.warning("⚠️ Visual Studio環境が見つかりません")
            logger.info("📋 Visual Studio Build Toolsインストール手順:")
            logger.info("   1. https://visualstudio.microsoft.com/downloads/")
            logger.info("   2. 'Tools for Visual Studio 2022' セクション")
            logger.info("   3. 'Build Tools for Visual Studio 2022' をダウンロード")
            logger.info("   4. C++ build tools ワークロードを選択してインストール")
            
        return True
        
    def create_fixed_cmake_script(self):
        """修正されたCMakeビルドスクリプトを作成"""
        logger.info("📝 修正CMakeスクリプト作成中...")
        
        cmake_script = """@echo off
chcp 65001 > nul
echo 🔧 NKAT-llama.cpp CUDA統合ビルドスクリプト
echo.

REM Visual Studio環境設定
echo 🔨 Visual Studio環境設定中...
call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" 2>nul
if %errorLevel% NEQ 0 (
    call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Professional\\VC\\Auxiliary\\Build\\vcvars64.bat" 2>nul
    if %errorLevel% NEQ 0 (
        call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Enterprise\\VC\\Auxiliary\\Build\\vcvars64.bat" 2>nul
        if %errorLevel% NEQ 0 (
            call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" 2>nul
            if %errorLevel% NEQ 0 (
                echo ❌ Visual Studio環境が見つかりません
                echo   Visual Studio 2019/2022をインストールしてください
                pause
                exit /b 1
            )
        )
    )
)

echo ✅ Visual Studio環境設定完了

REM CUDA環境確認
nvcc --version >nul 2>&1
if %errorLevel% NEQ 0 (
    echo ❌ CUDA環境が見つかりません
    echo   CUDA Toolkit 11.8以上をインストールしてください
    pause
    exit /b 1
)

echo ✅ CUDA環境確認完了

REM ビルドディレクトリ準備
if exist build rmdir /s /q build
mkdir build
cd build

echo.
echo 🔧 CMake設定実行中...

REM CMake設定（修正版）
cmake .. ^
    -DGGML_CUDA=ON ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCUDA_ARCHITECTURES=86 ^
    -DCMAKE_GENERATOR="Visual Studio 17 2022" ^
    -DCMAKE_GENERATOR_PLATFORM=x64 ^
    -DCMAKE_VS_PLATFORM_NAME=x64

if %errorLevel% NEQ 0 (
    echo ❌ CMake設定に失敗しました
    pause
    exit /b 1
)

echo ✅ CMake設定完了

echo.
echo 🔨 プロジェクトビルド実行中...
cmake --build . --config Release --parallel

if %errorLevel% NEQ 0 (
    echo ❌ ビルドに失敗しました
    pause
    exit /b 1
)

echo.
echo 🎉 ビルド成功！
echo.
echo 📁 実行ファイル場所:
echo   - main.exe: .\\bin\\Release\\main.exe
echo   - server.exe: .\\bin\\Release\\server.exe
echo.
echo 🔧 テスト実行例:
echo   .\\bin\\Release\\main.exe -m ..\\models\\your_model.gguf -p "Hello world"
echo.
pause
"""

        script_path = self.llama_dir / "build_nkat_cuda.bat"
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(cmake_script)
            
        logger.info(f"✅ 修正スクリプト作成完了: {script_path}")
        return script_path
        
    def create_ninja_build_script(self):
        """Ninja使用の軽量ビルドスクリプトを作成"""
        logger.info("📝 Ninjaビルドスクリプト作成中...")
        
        ninja_script = """@echo off
chcp 65001 > nul
echo 🥷 NKAT-llama.cpp Ninja高速ビルド
echo.

REM Ninja確認
ninja --version >nul 2>&1
if %errorLevel% NEQ 0 (
    echo 📦 Ninjaをインストール中...
    winget install Ninja-build.Ninja
    if %errorLevel% NEQ 0 (
        echo ❌ Ninjaインストールに失敗しました
        echo   手動でhttps://github.com/ninja-build/ninja/releasesからダウンロードしてください
        pause
        exit /b 1
    )
)

REM Visual Studio環境設定
call "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat" 2>nul
if %errorLevel% NEQ 0 (
    call "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"
)

REM ビルドディレクトリ準備
if exist build_ninja rmdir /s /q build_ninja
mkdir build_ninja
cd build_ninja

echo 🔧 Ninja CMake設定中...
cmake .. ^
    -G Ninja ^
    -DGGML_CUDA=ON ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCUDA_ARCHITECTURES=86

if %errorLevel% NEQ 0 (
    echo ❌ CMake設定失敗
    pause
    exit /b 1
)

echo 🥷 Ninjaビルド実行中...
ninja

if %errorLevel% NEQ 0 (
    echo ❌ ビルド失敗
    pause
    exit /b 1
)

echo 🎉 Ninjaビルド成功！
echo.
echo 📁 実行ファイル: .\\main.exe, .\\server.exe
echo.
pause
"""

        ninja_path = self.llama_dir / "build_nkat_ninja.bat"
        with open(ninja_path, 'w', encoding='utf-8') as f:
            f.write(ninja_script)
            
        logger.info(f"✅ Ninjaスクリプト作成完了: {ninja_path}")
        return ninja_path
        
    def fix_cmake_files(self):
        """CMakeファイルの修正"""
        logger.info("🔧 CMakeファイル修正中...")
        
        # llama.cpp/CMakeLists.txtの修正
        main_cmake = self.llama_dir / "CMakeLists.txt"
        if main_cmake.exists():
            with open(main_cmake, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # CUDA設定の改善
            if "enable_language(CUDA)" in content:
                # CUDA言語有効化の条件を改善
                content = content.replace(
                    "enable_language(CUDA)",
                    """if(GGML_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()"""
                )
                
            with open(main_cmake, 'w', encoding='utf-8') as f:
                f.write(content)
                
            logger.info("✅ メインCMakeLists.txt修正完了")
            
    def run_integration_fix(self):
        """統合修正の実行"""
        logger.info("🚀 CUDA-CMake統合修正開始")
        
        try:
            # 1. CUDA環境確認
            if not self.check_cuda_environment():
                return False
                
            # 2. CMakeファイル修正
            self.fix_cmake_files()
            
            # 3. ビルドスクリプト作成
            vs_script = self.create_fixed_cmake_script()
            ninja_script = self.create_ninja_build_script()
            
            # 4. ヘルプガイド作成
            self.create_build_guide()
            
            logger.info("🎉 統合修正完了！")
            logger.info("📋 次のステップ:")
            logger.info(f"   1. cd {self.llama_dir}")
            logger.info("   2. .\\build_nkat_cuda.bat を実行")
            logger.info("   または")
            logger.info("   2. .\\build_nkat_ninja.bat を実行（高速）")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 統合修正中にエラー: {e}")
            return False
            
    def create_build_guide(self):
        """ビルドガイドの作成"""
        guide_content = """# NKAT-llama.cpp CUDA統合ビルドガイド

## 📋 必要な環境

1. **Visual Studio 2019/2022**
   - C++ デスクトップ開発ワークロードを含む

2. **CUDA Toolkit 11.8以上**
   - 確認済み: CUDA 12.8

3. **CMake 3.20以上**

## 🔧 ビルド手順

### 方法1: Visual Studioビルド（推奨）
```batch
cd llama.cpp
.\\build_nkat_cuda.bat
```

### 方法2: Ninjaビルド（高速）
```batch
cd llama.cpp
.\\build_nkat_ninja.bat
```

## 🎯 実行テスト

ビルド完了後:
```batch
# Visual Studioビルドの場合
.\\build\\bin\\Release\\main.exe -m models\\your_model.gguf -p "Hello world"

# Ninjaビルドの場合
.\\build_ninja\\main.exe -m models\\your_model.gguf -p "Hello world"
```

## 🔍 トラブルシューティング

### CUDA toolset not found
- Visual Studio C++ Build Toolsをインストール
- vcvars64.batが正しく実行されているか確認

### CMake設定エラー
- CMakeバージョンを確認（3.20以上）
- CUDA Toolkit PATHを確認

### ビルドエラー
- 十分なディスク容量があるか確認
- ウイルス対策ソフトの除外設定を確認

## 📞 サポート

エラーが発生した場合:
1. cuda_cmake_fix.log を確認
2. 環境変数を再設定
3. PowerShellを管理者として再起動
"""

        guide_path = self.llama_dir / "CUDA_BUILD_GUIDE.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
            
        logger.info(f"✅ ビルドガイド作成完了: {guide_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CUDA-CMake統合修正ツール")
    parser.add_argument("--nkat-dir", default=".", help="NKATプロジェクトディレクトリ")
    parser.add_argument("--llama-dir", default="llama.cpp", help="llama.cppディレクトリ")
    
    args = parser.parse_args()
    
    fixer = CUDACMakeIntegrationFixer(args.nkat_dir, args.llama_dir)
    success = fixer.run_integration_fix()
    
    if success:
        print("\n🎉 統合修正が完了しました！")
        print("   詳細は cuda_cmake_fix.log を確認してください")
        sys.exit(0)
    else:
        print("\n❌ 統合修正に失敗しました")
        print("   cuda_cmake_fix.log でエラー詳細を確認してください")
        sys.exit(1)

if __name__ == "__main__":
    main() 