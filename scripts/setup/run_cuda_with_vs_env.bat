@echo off
echo 🚀 NKAT CUDA環境セットアップ

rem Visual Studio環境設定
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 (
    echo ❌ Visual Studio環境設定失敗
    pause
    exit /b 1
)

rem CUDA環境設定
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_PATH%\bin;%PATH%

echo ✅ 環境設定完了
echo    Visual Studio: C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat
echo    CUDA Path: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8

rem CUDAコンパイルテスト
echo.
echo 🔧 CUDAコンパイルテスト開始...
py -3 cuda_direct_build_fixed.py

pause
