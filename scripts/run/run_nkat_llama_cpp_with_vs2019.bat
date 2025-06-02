@echo off
chcp 65001 > nul
echo 🚀 NKAT-llama.cpp統合 (VS2019環境)

rem Visual Studio 2019環境設定
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
if %errorlevel% neq 0 (
    echo ❌ Visual Studio 2019環境設定失敗
    pause
    exit /b 1
)

rem CUDA環境設定
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set PATH=%CUDA_PATH%\bin;%PATH%

rem CMake設定（Visual Studio 2019を明示的に指定）
set CMAKE_GENERATOR="Visual Studio 16 2019"
set CMAKE_GENERATOR_PLATFORM=x64

echo ✅ 環境設定完了
echo    Visual Studio: 2019 BuildTools
echo    CUDA Path: %CUDA_PATH%
echo    Generator: %CMAKE_GENERATOR%

rem llama.cppディレクトリ作成
if not exist "llama.cpp\build" mkdir "llama.cpp\build"

rem CMake設定実行（Visual Studio 2019指定）
echo.
echo 🔧 CMake設定中（VS2019）...
cd llama.cpp\build
cmake .. -G %CMAKE_GENERATOR% -A %CMAKE_GENERATOR_PLATFORM% -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHITECTURES=86

if %errorlevel% neq 0 (
    echo ❌ CMake設定失敗
    cd ..\..
    pause
    exit /b 1
)

rem ビルド実行
echo.
echo 🔨 ビルド実行中...
cmake --build . --config Release --target main

if %errorlevel% neq 0 (
    echo ❌ ビルド失敗
    cd ..\..
    pause
    exit /b 1
)

echo.
echo 🎉 NKAT-llama.cpp統合完了！
echo.
echo 📁 実行ファイル: .\Release\main.exe
echo 🔧 テスト実行例:
echo    .\Release\main.exe -m ..\..\models\test\your_model.gguf -p "テストプロンプト"

cd ..\..
pause 