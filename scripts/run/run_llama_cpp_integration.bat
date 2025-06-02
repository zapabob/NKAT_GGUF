@echo off
chcp 65001 > nul
echo 🚀 NKAT-llama.cpp統合システム起動
echo.

REM 管理者権限確認
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ 管理者権限で実行中
) else (
    echo ⚠️ 管理者権限が推奨されます
    echo   右クリックで「管理者として実行」を選択してください
    pause
)

echo.
echo 📋 統合プロセス概要:
echo   1. llama.cppプロジェクト準備
echo   2. NKAT CUDAカーネル統合  
echo   3. CMake設定修正
echo   4. ソースファイル修正
echo   5. コンパイル実行
echo   6. 統合テスト
echo   7. 性能ベンチマーク
echo.

REM Python環境確認（複数コマンド対応）
set PYTHON_CMD=
echo 🐍 Python環境確認中...

REM python コマンドを試行
python --version >nul 2>&1
if %errorLevel% EQU 0 (
    echo ✅ python コマンドが利用可能です
    set PYTHON_CMD=python
    python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    goto PYTHON_FOUND
)

REM py -3 コマンドを試行
py -3 --version >nul 2>&1
if %errorLevel% EQU 0 (
    echo ✅ py -3 コマンドが利用可能です  
    set PYTHON_CMD=py -3
    py -3 -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    goto PYTHON_FOUND
)

REM python3 コマンドを試行
python3 --version >nul 2>&1
if %errorLevel% EQU 0 (
    echo ✅ python3 コマンドが利用可能です
    set PYTHON_CMD=python3
    python3 -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    goto PYTHON_FOUND
)

REM Pythonが見つからない場合
echo ❌ Pythonが見つかりません
echo.
echo 📥 Pythonインストール手順:
echo   1. https://www.python.org/downloads/ にアクセス
echo   2. Python 3.8以上をダウンロード  
echo   3. インストーラーを管理者として実行
echo   4. "Add Python to PATH" を必ずチェック
echo   5. "Install Now" をクリック
echo   6. インストール完了後、PowerShellを再起動
echo   7. check_python_installation.bat で確認
echo.
pause
exit /b 1

:PYTHON_FOUND
echo.

REM 必要ライブラリ確認
echo 📦 必要ライブラリ確認中...
%PYTHON_CMD% -c "import subprocess, shutil, pathlib" 2>nul
if %errorLevel% NEQ 0 (
    echo ❌ 必要なPythonライブラリが不足しています
    pause
    exit /b 1
) else (
    echo ✅ 必要ライブラリ確認完了
)

REM Git確認
git --version >nul 2>&1
if %errorLevel% NEQ 0 (
    echo ❌ Gitが見つかりません
    echo   Git for Windowsをインストールしてください: https://git-scm.com/
    pause
    exit /b 1
) else (
    echo ✅ Git環境確認完了
)

REM CUDA確認
nvcc --version >nul 2>&1
if %errorLevel% NEQ 0 (
    echo ❌ CUDA Toolkitが見つかりません
    echo   CUDA Toolkit 11.8以上をインストールしてください
    echo   https://developer.nvidia.com/cuda-toolkit
    pause
    exit /b 1
) else (
    echo ✅ CUDA環境確認完了
)

REM CMake確認
cmake --version >nul 2>&1
if %errorLevel% NEQ 0 (
    echo ❌ CMakeが見つかりません
    echo   CMake 3.20以上をインストールしてください: https://cmake.org/
    echo.
    echo 💡 CMakeインストール方法:
    echo   Option 1: 公式サイトからダウンロード
    echo   Option 2: Chocolateyを使用 - choco install cmake
    echo   Option 3: wingetを使用 - winget install Kitware.CMake
    echo.
    pause
    exit /b 1
) else (
    echo ✅ CMake環境確認完了
)

echo.
echo 🔧 環境確認完了 - 統合を開始しますか？
echo   [Y] はい - 統合を開始
echo   [N] いいえ - キャンセル
echo.
set /p confirm="選択してください (Y/N): "

if /i "%confirm%"=="Y" goto START_INTEGRATION
if /i "%confirm%"=="y" goto START_INTEGRATION
echo 統合をキャンセルしました。
pause
exit /b 0

:START_INTEGRATION
echo.
echo 🚀 NKAT-llama.cpp統合開始...
echo   使用Pythonコマンド: %PYTHON_CMD%
echo.

REM 統合スクリプト実行
%PYTHON_CMD% "scripts\llama_cpp_nkat_integration.py" --nkat-dir . --llama-dir llama.cpp

if %errorLevel% EQU 0 (
    echo.
    echo 🎉 統合が正常に完了しました！
    echo.
    echo 📁 ファイル出力先:
    echo   - llama.cppディレクトリ: .\llama.cpp\
    echo   - ビルドディレクトリ: .\llama.cpp\build\
    echo   - 統合レポート: .\LLAMA_CPP_INTEGRATION_REPORT.md
    echo.
    echo 🔧 次のステップ:
    echo   1. cd llama.cpp\build
    echo   2. .\main.exe -m ..\models\your_nkat_model.gguf -p "テストプロンプト"
    echo.
    echo 📖 詳細な使用方法は llama_cpp_integration_guide.md をご確認ください
) else (
    echo.
    echo ❌ 統合に失敗しました
    echo 📋 トラブルシューティング:
    echo   - llama_cpp_nkat_integration.log でエラー詳細を確認
    echo   - CUDA環境設定を確認
    echo   - 十分なディスク容量があるか確認
    echo   - ネットワーク接続を確認
)

echo.
pause 