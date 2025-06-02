@echo off
chcp 65001 > nul
echo 🐍 Python環境確認スクリプト
echo.

echo 📋 インストール確認中...
echo.

REM 複数のPythonコマンドを試行
echo ▶️ python --version で確認中...
python --version 2>nul
if %errorLevel% EQU 0 (
    echo ✅ python コマンドが利用可能です
    python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    goto CHECK_LIBRARIES
) else (
    echo ❌ python コマンドが見つかりません
)

echo.
echo ▶️ py -3 --version で確認中...
py -3 --version 2>nul
if %errorLevel% EQU 0 (
    echo ✅ py -3 コマンドが利用可能です
    py -3 -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    goto CHECK_LIBRARIES
) else (
    echo ❌ py -3 コマンドが見つかりません
)

echo.
echo ▶️ python3 --version で確認中...
python3 --version 2>nul
if %errorLevel% EQU 0 (
    echo ✅ python3 コマンドが利用可能です
    python3 -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    goto CHECK_LIBRARIES
) else (
    echo ❌ python3 コマンドが見つかりません
)

echo.
echo ❌ Pythonがインストールされていないか、PATHに追加されていません
echo.
echo 📥 Pythonインストール手順:
echo   1. https://www.python.org/downloads/ にアクセス
echo   2. "Download Python 3.x.x" をクリック
echo   3. インストーラーを管理者として実行
echo   4. "Add Python to PATH" をチェック
echo   5. "Install Now" をクリック
echo   6. インストール完了後、PowerShellを再起動
echo.
goto END

:CHECK_LIBRARIES
echo.
echo 📦 必要ライブラリ確認中...

REM 標準ライブラリ確認
python -c "import subprocess, shutil, pathlib, os, sys, json, re, logging, datetime" 2>nul
if %errorLevel__ EQU 0 (
    echo ✅ 必要な標準ライブラリはすべて利用可能です
) else (
    echo ❌ 一部の標準ライブラリが不足しています
    py -3 -c "import subprocess, shutil, pathlib, os, sys, json, re, logging, datetime" 2>nul
    if %errorLevel% EQU 0 (
        echo ✅ py -3 で必要な標準ライブラリが利用可能です
    )
)

echo.
echo 🎉 Python環境確認完了！
echo   NKAT-llama.cpp統合の実行準備が整いました。
echo.

:END
echo.
echo 📖 次のステップ:
echo   1. PowerShell/コマンドプロンプトを再起動
echo   2. .\run_llama_cpp_integration.bat を実行
echo.
pause 