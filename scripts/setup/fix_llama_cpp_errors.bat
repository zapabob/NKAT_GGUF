@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

REM ========================================================================
REM 🔧 llama.cpp エラー修復統合ツール
REM llama.cpp Error Fix Integrated Tool
REM 
REM 機能:
REM - MoEモデル量子化タイプ不一致修復
REM - KoboldCPP緊急修復
REM - トークナイザーエラー修復
REM - 統合分析とレポート
REM ========================================================================

echo.
echo ========================================================================
echo 🔧 llama.cpp エラー修復統合ツール v1.0
echo ========================================================================
echo.

REM Python環境確認
py -3 --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python 3が見つかりません
    echo 💡 Python 3.8以上をインストールしてください
    pause
    exit /b 1
)

echo ✅ Python環境確認完了

REM 必要なディレクトリ作成
if not exist "emergency_backups" mkdir emergency_backups
if not exist "output" mkdir output

echo.
echo 📁 利用可能なGGUFファイル:
echo ========================================================================
set file_count=0
for %%f in (*.gguf) do (
    set /a file_count+=1
    echo !file_count!. %%f
)
for %%f in (models\*.gguf) do (
    set /a file_count+=1
    echo !file_count!. %%f
)
for %%f in (downloads\*.gguf) do (
    set /a file_count+=1
    echo !file_count!. %%f
)

if %file_count%==0 (
    echo ⚠️ GGUFファイルが見つかりません
    echo 💡 修復したいGGUFファイルのパスを手動で入力してください
)

echo.
echo 🔧 修復オプション:
echo ========================================================================
echo 1. 📊 包括的分析（推奨・最初に実行）
echo 2. 🧠 MoEモデル量子化修復
echo 3. 🆘 KoboldCPP緊急修復
echo 4. 📝 トークナイザー修復のみ
echo 5. ⚙️ KoboldCPP起動設定作成
echo 6. 🔄 統合修復（全問題対応）
echo 7. ❓ ヘルプ・詳細情報
echo.

set /p choice="選択してください (1-7): "

if "%choice%"=="1" goto analyze
if "%choice%"=="2" goto moe_fix
if "%choice%"=="3" goto kobold_fix
if "%choice%"=="4" goto tokenizer_fix
if "%choice%"=="5" goto create_config
if "%choice%"=="6" goto comprehensive_fix
if "%choice%"=="7" goto help
goto invalid_choice

:analyze
echo.
echo 📊 包括的分析を開始します...
set /p filepath="GGUFファイルのパス: "
if not exist "%filepath%" (
    echo ❌ ファイルが見つかりません: %filepath%
    pause
    exit /b 1
)

echo 🔍 分析中...
py -3 scripts/koboldcpp_emergency_fix.py "%filepath%" --analyze-only
echo.
echo ✅ 分析完了
pause
exit /b 0

:moe_fix
echo.
echo 🧠 MoE量子化修復を開始します...
set /p filepath="GGUFファイルのパス: "
if not exist "%filepath%" (
    echo ❌ ファイルが見つかりません: %filepath%
    pause
    exit /b 1
)

echo 🔧 MoEモデル修復中...
py -3 scripts/llama_cpp_moe_fix.py "%filepath%"

if errorlevel 1 (
    echo ❌ MoE修復に失敗しました
) else (
    echo ✅ MoE修復完了
)
pause
exit /b 0

:kobold_fix
echo.
echo 🆘 KoboldCPP緊急修復を開始します...
set /p filepath="GGUFファイルのパス: "
if not exist "%filepath%" (
    echo ❌ ファイルが見つかりません: %filepath%
    pause
    exit /b 1
)

echo 🔧 KoboldCPP緊急修復中...
py -3 scripts/koboldcpp_emergency_fix.py "%filepath%" --comprehensive

if errorlevel 1 (
    echo ❌ 緊急修復に失敗しました
) else (
    echo ✅ 緊急修復完了
)
pause
exit /b 0

:tokenizer_fix
echo.
echo 📝 トークナイザー修復を開始します...
set /p filepath="GGUFファイルのパス: "
if not exist "%filepath%" (
    echo ❌ ファイルが見つかりません: %filepath%
    pause
    exit /b 1
)

echo 🔧 トークナイザー修復中...
py -3 scripts/koboldcpp_emergency_fix.py "%filepath%" --tokenizer-only

if errorlevel 1 (
    echo ❌ トークナイザー修復に失敗しました
) else (
    echo ✅ トークナイザー修復完了
)
pause
exit /b 0

:create_config
echo.
echo ⚙️ KoboldCPP起動設定作成を開始します...
set /p filepath="GGUFファイルのパス: "
if not exist "%filepath%" (
    echo ❌ ファイルが見つかりません: %filepath%
    pause
    exit /b 1
)

echo 📝 起動設定作成中...
py -3 scripts/koboldcpp_emergency_fix.py "%filepath%" --create-config

if errorlevel 1 (
    echo ❌ 起動設定作成に失敗しました
) else (
    echo ✅ 起動設定作成完了
)
pause
exit /b 0

:comprehensive_fix
echo.
echo 🔄 統合修復（全問題対応）を開始します...
echo ⚠️ この処理には時間がかかる場合があります
set /p filepath="GGUFファイルのパス: "
if not exist "%filepath%" (
    echo ❌ ファイルが見つかりません: %filepath%
    pause
    exit /b 1
)

echo 🔧 統合修復中...
echo 📊 ステップ1: 包括的分析...
py -3 scripts/koboldcpp_emergency_fix.py "%filepath%" --analyze-only

echo 🧠 ステップ2: MoE修復...
py -3 scripts/llama_cpp_moe_fix.py "%filepath%"

echo 🆘 ステップ3: KoboldCPP緊急修復...
py -3 scripts/koboldcpp_emergency_fix.py "%filepath%" --comprehensive

echo ⚙️ ステップ4: 起動設定作成...
py -3 scripts/koboldcpp_emergency_fix.py "%filepath%" --create-config

echo.
echo ✅ 統合修復完了
echo 💡 修復されたファイルは以下の場所にあります:
echo    - emergency_backups\ (バックアップ)
echo    - *.gguf_*fixed.gguf (修復版)
echo    - run_*_optimized.bat (起動設定)
pause
exit /b 0

:help
echo.
echo ❓ llama.cpp エラー修復ツール - 詳細情報
echo ========================================================================
echo.
echo 🎯 主な対応エラー:
echo   - MoE（Mixture of Experts）量子化タイプ不一致
echo   - tokenizer.ggml.tokens bad_allocエラー
echo   - KoboldCPPアクセス違反エラー
echo   - メモリ不足エラー
echo.
echo 📚 参考資料:
echo   - GitHub討論: https://github.com/ggerganov/llama.cpp/discussions/9299
echo   - llama.cpp最新版対応
echo.
echo 🔧 修復手順（推奨）:
echo   1. まず「包括的分析」でファイルの問題を特定
echo   2. 問題に応じて適切な修復オプションを選択
echo   3. MoE問題がある場合はMoE修復を実行
echo   4. トークナイザー問題がある場合はトークナイザー修復
echo   5. 最後にKoboldCPP起動設定を作成
echo.
echo 💾 バックアップ:
echo   - すべての修復前に自動バックアップを作成
echo   - emergency_backupsフォルダに保存
echo.
echo 📁 ファイル構成:
echo   - scripts/llama_cpp_moe_fix.py: MoE修復専用
echo   - scripts/koboldcpp_emergency_fix.py: 統合修復システム
echo   - emergency_backups/: バックアップ保存先
echo   - *.log: 修復ログファイル
echo.
echo 🚨 注意事項:
echo   - Windows PowerShell環境で実行
echo   - Python 3.8以上が必要
echo   - 大容量ファイルの場合は時間がかかります
echo   - CUDA対応GPUがある場合は自動で最適化
echo.
pause
exit /b 0

:invalid_choice
echo ❌ 無効な選択です
echo 💡 1-7の数字を入力してください
pause
goto main

:main
echo 🔄 メニューに戻ります...
timeout /t 2 >nul
cls
goto start 