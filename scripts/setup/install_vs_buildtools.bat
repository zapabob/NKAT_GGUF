@echo off
chcp 65001 > nul
echo 🔧 Visual Studio Build Tools 2022 自動インストーラー
echo.

REM 管理者権限確認
net session >nul 2>&1
if %errorLevel% NEQ 0 (
    echo ❌ 管理者権限が必要です
    echo   右クリックで「管理者として実行」を選択してください
    pause
    exit /b 1
)

echo ✅ 管理者権限で実行中
echo.

REM 一時ディレクトリ作成
set TEMP_DIR=%TEMP%\VSBuildTools
if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"

echo 📥 Visual Studio Build Tools 2022 をダウンロード中...
echo   ファイルサイズ: 約4MB、ダウンロード時間: 約1-3分
echo.

REM PowerShellでダウンロード
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vs_buildtools.exe' -OutFile '%TEMP_DIR%\vs_buildtools.exe'}"

if not exist "%TEMP_DIR%\vs_buildtools.exe" (
    echo ❌ ダウンロードに失敗しました
    echo   インターネット接続を確認してください
    pause
    exit /b 1
)

echo ✅ ダウンロード完了
echo.
echo 🔨 Visual Studio Build Tools 2022 インストール開始...
echo   以下のコンポーネントがインストールされます:
echo   - MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)
echo   - Windows 11 SDK (10.0.22621.0)
echo   - CMake tools for Visual Studio
echo   - MSVC v143 - VS 2022 C++ x64/x86 Spectre-mitigated libs (Latest)
echo.
echo   インストール時間: 約10-20分
echo   インストール中は他のプログラムを閉じることを推奨します
echo.

REM インストール実行（サイレントモード）
"%TEMP_DIR%\vs_buildtools.exe" --quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 --add Microsoft.VisualStudio.Component.Windows11SDK.22621 --add Microsoft.VisualStudio.Component.VC.CMake.Project --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset

if %errorLevel% EQU 0 (
    echo.
    echo 🎉 Visual Studio Build Tools 2022 インストール完了！
    echo.
    echo 📁 インストール場所:
    echo   C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\
    echo.
    echo 🔧 次のステップ:
    echo   1. PowerShellを再起動
    echo   2. 以下のコマンドで環境変数を設定:
    echo      "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
    echo   3. run_llama_cpp_integration.bat を再実行
    echo.
    echo ✅ CUDA-CMake統合の準備が完了しました
) else (
    echo.
    echo ❌ インストールに失敗しました (エラーコード: %errorLevel%)
    echo.
    echo 🔧 手動インストール手順:
    echo   1. https://visualstudio.microsoft.com/downloads/
    echo   2. "Tools for Visual Studio 2022" セクション
    echo   3. "Build Tools for Visual Studio 2022" をダウンロード
    echo   4. インストーラーを管理者として実行
    echo   5. "C++ build tools" ワークロードを選択
    echo   6. 以下のコンポーネントを追加:
    echo      - MSVC v143 - VS 2022 C++ x64/x86 build tools (Latest)
    echo      - Windows 11 SDK (10.0.22621.0) 
    echo      - CMake tools for Visual Studio
    echo   7. "インストール" をクリック
)

REM 一時ファイル削除
if exist "%TEMP_DIR%\vs_buildtools.exe" del "%TEMP_DIR%\vs_buildtools.exe"
if exist "%TEMP_DIR%" rmdir "%TEMP_DIR%"

echo.
pause 