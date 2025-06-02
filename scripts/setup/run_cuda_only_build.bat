@echo off
chcp 65001 > nul
echo 🔧 CUDA専用ビルドスクリプト - Visual Studio不要
echo.

REM 管理者権限確認
net session >nul 2>&1
if %errorLevel% == 0 (
    echo ✅ 管理者権限で実行中
) else (
    echo ⚠️ 管理者権限が推奨されます
    echo   右クリックで「管理者として実行」を選択してください
)

echo.
echo 📋 CUDA専用ビルド概要:
echo   - CUDAコンパイラ (nvcc) を直接使用
echo   - Visual Studio Build Tools不要
echo   - 軽量で高速なビルド
echo.

REM Python環境確認
set PYTHON_CMD=
echo 🐍 Python環境確認中...

py -3 --version >nul 2>&1
if %errorLevel% EQU 0 (
    echo ✅ py -3 コマンドが利用可能です  
    set PYTHON_CMD=py -3
    py -3 -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    goto PYTHON_FOUND
)

python --version >nul 2>&1
if %errorLevel% EQU 0 (
    echo ✅ python コマンドが利用可能です
    set PYTHON_CMD=python
    python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    goto PYTHON_FOUND
)

echo ❌ Pythonが見つかりません
pause
exit /b 1

:PYTHON_FOUND
echo.

REM CUDA確認
nvcc --version >nul 2>&1
if %errorLevel% NEQ 0 (
    echo ❌ CUDA Toolkitが見つかりません
    pause
    exit /b 1
) else (
    echo ✅ CUDA環境確認完了
    nvcc --version | findstr "release"
)

echo.
echo 🔧 CUDA専用ビルド開始しますか？
echo   [Y] はい - ビルドを開始
echo   [N] いいえ - キャンセル
echo.
set /p confirm="選択してください (Y/N): "

if /i "%confirm%"=="Y" goto START_BUILD
if /i "%confirm%"=="y" goto START_BUILD
echo ビルドをキャンセルしました。
pause
exit /b 0

:START_BUILD
echo.
echo 🚀 CUDA専用ビルド開始...
echo   使用Pythonコマンド: %PYTHON_CMD%
echo.

REM CUDA専用ビルドスクリプト実行
%PYTHON_CMD% -c "
import subprocess
import os
import sys
from pathlib import Path

def log_info(msg):
    print(f'✅ {msg}')

def log_error(msg):
    print(f'❌ {msg}')
    
def run_command(cmd, cwd=None):
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

# プロジェクト設定
nkat_dir = Path('.')
llama_dir = nkat_dir / 'llama.cpp'
output_dir = nkat_dir / 'output' / 'cuda_kernels'

log_info('CUDA専用ビルド開始')

# 出力ディレクトリ作成
output_dir.mkdir(parents=True, exist_ok=True)
log_info(f'出力ディレクトリ作成: {output_dir}')

# NKATカーネルファイル確認
cuda_files = [
    'scripts/nkat_star_gemm_kernels.cu',
    'scripts/nkat_cuda_interface.cpp', 
    'scripts/nkat_cuda.h'
]

missing_files = []
for file in cuda_files:
    if not (nkat_dir / file).exists():
        missing_files.append(file)

if missing_files:
    log_error(f'必要なCUDAファイルが見つかりません: {missing_files}')
    sys.exit(1)

log_info('NKATカーネルファイル確認完了')

# CUDAカーネル直接コンパイル
cuda_kernel = nkat_dir / 'scripts' / 'nkat_star_gemm_kernels.cu'
output_obj = output_dir / 'nkat_star_gemm_kernels.obj'

compile_cmd = f'nvcc -c \"{cuda_kernel}\" -o \"{output_obj}\" -arch=sm_86 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr'

log_info('CUDAカーネルコンパイル中...')
log_info(f'実行コマンド: {compile_cmd}')

success, output = run_command(compile_cmd)
if success:
    log_info('CUDAカーネルコンパイル成功')
    log_info(f'出力ファイル: {output_obj}')
else:
    log_error(f'CUDAカーネルコンパイル失敗: {output}')
    sys.exit(1)

# C++インターフェースコンパイル (nvccを使用)
cpp_interface = nkat_dir / 'scripts' / 'nkat_cuda_interface.cpp'
output_cpp_obj = output_dir / 'nkat_cuda_interface.obj'

cpp_compile_cmd = f'nvcc -c \"{cpp_interface}\" -o \"{output_cpp_obj}\" -std=c++17'

log_info('C++インターフェースコンパイル中...')
log_info(f'実行コマンド: {cpp_compile_cmd}')

success, output = run_command(cpp_compile_cmd)
if success:
    log_info('C++インターフェースコンパイル成功')
    log_info(f'出力ファイル: {output_cpp_obj}')
else:
    log_error(f'C++インターフェースコンパイル失敗: {output}')
    # C++コンパイル失敗は警告として継続
    print(f'⚠️ C++インターフェースコンパイル失敗: {output}')

# ヘッダーファイルコピー
header_file = nkat_dir / 'scripts' / 'nkat_cuda.h'
output_header = output_dir / 'nkat_cuda.h'

import shutil
shutil.copy2(header_file, output_header)
log_info(f'ヘッダーファイルコピー完了: {output_header}')

# llama.cpp統合用設定ファイル作成
integration_info = output_dir / 'integration_info.txt'
with open(integration_info, 'w', encoding='utf-8') as f:
    f.write('NKAT CUDA Kernel Integration Info\\n')
    f.write('===============================\\n\\n')
    f.write(f'Build Date: {subprocess.run(\"date /t\", shell=True, capture_output=True, text=True).stdout.strip()}\\n')
    f.write(f'CUDA Version: {subprocess.run(\"nvcc --version\", shell=True, capture_output=True, text=True).stdout}\\n')
    f.write(f'Kernel Object: {output_obj}\\n')
    f.write(f'Interface Object: {output_cpp_obj}\\n')
    f.write(f'Header File: {output_header}\\n\\n')
    f.write('Integration Steps:\\n')
    f.write('1. Copy files to llama.cpp/ggml/src/ggml-cuda/\\n')
    f.write('2. Update CMakeLists.txt to include NKAT objects\\n')
    f.write('3. Build llama.cpp with CUDA support\\n')

log_info(f'統合情報ファイル作成: {integration_info}')

print('\\n🎉 CUDA専用ビルド完了！')
print('\\n📁 出力ファイル:')
if output_obj.exists():
    print(f'   - CUDAカーネル: {output_obj}')
if output_cpp_obj.exists():
    print(f'   - C++インターフェース: {output_cpp_obj}')
print(f'   - ヘッダーファイル: {output_header}')
print(f'   - 統合情報: {integration_info}')
print('\\n🔧 次のステップ:')
print('   1. これらのファイルをllama.cppプロジェクトに統合')
print('   2. 軽量CMake設定でビルド実行')
print('   3. 統合テスト実行')
"

if %errorLevel% EQU 0 (
    echo.
    echo 🎉 CUDA専用ビルドが正常に完了しました！
    echo.
    echo 📁 出力先: .\output\cuda_kernels\
    echo.
    echo 🔧 次のステップ:
    echo   1. 統合情報を確認: .\output\cuda_kernels\integration_info.txt
    echo   2. llama.cpp軽量統合実行: run_lightweight_integration.bat
    echo.
) else (
    echo.
    echo ❌ CUDA専用ビルドに失敗しました
    echo 📋 確認事項:
    echo   - CUDA Toolkit 11.8以上がインストール済み
    echo   - scripts/フォルダにNKATファイルが存在
    echo   - 十分なディスク容量があること
)

echo.
pause 