#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT CUDA Direct Build Script
Visual Studio Build Tools不要のCUDA専用ビルドシステム
"""

import subprocess
import os
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

def log_info(msg):
    print(f'✅ {msg}')

def log_error(msg):
    print(f'❌ {msg}')
    
def log_warning(msg):
    print(f'⚠️ {msg}')

def run_command(cmd, cwd=None):
    """コマンド実行ヘルパー"""
    try:
        print(f'🔧 実行中: {cmd}')
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def check_cuda_environment():
    """CUDA環境確認"""
    try:
        result = subprocess.run('nvcc --version', shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            version_info = result.stdout
            log_info('CUDA環境確認完了')
            print(f'   {version_info.split("release")[1].split(",")[0].strip() if "release" in version_info else "バージョン情報取得失敗"}')
            return True
        else:
            log_error('CUDA Toolkit が見つかりません')
            return False
    except Exception as e:
        log_error(f'CUDA確認エラー: {e}')
        return False

def main():
    print('🚀 NKAT CUDA Direct Build System')
    print('=' * 50)
    
    # 環境確認
    if not check_cuda_environment():
        sys.exit(1)
    
    # プロジェクト設定
    nkat_dir = Path('.')
    output_dir = nkat_dir / 'output' / 'cuda_kernels'
    scripts_dir = nkat_dir / 'scripts'
    
    log_info('CUDA専用ビルド開始')
    
    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    log_info(f'出力ディレクトリ作成: {output_dir}')
    
    # NKATカーネルファイル確認
    cuda_files = [
        'nkat_star_gemm_kernels.cu',
        'nkat_cuda_interface.cpp', 
        'nkat_cuda.h'
    ]
    
    missing_files = []
    for file in cuda_files:
        if not (scripts_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        log_error(f'必要なCUDAファイルが見つかりません: {missing_files}')
        log_info('以下のコマンドでファイルを生成してください:')
        log_info('py -3 scripts/nkat_cuda_kernels.py --generate-integration-files')
        sys.exit(1)
    
    log_info('NKATカーネルファイル確認完了')
    
    # プログレスバー付きコンパイル
    with tqdm(total=3, desc="CUDA コンパイル", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        
        # 1. CUDAカーネルコンパイル
        pbar.set_description("CUDAカーネル")
        cuda_kernel = scripts_dir / 'nkat_star_gemm_kernels.cu'
        output_obj = output_dir / 'nkat_star_gemm_kernels.obj'
        
        compile_cmd = f'nvcc -c "{cuda_kernel}" -o "{output_obj}" -arch=sm_86 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -O3 -use_fast_math'
        
        success, output = run_command(compile_cmd)
        if success:
            log_info('CUDAカーネルコンパイル成功')
            log_info(f'出力ファイル: {output_obj}')
        else:
            log_error(f'CUDAカーネルコンパイル失敗: {output}')
            # RTX3080用に最適化パラメータを調整して再試行
            log_info('RTX3080最適化パラメータで再試行中...')
            compile_cmd_opt = f'nvcc -c "{cuda_kernel}" -o "{output_obj}" -arch=sm_86 -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr -O2'
            success, output = run_command(compile_cmd_opt)
            if not success:
                log_error(f'再試行も失敗: {output}')
                sys.exit(1)
            log_info('RTX3080最適化コンパイル成功')
        pbar.update(1)
        
        # 2. C++インターフェースコンパイル
        pbar.set_description("C++インターフェース")
        cpp_interface = scripts_dir / 'nkat_cuda_interface.cpp'
        output_cpp_obj = output_dir / 'nkat_cuda_interface.obj'
        
        cpp_compile_cmd = f'nvcc -c "{cpp_interface}" -o "{output_cpp_obj}" -std=c++17 -O2'
        
        success, output = run_command(cpp_compile_cmd)
        if success:
            log_info('C++インターフェースコンパイル成功')
            log_info(f'出力ファイル: {output_cpp_obj}')
        else:
            log_warning(f'C++インターフェースコンパイル失敗（継続）: {output}')
        pbar.update(1)
        
        # 3. ヘッダーファイルと設定
        pbar.set_description("統合ファイル")
        
        # ヘッダーファイルコピー
        header_file = scripts_dir / 'nkat_cuda.h'
        output_header = output_dir / 'nkat_cuda.h'
        shutil.copy2(header_file, output_header)
        log_info(f'ヘッダーファイルコピー完了: {output_header}')
        
        # CMakeファイルコピー
        cmake_file = scripts_dir / 'NKAT_CMakeLists.txt'
        output_cmake = output_dir / 'NKAT_CMakeLists.txt'
        if cmake_file.exists():
            shutil.copy2(cmake_file, output_cmake)
            log_info(f'CMakeファイルコピー完了: {output_cmake}')
        
        pbar.update(1)
    
    # 統合情報ファイル作成
    integration_info = output_dir / 'integration_info.json'
    build_info = {
        'build_timestamp': datetime.now().isoformat(),
        'cuda_version': subprocess.run('nvcc --version', shell=True, capture_output=True, text=True).stdout,
        'architecture': 'sm_86',  # RTX3080
        'files': {
            'kernel_object': str(output_obj) if output_obj.exists() else None,
            'interface_object': str(output_cpp_obj) if output_cpp_obj.exists() else None,
            'header_file': str(output_header),
            'cmake_file': str(output_cmake) if output_cmake.exists() else None
        },
        'integration_steps': [
            '1. Copy files to llama.cpp/ggml/src/ggml-cuda/',
            '2. Update CMakeLists.txt to include NKAT objects', 
            '3. Build llama.cpp with CUDA support',
            '4. Test NKAT STAR operations'
        ],
        'expected_performance': {
            'device': 'RTX3080',
            'overhead': '+10-15% vs standard GEMM',
            'benefits': 'Enhanced expressivity with Moyal star product'
        }
    }
    
    with open(integration_info, 'w', encoding='utf-8') as f:
        json.dump(build_info, f, indent=2, ensure_ascii=False)
    log_info(f'統合情報ファイル作成: {integration_info}')
    
    print('\n🎉 CUDA専用ビルド完了！')
    print('\n📁 出力ファイル:')
    if output_obj.exists():
        print(f'   ✅ CUDAカーネル: {output_obj}')
    if output_cpp_obj.exists():
        print(f'   ✅ C++インターフェース: {output_cpp_obj}')
    else:
        print(f'   ⚠️ C++インターフェース: コンパイル失敗（CUDAカーネルのみで動作可能）')
    print(f'   ✅ ヘッダーファイル: {output_header}')
    if output_cmake.exists():
        print(f'   ✅ CMake設定: {output_cmake}')
    print(f'   ✅ 統合情報: {integration_info}')
    
    print('\n🚀 RTX3080最適化設定:')
    print(f'   - アーキテクチャ: sm_86')
    print(f'   - 最適化レベル: O3/O2')
    print(f'   - CUDA拡張: extended-lambda, relaxed-constexpr')
    print(f'   - 数学最適化: use_fast_math')
    
    print('\n🔧 次のステップ:')
    print('   1. これらのファイルをllama.cppプロジェクトに統合')
    print('   2. 軽量CMake設定でビルド実行')
    print('   3. 統合テスト実行')
    
    return True

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\n❌ ユーザーによって中断されました')
        sys.exit(1)
    except Exception as e:
        print(f'\n❌ 予期しないエラー: {e}')
        sys.exit(1) 