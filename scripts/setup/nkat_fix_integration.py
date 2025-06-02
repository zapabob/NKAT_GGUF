#!/usr/bin/env python3
"""
NKAT統合エラー修正スクリプト
ggml.cからNKAT関数呼び出しを削除し、CPU専用ビルドを可能にする
"""

import re
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_ggml_c_nkat_issues():
    """ggml.cのNKAT関連エラーを修正"""
    
    print("🔧 NKAT統合エラーの修正...")
    
    ggml_c_path = Path("llama.cpp/ggml/src/ggml.c")
    
    if not ggml_c_path.exists():
        logger.error(f"❌ ggml.cが見つかりません: {ggml_c_path}")
        return False
        
    # ファイルを読み込み
    with open(ggml_c_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    logger.info("📄 ggml.c読み込み完了")
    
    # NKAT関連のエラーのある行を削除または修正
    fixes = [
        # NKAT_STAR_GEMM caseブロックを削除
        (
            r'case GGML_OP_NKAT_STAR_GEMM:\s*\{[^}]*ggml_nkat_star_gemm_back[^}]*\}',
            '// NKAT_STAR_GEMM case removed for CPU-only build'
        ),
        
        # NKAT関数の定義が不足している場合はスタブを追加
        (
            r'(static inline bool ggml_can_nkat_star_gemm\([^)]*\)[^{]*\{[^}]*\})',
            r'// NKAT function stubs for CPU-only build\n\1'
        ),
        
        # NKAT関数宣言をコメントアウト
        (
            r'struct ggml_tensor \* ggml_nkat_star_gemm\(',
            r'// struct ggml_tensor * ggml_nkat_star_gemm('
        )
    ]
    
    modified = False
    for pattern, replacement in fixes:
        if re.search(pattern, content, re.DOTALL):
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            modified = True
            logger.info(f"✅ パターン修正完了: {pattern[:50]}...")
    
    # CMakeLists.txtのNKAT部分をコメントアウト
    cmake_path = Path("llama.cpp/CMakeLists.txt")
    if cmake_path.exists():
        with open(cmake_path, 'r', encoding='utf-8') as f:
            cmake_content = f.read()
            
        # NKAT関連の設定をコメントアウト
        nkat_blocks = [
            r'# NKAT.*?endif\(\)',
            r'# NKAT CUDA Integration.*?endif\(\)'
        ]
        
        for block_pattern in nkat_blocks:
            if re.search(block_pattern, cmake_content, re.DOTALL):
                # ブロック全体をコメントアウト
                def comment_block(match):
                    return '\n'.join(['# ' + line for line in match.group(0).split('\n')])
                
                cmake_content = re.sub(block_pattern, comment_block, cmake_content, flags=re.DOTALL)
                modified = True
                logger.info("✅ CMakeLists.txtのNKATブロックをコメントアウト")
        
        if modified:
            with open(cmake_path, 'w', encoding='utf-8') as f:
                f.write(cmake_content)
    
    # ggml.cを更新
    if modified:
        with open(ggml_c_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info("✅ ggml.c修正完了")
    else:
        logger.info("ℹ️ 修正する箇所が見つかりませんでした")
    
    return True

def clean_nkat_files():
    """NKATファイルを一時的に移動"""
    
    print("📁 NKAT関連ファイルの一時移動...")
    
    nkat_files = [
        "llama.cpp/ggml/src/ggml-cuda/nkat_star_gemm_kernels.cu",
        "llama.cpp/ggml/src/ggml-cuda/nkat_cuda_interface.cpp",
        "llama.cpp/ggml/src/ggml-cuda/nkat_cuda.h"
    ]
    
    backup_dir = Path("emergency_backups/nkat_files_temp")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in nkat_files:
        src = Path(file_path)
        if src.exists():
            dst = backup_dir / src.name
            src.rename(dst)
            logger.info(f"📦 移動: {src} → {dst}")
    
    logger.info("✅ NKATファイル一時移動完了")
    return True

def main():
    """メイン実行関数"""
    
    print("🛠️ NKAT統合エラー修正ツール")
    print("=" * 50)
    
    try:
        # 1. NKATファイルを一時移動
        if not clean_nkat_files():
            logger.error("❌ NKATファイル移動失敗")
            return False
            
        # 2. ggml.cのNKATエラーを修正
        if not fix_ggml_c_nkat_issues():
            logger.error("❌ ggml.c修正失敗")
            return False
            
        print("\n🎉 修正完了!")
        print("=" * 50)
        print("📋 次のステップ:")
        print("   1. cd llama.cpp/build")
        print("   2. cmake --build . --config Release --target llama-cli")
        print("   3. CPU専用でのビルド確認")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 予期しないエラー: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 