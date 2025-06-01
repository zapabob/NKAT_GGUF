#!/bin/bash
# -*- coding: utf-8 -*-
# 🚀 Google Colab NKAT-GGUF クイックセットアップ
# ワンコマンドでNKAT-GGUFシステムを完全セットアップ

set -e  # エラー時に停止

echo "🚀 NKAT-GGUF Google Colab クイックセットアップを開始します"
echo "========================================================"

# 色付きecho関数
red() { echo -e "\033[31m$1\033[0m"; }
green() { echo -e "\033[32m$1\033[0m"; }
yellow() { echo -e "\033[33m$1\033[0m"; }
blue() { echo -e "\033[34m$1\033[0m"; }

# 進捗表示関数
progress() {
    local current=$1
    local total=$2
    local message=$3
    local percent=$((current * 100 / total))
    printf "\r[%3d%%] %s" "$percent" "$message"
    if [ $current -eq $total ]; then
        echo ""
    fi
}

# エラーハンドリング
error_exit() {
    red "❌ エラー: $1"
    exit 1
}

# 成功メッセージ
success() {
    green "✅ $1"
}

# 環境チェック
echo "🔍 環境チェック中..."

# Python環境確認
if ! command -v python3 &> /dev/null; then
    error_exit "Python3がインストールされていません"
fi

# pip確認
if ! command -v pip &> /dev/null; then
    error_exit "pipがインストールされていません"
fi

# Google Colab環境チェック
python3 -c "import google.colab" 2>/dev/null && {
    success "Google Colab環境を確認"
    COLAB_ENV=true
} || {
    yellow "⚠️ Google Colab環境ではありません（ローカル環境）"
    COLAB_ENV=false
}

# GPU確認
if command -v nvidia-smi &> /dev/null; then
    success "NVIDIA GPU検出済み"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
        echo "  🎮 GPU: $line"
    done
else
    yellow "⚠️ NVIDIA GPUが見つかりません（CPUモードで動作）"
fi

echo ""

# ステップ1: 作業ディレクトリ作成
echo "📁 作業ディレクトリを作成中..."
progress 1 10 "作業ディレクトリ作成中..."

# Colab環境用ディレクトリ
if [ "$COLAB_ENV" = true ]; then
    WORK_DIR="/content/NKAT_GGUF"
    CHECKPOINT_DIR="/content/nkat_checkpoints"
    WORKSPACE_DIR="/content/nkat_workspace"
else
    WORK_DIR="$PWD/NKAT_GGUF"
    CHECKPOINT_DIR="$PWD/nkat_checkpoints"  
    WORKSPACE_DIR="$PWD/nkat_workspace"
fi

# ディレクトリ作成
mkdir -p "$CHECKPOINT_DIR" "$WORKSPACE_DIR/input" "$WORKSPACE_DIR/output" "$WORKSPACE_DIR/temp"
success "作業ディレクトリ作成完了"

# ステップ2: システムダウンロード
echo "📥 NKAT-GGUFシステムをダウンロード中..."
progress 2 10 "GitHubからダウンロード中..."

if [ -d "$WORK_DIR" ]; then
    echo "  既存ディレクトリを削除中..."
    rm -rf "$WORK_DIR"
fi

git clone -q https://github.com/zapabob/NKAT_GGUF.git "$WORK_DIR" || error_exit "GitHubからのダウンロードに失敗"
cd "$WORK_DIR"
success "システムダウンロード完了"

# ステップ3: 基本パッケージインストール
echo "📦 基本パッケージをインストール中..."
progress 3 10 "基本パッケージインストール中..."

pip install -q --upgrade pip
pip install -q numpy>=1.21.0 tqdm matplotlib psutil ipywidgets || error_exit "基本パッケージのインストールに失敗"
success "基本パッケージインストール完了"

# ステップ4: PyTorchインストール
echo "🎮 PyTorch (CUDA対応) をインストール中..."
progress 4 10 "PyTorchインストール中..."

# CUDA利用可能かチェック
if command -v nvidia-smi &> /dev/null; then
    echo "  CUDA対応版PyTorchをインストール中..."
    pip install -q torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cu121 || {
        yellow "⚠️ CUDA版インストール失敗、CPU版を試行中..."
        pip install -q torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
    }
else
    echo "  CPU版PyTorchをインストール中..."
    pip install -q torch>=2.0.0 torchvision>=0.15.0 torchaudio>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
fi
success "PyTorchインストール完了"

# ステップ5: 依存関係確認
echo "🔍 依存関係を確認中..."
progress 5 10 "依存関係確認中..."

python3 -c "
import sys
modules = ['numpy', 'torch', 'tqdm']
for module in modules:
    try:
        __import__(module)
        print(f'✅ {module}: OK')
    except ImportError:
        print(f'❌ {module}: 未インストール')
        sys.exit(1)
" || error_exit "依存関係確認に失敗"

success "依存関係確認完了"

# ステップ6: GPU/CUDA確認
echo "🎮 GPU/CUDA設定を確認中..."
progress 6 10 "GPU/CUDA確認中..."

python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.cuda.is_available():
    print(f'🎮 CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
else:
    print('⚠️ CUDA not available (CPU mode)')
"
success "GPU/CUDA確認完了"

# ステップ7: 設定ファイル作成
echo "⚙️ 設定ファイルを作成中..."
progress 7 10 "設定ファイル作成中..."

# デフォルト設定ファイル作成
cat > scripts/colab_default_config.json << 'EOF'
{
    "enable_ka_operators": true,
    "ka_grid_size": 8,
    "lie_algebra_dim": 4,
    "noncommutative_strength": 0.1,
    "differential_geometric_scale": 0.01,
    "spectral_radius_bound": 1.0,
    "use_64bit_precision": true,
    "data_alignment": 8,
    "enable_cuda_optimization": true,
    "enable_performance_monitoring": true,
    "quantization_aware": true,
    "quantization_bits": 8,
    "max_memory_gb": 15.0,
    "chunk_size_mb": 512,
    "enable_checkpoint": true,
    "checkpoint_interval": 100
}
EOF

success "設定ファイル作成完了"

# ステップ8: サンプルファイル準備
echo "📋 サンプルファイルを準備中..."
progress 8 10 "サンプルファイル準備中..."

# サンプル実行スクリプト作成
cat > run_nkat_colab.py << 'EOF'
#!/usr/bin/env python3
"""
Google Colab用 NKAT-GGUF実行スクリプト
使用方法: python3 run_nkat_colab.py
"""

import sys
import os

# スクリプトディレクトリをパスに追加
script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.append(script_dir)

try:
    from nkat_gguf_colab_main import main
    print("🚀 NKAT-GGUF システムを起動中...")
    main()
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("セットアップスクリプトを再実行してください")
except Exception as e:
    print(f"❌ 実行エラー: {e}")
    print("詳細なエラー情報:")
    import traceback
    traceback.print_exc()
EOF

chmod +x run_nkat_colab.py
success "サンプルファイル準備完了"

# ステップ9: ドキュメント準備
echo "📚 ドキュメントを準備中..."
progress 9 10 "ドキュメント準備中..."

# クイックスタートファイル作成
cat > COLAB_QUICKSTART.md << 'EOF'
# 🚀 NKAT-GGUF Google Colab クイックスタート

## すぐに使い始める

### 1. システム起動
```python
# メインシステム起動
python3 run_nkat_colab.py
```

### 2. 手動起動
```python
import sys
sys.path.append('scripts')
from nkat_gguf_colab_main import main
main()
```

### 3. コマンドライン使用
```python
from scripts.nkat_gguf_colab_main import NKATGGUFConverter, NKATConfig

# 設定作成
config = NKATConfig()

# 変換実行
converter = NKATGGUFConverter(config)
success = converter.convert_to_nkat('input.gguf', 'output.gguf')
```

## 主要ファイル

- `run_nkat_colab.py` - メイン実行スクリプト
- `scripts/nkat_gguf_colab_main.py` - コア機能
- `scripts/colab_default_config.json` - デフォルト設定
- `docs/Google_Colab_NKAT_使用マニュアル.md` - 詳細マニュアル

## サポート

問題が発生した場合:
1. セットアップスクリプトを再実行
2. GitHubでIssueを作成
3. ドキュメントを確認

Happy Converting! 🎉
EOF

success "ドキュメント準備完了"

# ステップ10: 最終検証
echo "🔍 セットアップ最終検証中..."
progress 10 10 "最終検証中..."

# Pythonパス設定テスト
python3 -c "
import sys
sys.path.append('scripts')
try:
    from nkat_gguf_colab_main import NKATConfig
    print('✅ NKAT-GGUFモジュール: インポート成功')
except ImportError as e:
    print(f'❌ インポートエラー: {e}')
    sys.exit(1)
" || error_exit "最終検証に失敗"

success "セットアップ完了！"

# 完了メッセージ
echo ""
echo "========================================================"
green "🎉 NKAT-GGUF Google Colab セットアップ完了！"
echo "========================================================"
echo ""
blue "📋 次の手順:"
echo "  1. システム起動: python3 run_nkat_colab.py"
echo "  2. または手動で: python3 -c \"from scripts.nkat_gguf_colab_main import main; main()\""
echo "  3. 詳細マニュアル: docs/Google_Colab_NKAT_使用マニュアル.md"
echo ""
blue "🎯 主な機能:"
echo "  💾 GGUFファイル最適化"
echo "  🎮 GPU加速処理"
echo "  📊 推論性能向上 (平均15%)"
echo "  🔄 電源断対応リカバリー"
echo "  ☁️ Google Drive連携"
echo ""
blue "📞 サポート:"
echo "  GitHub: https://github.com/zapabob/NKAT_GGUF"
echo "  Issues: https://github.com/zapabob/NKAT_GGUF/issues"
echo ""
green "Happy Converting with NKAT-GGUF! 🚀" 