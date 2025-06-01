#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 Google Colab NKAT-GGUF セットアップスクリプト
Google Colab環境でNKAT-GGUFシステムを簡単にセットアップ

使用方法:
1. Google Colabで新しいノートブックを作成
2. このスクリプトを実行
3. NKAT-GGUF変換システムが自動で起動
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def check_colab_environment():
    """Google Colab環境の確認"""
    try:
        import google.colab
        print("✅ Google Colab環境を確認しました")
        return True
    except ImportError:
        print("⚠️ Google Colab環境ではありません")
        return False

def install_dependencies():
    """必要な依存関係のインストール"""
    print("📦 依存関係をインストール中...")
    
    # 基本パッケージ
    packages = [
        "numpy>=1.21.0",
        "tqdm",
        "ipywidgets",
        "matplotlib",
        "psutil",
    ]
    
    # PyTorchとCUDAサポート
    pytorch_packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0", 
        "torchaudio>=2.0.0"
    ]
    
    try:
        # 基本パッケージインストール
        for package in packages:
            print(f"📥 {package} をインストール中...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)
        
        # PyTorchインストール（CUDA付き）
        print("🎮 PyTorch (CUDA対応) をインストール中...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "--index-url", "https://download.pytorch.org/whl/cu121"
        ] + pytorch_packages, check=True)
        
        print("✅ 依存関係のインストール完了")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ インストールエラー: {e}")
        return False

def setup_workspace():
    """ワークスペースの設定"""
    print("📁 ワークスペースを設定中...")
    
    # 作業ディレクトリ作成
    workspace_dirs = [
        "/content/nkat_workspace",
        "/content/nkat_workspace/input",
        "/content/nkat_workspace/output",
        "/content/nkat_workspace/temp",
        "/content/nkat_checkpoints"
    ]
    
    for dir_path in workspace_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ ワークスペース設定完了")

def download_nkat_system():
    """NKAT-GGUFシステムのダウンロード"""
    print("🚀 NKAT-GGUFシステムをダウンロード中...")
    
    try:
        # GitHubからクローン
        repo_url = "https://github.com/zapabob/NKAT_GGUF.git"
        subprocess.run([
            "git", "clone", "-q", repo_url, "/content/NKAT_GGUF"
        ], check=True)
        
        # Pythonパスに追加
        sys.path.append("/content/NKAT_GGUF/scripts")
        
        print("✅ NKAT-GGUFシステムダウンロード完了")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ ダウンロードエラー: {e}")
        print("📝 手動でGitHubからファイルをアップロードしてください")
        return False

def verify_installation():
    """インストール確認"""
    print("🔍 インストール確認中...")
    
    # 重要なライブラリの確認
    required_modules = [
        "numpy",
        "torch", 
        "tqdm",
        "ipywidgets"
    ]
    
    for module_name in required_modules:
        try:
            importlib.import_module(module_name)
            print(f"✅ {module_name}: OK")
        except ImportError:
            print(f"❌ {module_name}: 未インストール")
            return False
    
    # CUDA確認
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 CUDA: {device_name} ({vram:.1f}GB VRAM)")
        else:
            print("⚠️ CUDA: 利用不可（CPUモードで動作）")
    except Exception as e:
        print(f"⚠️ CUDA確認エラー: {e}")
    
    print("✅ インストール確認完了")
    return True

def display_welcome_message():
    """ウェルカムメッセージ表示"""
    try:
        from IPython.display import display, HTML
        
        welcome_html = """
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin: 20px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
            <h1 style="margin: 0; font-size: 2.5em;">🚀 NKAT-GGUF システム</h1>
            <h2 style="margin: 10px 0; font-weight: 300;">非可換コルモゴロフアーノルド表現理論</h2>
            <p style="margin: 20px 0; font-size: 1.2em;">GGUFファイル最適化システムへようこそ！</p>
            <div style="background: rgba(255,255,255,0.1); border-radius: 10px; padding: 20px; margin: 20px 0;">
                <h3>🎯 主な機能</h3>
                <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
                    <li>💾 GGUF形式の機械学習モデルをNKAT理論で最適化</li>
                    <li>🎮 CUDA GPU加速による高速処理</li>
                    <li>📊 推論速度・精度・メモリ効率の向上</li>
                    <li>🔄 電源断対応リカバリーシステム</li>
                    <li>☁️ Google Drive連携</li>
                </ul>
            </div>
            <p style="font-size: 1.1em; margin-top: 20px;">
                <strong>次のセルでシステムを起動してください！</strong>
            </p>
        </div>
        """
        
        display(HTML(welcome_html))
        
    except ImportError:
        print("=" * 60)
        print("🚀 NKAT-GGUF システム セットアップ完了!")
        print("非可換コルモゴロフアーノルド表現理論によるGGUF最適化")
        print("=" * 60)
        print("📋 主な機能:")
        print("  💾 GGUF形式モデルの最適化")
        print("  🎮 CUDA GPU加速")
        print("  📊 推論性能向上")
        print("  🔄 リカバリーシステム")
        print("  ☁️ Google Drive連携")
        print("=" * 60)
        print("次のセルでシステムを起動してください！")

def create_launch_code():
    """起動用コード生成"""
    launch_code = '''
# NKAT-GGUF システム起動
from nkat_gguf_colab_main import main
main()
'''
    
    try:
        from IPython.display import display, HTML, Javascript
        
        code_html = f"""
        <div style="background: #f8f9fa; border: 2px solid #28a745; border-radius: 10px; padding: 20px; margin: 20px 0;">
            <h3 style="color: #28a745; margin-top: 0;">🔥 次のセルで実行してください:</h3>
            <pre style="background: #343a40; color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto;"><code>{launch_code.strip()}</code></pre>
            <button onclick="copyLaunchCode()" style="background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin-top: 10px;">
                📋 コードをコピー
            </button>
        </div>
        
        <script>
        function copyLaunchCode() {{
            const code = `{launch_code.strip()}`;
            navigator.clipboard.writeText(code).then(function() {{
                alert('コードをクリップボードにコピーしました！');
            }});
        }}
        </script>
        """
        
        display(HTML(code_html))
        
    except ImportError:
        print("\n" + "="*50)
        print("🔥 次のコードを新しいセルで実行してください:")
        print("="*50)
        print(launch_code.strip())
        print("="*50)

def main():
    """メインセットアップ処理"""
    print("🚀 NKAT-GGUF Google Colab セットアップを開始します\n")
    
    # 環境確認
    is_colab = check_colab_environment()
    
    # 依存関係インストール
    if not install_dependencies():
        print("❌ セットアップに失敗しました")
        return False
    
    # ワークスペース設定
    setup_workspace()
    
    # システムダウンロード（オプション）
    download_success = download_nkat_system()
    
    # インストール確認
    if not verify_installation():
        print("❌ セットアップに失敗しました")
        return False
    
    # ウェルカムメッセージ
    display_welcome_message()
    
    # 起動コード表示
    if download_success:
        create_launch_code()
    else:
        print("\n📝 手動セットアップが必要です:")
        print("1. GitHubからNKAT_GGUFをダウンロード")
        print("2. scripts/nkat_gguf_colab_main.py を実行")
    
    print("\n✅ セットアップ完了!")
    return True

if __name__ == "__main__":
    main() 