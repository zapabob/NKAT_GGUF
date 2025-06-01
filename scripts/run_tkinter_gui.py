#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT-GGUF Tkinter GUI 起動スクリプト

使用方法:
    py -3 scripts/run_tkinter_gui.py

必要な依存関係:
    pip install tkinterdnd2
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """依存関係チェック"""
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'), 
        ('huggingface_hub', 'Hugging Face Hub'),
        ('tkinter', 'Tkinter (標準ライブラリ)'),
    ]
    
    optional_packages = [
        ('tkinterdnd2', 'ドラッグ&ドロップサポート'),
        ('tqdm', '進捗表示'),
        ('psutil', 'ディスク容量チェック'),
    ]
    
    print("📋 依存関係チェック中...")
    
    missing_required = []
    missing_optional = []
    
    # 必須パッケージチェック
    for package, description in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            else:
                __import__(package)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description}")
            missing_required.append(package)
    
    # オプションパッケージチェック
    for package, description in optional_packages:
        try:
            __import__(package)
            print(f"✅ {description}")
        except ImportError:
            print(f"⚠️ {description} (オプション)")
            missing_optional.append(package)
    
    return missing_required, missing_optional

def install_missing_packages(packages):
    """不足パッケージのインストール"""
    if not packages:
        return True
    
    print(f"\n📦 不足パッケージのインストール: {', '.join(packages)}")
    
    try:
        # pipでインストール実行
        cmd = [sys.executable, '-m', 'pip', 'install'] + packages
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ インストール完了")
            return True
        else:
            print(f"❌ インストール失敗: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ インストールエラー: {e}")
        return False

def main():
    """メイン関数"""
    print("🚀 NKAT-GGUF Tkinter GUI 起動中...")
    print("="*50)
    
    # 依存関係チェック
    missing_required, missing_optional = check_dependencies()
    
    # 必須パッケージが不足している場合
    if missing_required:
        print(f"\n❌ 必須パッケージが不足しています: {', '.join(missing_required)}")
        
        # tkinterは別途対応が必要
        if 'tkinter' in missing_required:
            print("⚠️ Tkinterが利用できません。以下を確認してください:")
            print("   - Windowsの場合: Python本体の再インストール")
            print("   - Linuxの場合: sudo apt-get install python3-tk")
            print("   - macOSの場合: brew install python-tk")
            sys.exit(1)
        
        # 自動インストール確認
        response = input("\n自動でインストールしますか？ (y/N): ").lower()
        if response in ['y', 'yes']:
            if not install_missing_packages(missing_required):
                sys.exit(1)
        else:
            print("手動でインストールしてください:")
            print(f"pip install {' '.join(missing_required)}")
            sys.exit(1)
    
    # オプションパッケージの推奨インストール
    if missing_optional:
        print(f"\n💡 推奨パッケージ: {', '.join(missing_optional)}")
        if 'tkinterdnd2' in missing_optional:
            print("   tkinterdnd2: ドラッグ&ドロップ機能のために推奨")
        if 'psutil' in missing_optional:
            print("   psutil: ディスク容量チェック機能のために推奨")
            if sys.version_info >= (3, 12):
                print("   ⚠️ Python 3.12環境では、最新版psutil（5.9.6+）が必要です")
        
        response = input("推奨パッケージをインストールしますか？ (y/N): ").lower()
        if response in ['y', 'yes']:
            # Python 3.12の場合、psutilの特定バージョンを指定
            if 'psutil' in missing_optional and sys.version_info >= (3, 12):
                missing_optional = [pkg if pkg != 'psutil' else 'psutil>=5.9.6' for pkg in missing_optional]
            install_missing_packages(missing_optional)
    
    print("\n" + "="*50)
    
    # スクリプトディレクトリに移動
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)  # プロジェクトルートに移動
    
    # GUIスクリプトのパス確認
    gui_script = script_dir / "nkat_tkinter_gui.py"
    if not gui_script.exists():
        print(f"❌ GUIスクリプトが見つかりません: {gui_script}")
        sys.exit(1)
    
    # GPU情報表示
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🎮 GPU検出: {gpu_name} ({vram_gb:.1f}GB VRAM)")
        else:
            print("💻 CPUモードで動作します")
    except ImportError:
        print("⚠️ PyTorchが利用できません")
    
    # GUI起動
    print("🚀 Tkinter GUI を起動しています...")
    try:
        # GUIモジュールをインポートして実行
        sys.path.insert(0, str(script_dir))
        from nkat_tkinter_gui import main as gui_main
        gui_main()
    except Exception as e:
        print(f"❌ GUI起動エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 