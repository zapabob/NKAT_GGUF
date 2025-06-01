#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 高機能NKAT-GGUF変換システム実行スクリプト
Google Colab環境で高機能GUIを起動
"""

import sys
import os
from pathlib import Path

# スクリプトディレクトリをパスに追加
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir))

try:
    # GUI拡張機能をインポート
    from nkat_gui_extensions import setup_gui_extensions
    from nkat_gguf_advanced_gui import AdvancedNKATGUI
    
    # Google Colab環境検出
    from google.colab import drive, files
    import IPython.display as display
    from IPython.display import clear_output, HTML
    import ipywidgets as widgets
    
    COLAB_ENV = True
    print("✅ Google Colab環境を検出しました")
    
except ImportError as e:
    print(f"⚠️ Google Colab環境でない可能性があります: {e}")
    COLAB_ENV = False

def display_welcome_banner():
    """ウェルカムバナー表示"""
    welcome_html = """
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                border-radius: 15px; 
                padding: 30px; 
                text-align: center; 
                margin: 20px 0;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
        <h1 style="margin: 0; font-size: 2.5em;">🎨 NKAT-GGUF 高機能変換システム</h1>
        <h2 style="margin: 10px 0; font-weight: 300;">Non-commutative Kolmogorov-Arnold Theory</h2>
        <div style="background: rgba(255,255,255,0.2); 
                    border-radius: 10px; 
                    padding: 20px; 
                    margin: 20px 0;">
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap;">
                <div style="margin: 5px;">
                    <h3>🚀 DND対応</h3>
                    <p>ドラッグ&ドロップで簡単ファイル選択</p>
                </div>
                <div style="margin: 5px;">
                    <h3>🤗 HF連携</h3>
                    <p>Hugging Face直接ダウンロード</p>
                </div>
                <div style="margin: 5px;">
                    <h3>💾 自動バックアップ</h3>
                    <p>変換前の安全なファイル保護</p>
                </div>
                <div style="margin: 5px;">
                    <h3>📚 履歴管理</h3>
                    <p>過去の変換履歴を完全記録</p>
                </div>
            </div>
        </div>
        <p style="font-size: 1.2em; margin: 0;">🧠 数学的最適化による高品質GGUF変換</p>
    </div>
    """
    
    display.display(HTML(welcome_html))

def check_system_requirements():
    """システム要件チェック"""
    print("🔍 システム要件チェック中...")
    
    requirements = {
        "Google Colab": COLAB_ENV,
        "Python >= 3.7": sys.version_info >= (3, 7),
        "scripts directory": script_dir.exists(),
        "main conversion script": (script_dir / "nkat_gguf_colab_main.py").exists()
    }
    
    all_ok = True
    for req, status in requirements.items():
        icon = "✅" if status else "❌"
        print(f"{icon} {req}: {'OK' if status else 'NG'}")
        if not status:
            all_ok = False
    
    if not all_ok:
        print("\n⚠️ 一部要件が満たされていません。以下を確認してください：")
        print("1. Google Colab環境で実行していますか？")
        print("2. 必要なスクリプトファイルが存在しますか？")
        print("3. 依存関係がインストールされていますか？")
        return False
    
    print("✅ 全ての要件を満たしています！")
    return True

def install_dependencies():
    """依存関係インストール"""
    print("📦 依存関係をインストール中...")
    
    try:
        # 必要なパッケージをインポートテスト
        import numpy
        import tqdm
        import ipywidgets
        print("✅ 主要パッケージが既にインストールされています")
        
    except ImportError:
        print("📥 不足しているパッケージをインストール中...")
        os.system("pip install numpy tqdm ipywidgets --quiet")
        print("✅ パッケージインストール完了")

def setup_google_drive():
    """Google Drive セットアップガイド"""
    setup_html = """
    <div style="background: #f8f9fa; 
                border: 2px solid #007bff; 
                border-radius: 10px; 
                padding: 20px; 
                margin: 20px 0;">
        <h3 style="color: #007bff; margin-top: 0;">📁 Google Drive連携セットアップ</h3>
        <p><strong>推奨：</strong> より安全で高速な処理のため、Google Driveの利用をお勧めします。</p>
        <ol>
            <li>「📁 Google Drive接続」ボタンをクリック</li>
            <li>Googleアカウントでログイン</li>
            <li>権限を許可</li>
            <li>/content/drive/MyDrive にファイルが利用可能になります</li>
        </ol>
        <p><em>注：Drive接続はオプションです。ローカルアップロードも可能です。</em></p>
    </div>
    """
    display.display(HTML(setup_html))

def main():
    """メイン実行関数"""
    # 出力クリア
    if COLAB_ENV:
        clear_output(wait=True)
    
    # ウェルカムメッセージ
    display_welcome_banner()
    
    # システムチェック
    if not check_system_requirements():
        print("❌ システム要件を満たしていないため終了します")
        return
    
    # 依存関係インストール
    install_dependencies()
    
    # Drive セットアップガイド
    setup_google_drive()
    
    # GUI拡張機能セットアップ
    print("\n🔧 高機能GUI拡張をセットアップ中...")
    extensions = setup_gui_extensions()
    
    # メインGUI起動
    print("🎨 高機能GUIを起動中...")
    gui = AdvancedNKATGUI()
    
    # セットアップ完了メッセージ
    completion_html = """
    <div style="background: #d4edda; 
                border: 2px solid #28a745; 
                border-radius: 10px; 
                padding: 20px; 
                margin: 20px 0; 
                text-align: center;">
        <h3 style="color: #155724; margin-top: 0;">🎉 セットアップ完了！</h3>
        <p style="font-size: 1.1em; margin: 10px 0;">
            高機能NKAT-GGUF変換システムが利用可能になりました
        </p>
        <div style="background: rgba(255,255,255,0.7); 
                    border-radius: 5px; 
                    padding: 15px; 
                    margin: 15px 0;">
            <strong>🚀 使用方法:</strong><br>
            1. ファイルをドラッグ&ドロップまたはHF URLを入力<br>
            2. 必要に応じて設定を調整<br>
            3. 「🚀 一括NKAT変換実行」ボタンをクリック
        </div>
        <p><em>RTX3080対応・電源断リカバリー機能・自動バックアップ付き</em></p>
    </div>
    """
    display.display(HTML(completion_html))
    
    print("✅ 高機能NKAT-GGUF変換システム起動完了！")
    print("📋 上記のインターフェースをご利用ください。")

if __name__ == "__main__":
    main() 