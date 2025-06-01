#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT GGUF Colab Selector
Google Colab用GGUFファイル選択ツール
"""

import os
import glob
from pathlib import Path

def find_gguf_files():
    """利用可能なGGUFファイルを検索"""
    print("🔍 GGUFファイルを検索中...")
    
    # 検索パス
    search_paths = [
        "/content/drive/MyDrive/*.gguf",
        "/content/drive/MyDrive/*/*.gguf", 
        "/content/*.gguf",
        "/content/*/*.gguf",
        "*.gguf",
        "*/*.gguf",
        "models/*.gguf",
        "output/*.gguf"
    ]
    
    found_files = []
    
    for pattern in search_paths:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            if os.path.isfile(file):
                size_mb = os.path.getsize(file) / (1024 * 1024)
                found_files.append({
                    'path': file,
                    'name': os.path.basename(file),
                    'size_mb': round(size_mb, 2)
                })
    
    # 重複除去とソート
    unique_files = {}
    for file in found_files:
        unique_files[file['path']] = file
    
    sorted_files = sorted(unique_files.values(), key=lambda x: x['size_mb'])
    return sorted_files

def select_recommended_file(files):
    """推奨ファイルを自動選択"""
    if not files:
        return None
    
    # 推奨基準: 小さなファイル優先
    small_files = [f for f in files if f['size_mb'] < 50]
    if small_files:
        # demo, test, nkat が含まれるファイルを優先
        for keywords in [['demo'], ['test'], ['nkat'], ['small']]:
            for file in small_files:
                if any(keyword in file['name'].lower() for keyword in keywords):
                    return file
        # 最小ファイル
        return small_files[0]
    
    # 50MB未満がない場合は最小ファイル
    return files[0]

def save_selected_file(file_path):
    """選択されたファイルパスを保存"""
    try:
        # Colab環境用の設定ファイル
        config_content = f'''# NKAT GGUF Configuration
SELECTED_GGUF_FILE = "{file_path}"
GGUF_FILE_NAME = "{os.path.basename(file_path)}"
GGUF_FILE_SIZE_MB = {os.path.getsize(file_path) / (1024 * 1024):.2f}
'''
        
        with open('nkat_gguf_config.py', 'w', encoding='utf-8') as f:
            f.write(config_content)
        
        # 環境変数としても設定
        os.environ['SELECTED_GGUF_FILE'] = file_path
        os.environ['GGUF_FILE_NAME'] = os.path.basename(file_path)
        
        print(f"💾 設定ファイル nkat_gguf_config.py に保存されました")
        
    except Exception as e:
        print(f"⚠️ 設定保存エラー: {e}")

def main():
    """メイン処理"""
    print("🌀 NKAT GGUF Colab Selector")
    print("=" * 50)
    
    # ファイル検索
    files = find_gguf_files()
    
    if not files:
        print("❌ GGUFファイルが見つかりません")
        print("\n💡 ヒント:")
        print("  - Google Driveをマウントしてください")
        print("  - GGUFファイルがマイドライブにあることを確認してください")
        print("  - ファイルがアップロード済みか確認してください")
        return None
    
    # ファイル一覧表示
    print(f"\n📂 見つかったGGUFファイル: {len(files)}個")
    print("-" * 60)
    
    for i, file in enumerate(files[:10], 1):  # 最初の10個まで表示
        size_str = f"{file['size_mb']:.2f} MB"
        if file['size_mb'] > 1000:
            size_str = f"{file['size_mb']/1024:.2f} GB"
        print(f"  {i:2d}. {file['name'][:40]:<40} ({size_str:>10})")
    
    if len(files) > 10:
        print(f"  ... および他 {len(files) - 10} ファイル")
    
    # 推奨ファイル選択
    recommended = select_recommended_file(files)
    
    if recommended:
        print(f"\n⭐ 推奨ファイル:")
        size_str = f"{recommended['size_mb']:.2f} MB"
        if recommended['size_mb'] > 1000:
            size_str = f"{recommended['size_mb']/1024:.2f} GB"
        print(f"   📄 {recommended['name']}")
        print(f"   📂 {recommended['path']}")
        print(f"   📊 {size_str}")
        
        # 自動選択を保存
        save_selected_file(recommended['path'])
        
        print(f"\n✅ GGUFファイルが自動選択されました!")
        print(f"   使用方法: from nkat_gguf_config import SELECTED_GGUF_FILE")
        
        return recommended['path']
    
    else:
        print("❌ 推奨ファイルを選択できませんでした")
        return None

# 直接実行用の追加関数
def get_selected_file():
    """選択されたファイルパスを取得"""
    try:
        from nkat_gguf_config import SELECTED_GGUF_FILE
        return SELECTED_GGUF_FILE
    except ImportError:
        # 設定ファイルがない場合は自動実行
        return main()

def quick_select():
    """クイック選択（非対話モード）"""
    files = find_gguf_files()
    if files:
        recommended = select_recommended_file(files)
        if recommended:
            save_selected_file(recommended['path'])
            print(f"✅ 自動選択: {recommended['name']}")
            return recommended['path']
    print("❌ 利用可能なGGUFファイルがありません")
    return None

if __name__ == "__main__":
    selected_file = main()
    if selected_file:
        print(f"\n🎯 最終選択: {selected_file}")
    else:
        print("\n❌ ファイル選択に失敗しました") 