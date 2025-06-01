#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUF File Selector
利用可能なGGUFファイルを表示し、選択できるツール
"""

import os
import sys
from pathlib import Path
from typing import List, Dict

def get_gguf_files(directory: str = ".") -> List[Dict]:
    """ディレクトリからGGUFファイルを検索"""
    gguf_files = []
    search_dirs = [
        "models",
        "output", 
        ".",
        "../test_models",
        "../output"
    ]
    
    for search_dir in search_dirs:
        search_path = Path(directory) / search_dir
        if search_path.exists():
            for gguf_file in search_path.glob("*.gguf"):
                if gguf_file.is_file():
                    file_size_mb = gguf_file.stat().st_size / (1024 * 1024)
                    gguf_files.append({
                        'name': gguf_file.name,
                        'path': str(gguf_file),
                        'size_mb': round(file_size_mb, 2),
                        'directory': search_dir
                    })
    
    # サイズでソート
    gguf_files.sort(key=lambda x: x['size_mb'])
    return gguf_files

def display_gguf_files(files: List[Dict]):
    """GGUFファイル一覧を表示"""
    print("\n🔍 利用可能なGGUFファイル一覧:")
    print("=" * 70)
    
    if not files:
        print("❌ GGUFファイルが見つかりません")
        return
    
    # カテゴリ別に表示
    categories = {}
    for file in files:
        category = file['directory']
        if category not in categories:
            categories[category] = []
        categories[category].append(file)
    
    index = 1
    file_index_map = {}
    
    for category, cat_files in categories.items():
        print(f"\n📁 {category} フォルダー:")
        print("-" * 50)
        
        for file in cat_files:
            size_str = f"{file['size_mb']:.2f} MB"
            if file['size_mb'] < 1:
                size_str = f"{file['size_mb']*1024:.0f} KB"
            elif file['size_mb'] > 1000:
                size_str = f"{file['size_mb']/1024:.2f} GB"
            
            print(f"  {index:2d}. {file['name'][:50]:<50} ({size_str})")
            file_index_map[index] = file
            index += 1
    
    return file_index_map

def get_recommended_files(files: List[Dict]) -> List[Dict]:
    """推奨ファイルを選択"""
    recommended = []
    
    # 推奨基準
    for file in files:
        name = file['name'].lower()
        size_mb = file['size_mb']
        
        # 小さなテストファイル（推奨）
        if (('demo' in name or 'test_large' in name) and 
            size_mb < 50 and 
            'nkat' in name):
            recommended.append(file)
        
        # 中サイズの実用的ファイル
        elif (size_mb > 50 and size_mb < 500 and 
              ('qwen' in name or 'vecteus' in name) and
              'enhanced' in name):
            recommended.append(file)
    
    return recommended[:5]  # 最大5個

def select_gguf_file() -> str:
    """対話的にGGUFファイルを選択"""
    print("🌀 GGUF File Selector")
    print("=" * 50)
    
    # ファイル検索
    files = get_gguf_files()
    
    if not files:
        print("❌ GGUFファイルが見つかりません")
        print("\nヒント:")
        print("- 現在のディレクトリにGGUFファイルがあることを確認してください")
        print("- models/, output/ フォルダーを確認してください")
        return ""
    
    # ファイル一覧表示
    file_map = display_gguf_files(files)
    
    # 推奨ファイル表示
    recommended = get_recommended_files(files)
    if recommended:
        print(f"\n⭐ 推奨ファイル:")
        print("-" * 30)
        for i, file in enumerate(recommended[:3], 1):
            size_str = f"{file['size_mb']:.2f} MB"
            print(f"  🌟 {file['name'][:40]:<40} ({size_str})")
    
    print(f"\n📊 合計: {len(files)}個のGGUFファイル")
    print("\n選択方法:")
    print("  - 番号を入力してファイルを選択")
    print("  - 'q' で終了")
    print("  - 'auto' で推奨ファイルを自動選択")
    
    while True:
        try:
            choice = input("\n👉 選択してください (番号/auto/q): ").strip()
            
            if choice.lower() == 'q':
                return ""
            
            if choice.lower() == 'auto':
                if recommended:
                    selected = recommended[0]
                    print(f"✅ 自動選択: {selected['name']}")
                    return selected['path']
                else:
                    # フォールバック: 最小のテストファイル
                    test_files = [f for f in files if 'test' in f['name'].lower() or 'demo' in f['name'].lower()]
                    if test_files:
                        selected = min(test_files, key=lambda x: x['size_mb'])
                        print(f"✅ 自動選択: {selected['name']}")
                        return selected['path']
                    else:
                        selected = min(files, key=lambda x: x['size_mb'])
                        print(f"✅ 自動選択: {selected['name']}")
                        return selected['path']
            
            # 番号選択
            file_num = int(choice)
            if file_num in file_map:
                selected = file_map[file_num]
                print(f"✅ 選択: {selected['name']}")
                return selected['path']
            else:
                print(f"❌ 無効な番号です。1-{len(files)}の範囲で入力してください")
                
        except ValueError:
            print("❌ 数字、'auto'、または'q'を入力してください")
        except KeyboardInterrupt:
            print("\n\n👋 選択をキャンセルしました")
            return ""

def get_quick_selection() -> str:
    """クイック選択（最初の適切なファイル）"""
    files = get_gguf_files()
    
    if not files:
        return ""
    
    # 推奨ファイルを取得
    recommended = get_recommended_files(files)
    if recommended:
        return recommended[0]['path']
    
    # テストファイルを優先
    test_files = [f for f in files if 'test' in f['name'].lower() or 'demo' in f['name'].lower()]
    if test_files:
        return min(test_files, key=lambda x: x['size_mb'])['path']
    
    # 最小ファイル
    return min(files, key=lambda x: x['size_mb'])['path']

def main():
    """メイン実行"""
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # クイック選択モード
        selected_path = get_quick_selection()
        if selected_path:
            print(f"✅ 自動選択されたファイル: {selected_path}")
            return selected_path
        else:
            print("❌ 利用可能なGGUFファイルが見つかりません")
            return ""
    else:
        # 対話モード
        return select_gguf_file()

if __name__ == "__main__":
    selected_file = main()
    if selected_file:
        print(f"\n🎯 選択されたファイル: {selected_file}")
        
        # 環境変数として設定
        os.environ['SELECTED_GGUF_FILE'] = selected_file
        print(f"📝 環境変数 SELECTED_GGUF_FILE に設定しました")
    else:
        print("\n❌ ファイルが選択されませんでした") 