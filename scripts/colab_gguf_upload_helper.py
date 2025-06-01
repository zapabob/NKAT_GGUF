#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Colab GGUF アップロードヘルパー
Colab GGUF Upload Helper with Multiple Methods
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Dict
import zipfile
import requests
from tqdm import tqdm

def detect_colab_environment():
    """Colab環境検出"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

class ColabGGUFUploader:
    """Colab GGUF アップローダー"""
    
    def __init__(self):
        self.is_colab = detect_colab_environment()
        self.drive_mounted = False
        
    def method_1_google_drive(self):
        """方法1: Google Drive経由アップロード"""
        print("📁 方法1: Google Drive経由でのアップロード")
        print("=" * 50)
        
        if not self.is_colab:
            print("❌ この方法はGoogle Colab専用です")
            return False
        
        try:
            # Google Driveマウント
            from google.colab import drive
            print("🔗 Google Driveをマウント中...")
            drive.mount('/content/drive')
            self.drive_mounted = True
            print("✅ Google Driveマウント完了")
            
            # GGUFファイル検索
            drive_path = Path('/content/drive/MyDrive')
            gguf_files = list(drive_path.glob('**/*.gguf'))
            
            if gguf_files:
                print(f"\n📂 発見されたGGUFファイル:")
                for i, file_path in enumerate(gguf_files[:10]):  # 最初の10個表示
                    size_mb = file_path.stat().st_size / (1024*1024)
                    print(f"  {i+1}. {file_path.name} ({size_mb:.1f} MB)")
                
                print(f"\n💡 使用方法:")
                print(f"   # ファイルを作業ディレクトリにコピー")
                print(f"   !cp '/content/drive/MyDrive/your_model.gguf' '/content/'")
                print(f"   # または直接パスを指定")
                print(f"   input_path = '/content/drive/MyDrive/your_model.gguf'")
                
                return True
            else:
                print("⚠️ GGUFファイルが見つかりません")
                print("\n📋 Google Driveへのアップロード手順:")
                print("1. PCでdrive.google.comを開く")
                print("2. GGUFファイルをドラッグ&ドロップでアップロード")
                print("3. アップロード完了後、このスクリプトを再実行")
                
                return False
                
        except Exception as e:
            print(f"❌ Google Driveマウントエラー: {e}")
            return False
    
    def method_2_direct_upload(self):
        """方法2: 直接アップロード（小さなファイル用）"""
        print("\n📤 方法2: 直接アップロード（<100MB推奨）")
        print("=" * 50)
        
        if not self.is_colab:
            print("❌ この方法はGoogle Colab専用です")
            return False
        
        try:
            from google.colab import files
            print("📁 ファイル選択ダイアログを開いています...")
            print("⚠️ 注意: 大きなファイル（>100MB）は時間がかかる場合があります")
            
            uploaded = files.upload()
            
            if uploaded:
                for filename, data in uploaded.items():
                    size_mb = len(data) / (1024*1024)
                    print(f"✅ アップロード完了: {filename} ({size_mb:.1f} MB)")
                    
                    # ファイル保存
                    with open(f'/content/{filename}', 'wb') as f:
                        f.write(data)
                    
                    print(f"💾 保存完了: /content/{filename}")
                
                return True
            else:
                print("❌ ファイルがアップロードされませんでした")
                return False
                
        except Exception as e:
            print(f"❌ アップロードエラー: {e}")
            return False
    
    def method_3_url_download(self):
        """方法3: URL直接ダウンロード（強化版）"""
        print("\n🌐 方法3: URL直接ダウンロード")
        print("=" * 50)
        
        print("💡 サポートされるURL:")
        print("  🤗 Hugging Face: https://huggingface.co/user/model/resolve/main/model.gguf")
        print("  📦 GitHub Release: https://github.com/user/repo/releases/download/tag/model.gguf")
        print("  🔗 直接リンク: https://example.com/model.gguf")
        print("  📋 Hugging Face自動変換: https://huggingface.co/user/model")
        
        # Hugging Face URL例
        print("\n🤗 Hugging Face URL例:")
        hf_examples = [
            "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/pytorch_model.bin",
            "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.q4_0.bin",
            "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf"
        ]
        for i, example in enumerate(hf_examples, 1):
            print(f"  {i}. {example}")
        
        try:
            # URLの入力を促す
            print("\n📝 URLを入力してください（またはスキップでEnter）:")
            if self.is_colab:
                url = input("URL: ").strip()
            else:
                # 非Colab環境ではサンプルURL
                url = ""
                print("スキップしました（非Colab環境）")
            
            if not url:
                print("⏭️ URLダウンロードをスキップしました")
                return False
            
            # URL変換・検証
            processed_url, filename = self._process_download_url(url)
            
            print(f"🎯 処理URL: {processed_url}")
            print(f"📁 ファイル名: {filename}")
            
            # ダウンロード実行（リトライ付き）
            return self._download_with_retry(processed_url, filename, max_retries=3)
            
        except Exception as e:
            print(f"❌ URL処理エラー: {e}")
            return False
    
    def _process_download_url(self, url: str) -> tuple:
        """URL処理・変換"""
        url = url.strip()
        
        # Hugging Face URL変換
        if "huggingface.co/" in url and "/resolve/" not in url:
            # モデルページURLを直接リンクに変換
            if url.endswith('/'):
                url = url[:-1]
            
            # 一般的なGGUFファイル名を試行
            common_gguf_names = [
                "model.gguf",
                "ggml-model.gguf", 
                "pytorch_model.gguf",
                "model.q4_0.gguf",
                "model.q4_K_M.gguf"
            ]
            
            print("🔍 Hugging Face モデルページを検出")
            print("📋 一般的なGGUFファイル名を検索中...")
            
            for filename in common_gguf_names:
                test_url = f"{url}/resolve/main/{filename}"
                print(f"  🧪 試行: {filename}")
                
                try:
                    response = requests.head(test_url, timeout=10)
                    if response.status_code == 200:
                        print(f"  ✅ 発見: {filename}")
                        return test_url, filename
                except:
                    continue
            
            # 見つからない場合はユーザーに再入力を促す
            print("  ❌ 一般的なGGUFファイルが見つかりません")
            if self.is_colab:
                manual_filename = input("📝 ファイル名を手動入力してください (例: model.gguf): ").strip()
                if manual_filename:
                    manual_url = f"{url}/resolve/main/{manual_filename}"
                    return manual_url, manual_filename
            
            raise ValueError("GGUFファイルが見つかりません")
        
        # GitHub Release URL検証
        elif "github.com/" in url and "/releases/" in url:
            filename = Path(url).name
            if not filename.endswith('.gguf'):
                filename += '.gguf'
            return url, filename
        
        # 直接リンク
        else:
            filename = Path(url).name
            if not filename.endswith('.gguf'):
                filename += '.gguf'
            return url, filename
    
    def _download_with_retry(self, url: str, filename: str, max_retries: int = 3) -> bool:
        """リトライ機能付きダウンロード"""
        
        for attempt in range(max_retries):
            try:
                print(f"\n⬇️ ダウンロード試行 {attempt + 1}/{max_retries}: {filename}")
                
                # HEADリクエストでファイルサイズ取得
                head_response = requests.head(url, timeout=30)
                total_size = int(head_response.headers.get('content-length', 0))
                
                if total_size > 0:
                    size_gb = total_size / (1024**3)
                    print(f"📊 ファイルサイズ: {size_gb:.2f} GB")
                    
                    if size_gb > 10:
                        print("⚠️ 大容量ファイルです。ダウンロードに時間がかかる場合があります")
                
                # ダウンロード実行
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                
                # プログレス付き保存
                output_path = f'/content/{filename}'
                
                with open(output_path, 'wb') as f:
                    # tqdm進捗バー設定
                    progress_bar_config = {
                        'total': total_size,
                        'unit': 'B',
                        'unit_scale': True,
                        'unit_divisor': 1024,
                        'desc': f'⬇️ {filename}',
                        'ncols': 80,
                        'ascii': True,
                        'colour': 'green'
                    }
                    
                    with tqdm(**progress_bar_config) as pbar:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                chunk_size = len(chunk)
                                downloaded += chunk_size
                                pbar.update(chunk_size)
                                
                                # 進捗情報更新
                                if total_size > 0:
                                    progress_percent = (downloaded / total_size) * 100
                                    if downloaded % (1024 * 1024 * 10) == 0:  # 10MB毎
                                        pbar.set_postfix({
                                            'Speed': f'{chunk_size/1024:.1f}KB/s',
                                            'ETA': f'{((total_size - downloaded) / chunk_size):.0f}s'
                                        })
                
                # ダウンロード完了検証
                if os.path.exists(output_path):
                    actual_size = os.path.getsize(output_path)
                    actual_size_mb = actual_size / (1024*1024)
                    
                    print(f"✅ ダウンロード完了!")
                    print(f"   📁 ファイル: {output_path}")
                    print(f"   📊 サイズ: {actual_size_mb:.1f} MB")
                    
                    # サイズ検証
                    if total_size > 0:
                        size_match = abs(actual_size - total_size) < 1024  # 1KB以内
                        if size_match:
                            print(f"   ✅ サイズ検証: OK")
                        else:
                            print(f"   ⚠️ サイズ不一致: 期待{total_size}, 実際{actual_size}")
                    
                    return True
                else:
                    raise FileNotFoundError(f"ダウンロードファイルが見つかりません: {output_path}")
                    
            except requests.RequestException as e:
                print(f"   ❌ ネットワークエラー (試行 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # 2秒, 4秒, 6秒
                    print(f"   ⏳ {wait_time}秒後にリトライします...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ 最大リトライ回数に達しました")
                    
            except Exception as e:
                print(f"   ❌ 予期しないエラー (試行 {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"   ⏳ {wait_time}秒後にリトライします...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ 最大リトライ回数に達しました")
        
        return False
    
    def method_4_zip_upload(self):
        """方法4: ZIP圧縮アップロード"""
        print("\n📦 方法4: ZIP圧縮アップロード")
        print("=" * 50)
        
        print("💡 この方法の手順:")
        print("1. PCでGGUFファイルをZIP圧縮")
        print("2. ZIPファイルをColabにアップロード")  
        print("3. Colab内で解凍")
        
        if not self.is_colab:
            print("❌ この方法はGoogle Colab専用です")
            return False
        
        try:
            from google.colab import files
            print("\n📁 ZIPファイルを選択してください...")
            
            uploaded = files.upload()
            
            if uploaded:
                for filename, data in uploaded.items():
                    if filename.endswith('.zip'):
                        print(f"📦 ZIP解凍中: {filename}")
                        
                        # ZIPファイル保存
                        zip_path = f'/content/{filename}'
                        with open(zip_path, 'wb') as f:
                            f.write(data)
                        
                        # 解凍
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall('/content/')
                        
                        # ZIPファイル削除
                        os.remove(zip_path)
                        
                        # 解凍されたGGUFファイル検索
                        gguf_files = list(Path('/content/').glob('**/*.gguf'))
                        
                        if gguf_files:
                            for gguf_file in gguf_files:
                                size_mb = gguf_file.stat().st_size / (1024*1024)
                                print(f"✅ 解凍完了: {gguf_file.name} ({size_mb:.1f} MB)")
                            return True
                        else:
                            print("⚠️ 解凍されたGGUFファイルが見つかりません")
                            return False
                    else:
                        print(f"⚠️ ZIPファイルではありません: {filename}")
                        return False
            else:
                print("❌ ファイルがアップロードされませんでした")
                return False
                
        except Exception as e:
            print(f"❌ ZIP処理エラー: {e}")
            return False
    
    def list_uploaded_files(self):
        """アップロード済みファイル一覧"""
        print("\n📋 アップロード済みGGUFファイル一覧:")
        print("=" * 50)
        
        # 作業ディレクトリ検索
        content_files = list(Path('/content/').glob('*.gguf'))
        
        # Google Drive検索（マウント済みの場合）
        drive_files = []
        if self.drive_mounted:
            try:
                drive_files = list(Path('/content/drive/MyDrive').glob('**/*.gguf'))
            except:
                pass
        
        all_files = content_files + drive_files
        
        if all_files:
            for i, file_path in enumerate(all_files):
                try:
                    size_mb = file_path.stat().st_size / (1024*1024)
                    location = "作業ディレクトリ" if str(file_path).startswith('/content/') and 'drive' not in str(file_path) else "Google Drive"
                    print(f"  {i+1}. {file_path.name}")
                    print(f"      パス: {file_path}")
                    print(f"      サイズ: {size_mb:.1f} MB")
                    print(f"      場所: {location}")
                    print()
                except Exception as e:
                    print(f"  ❌ {file_path}: 情報取得エラー ({e})")
        else:
            print("  📭 GGUFファイルが見つかりません")
        
        return all_files
    
    def run_interactive_upload(self):
        """インタラクティブアップロード"""
        print("🚀 Google Colab GGUF アップロードヘルパー")
        print("=" * 50)
        
        if not self.is_colab:
            print("⚠️ Google Colab環境ではありませんが、デモモードで実行します")
        
        # 既存ファイル確認
        existing_files = self.list_uploaded_files()
        
        if existing_files:
            print("✅ 既にGGUFファイルが利用可能です")
            response = input("\n新しいファイルをアップロードしますか？ (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                return existing_files
        
        print("\n📋 アップロード方法を選択してください:")
        print("1. Google Drive経由（推奨・大ファイル対応）")
        print("2. 直接アップロード（<100MB推奨）")
        print("3. URL直接ダウンロード")
        print("4. ZIP圧縮アップロード")
        print("5. 既存ファイル一覧のみ表示")
        
        while True:
            try:
                if self.is_colab:
                    choice = input("\n選択 (1-5): ").strip()
                else:
                    choice = "1"  # デモ用
                    print(f"選択 (1-5): {choice}")
                
                if choice == "1":
                    success = self.method_1_google_drive()
                    break
                elif choice == "2":
                    success = self.method_2_direct_upload()
                    break
                elif choice == "3":
                    success = self.method_3_url_download()
                    break
                elif choice == "4":
                    success = self.method_4_zip_upload()
                    break
                elif choice == "5":
                    self.list_uploaded_files()
                    break
                else:
                    print("⚠️ 1-5の数字を入力してください")
                    continue
                    
            except KeyboardInterrupt:
                print("\n👋 アップロードをキャンセルしました")
                break
            except Exception as e:
                print(f"❌ エラー: {e}")
                break
        
        # 最終ファイル一覧
        return self.list_uploaded_files()


def main():
    """メイン実行"""
    uploader = ColabGGUFUploader()
    files = uploader.run_interactive_upload()
    
    print("\n🎯 次のステップ:")
    print("1. アップロードしたGGUFファイルパスを確認")
    print("2. NKAT統合システムで処理:")
    print("   # クイックスタート版")
    print("   !python colab_nkat_quickstart.py")
    print("   # 完全版")
    print("   !python run_integrated_nkat_system.py")

if __name__ == "__main__":
    main() 