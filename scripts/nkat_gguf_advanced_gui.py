#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎨 高機能GUI付きNKAT-GGUF変換システム
- Hugging Face URL入力
- ドラッグ&ドロップファイル選択
- 自動バックアップ機能
- 参照ファイル記憶機能
- 履歴管理
"""

import os
import sys
import json
import time
import shutil
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback

# Google Colab環境検出
try:
    from google.colab import drive, files
    import IPython.display as display
    from IPython.display import clear_output, HTML, Javascript
    import ipywidgets as widgets
    from tqdm.notebook import tqdm
    COLAB_ENV = True
except ImportError:
    from tqdm import tqdm
    COLAB_ENV = False

# NKAT変換システムをインポート
from nkat_gguf_colab_main import NKATGGUFConverter, NKATConfig, HuggingFaceDownloader

class FileHistory:
    """ファイル履歴管理"""
    
    def __init__(self, history_file: str = "/content/nkat_file_history.json"):
        self.history_file = Path(history_file)
        self.history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """履歴読み込み"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []
    
    def _save_history(self):
        """履歴保存"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 履歴保存エラー: {e}")
    
    def add_entry(self, file_info: Dict):
        """履歴エントリ追加"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'filename': file_info.get('filename', ''),
            'path': file_info.get('path', ''),
            'size_gb': file_info.get('size_gb', 0),
            'source': file_info.get('source', 'upload'),  # 'upload', 'hf_download'
            'hf_repo': file_info.get('hf_repo', ''),
            'status': 'added'
        }
        self.history.insert(0, entry)  # 最新を先頭に
        self.history = self.history[:50]  # 最新50件まで保持
        self._save_history()
    
    def get_recent_files(self, limit: int = 10) -> List[Dict]:
        """最近のファイル取得"""
        return self.history[:limit]

class BackupManager:
    """自動バックアップ管理"""
    
    def __init__(self, backup_dir: str = "/content/nkat_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, file_path: str, backup_type: str = "auto") -> Optional[str]:
        """バックアップ作成"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return None
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{source_path.stem}_{backup_type}_{timestamp}.gguf"
            backup_path = self.backup_dir / backup_filename
            
            print(f"💾 バックアップ作成中: {backup_filename}")
            shutil.copy2(source_path, backup_path)
            
            # メタデータ保存
            metadata = {
                'original_path': str(source_path),
                'backup_time': datetime.now().isoformat(),
                'backup_type': backup_type,
                'file_size': source_path.stat().st_size
            }
            
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"✅ バックアップ完了: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            print(f"❌ バックアップエラー: {e}")
            return None
    
    def list_backups(self) -> List[Dict]:
        """バックアップ一覧取得"""
        backups = []
        for backup_file in self.backup_dir.glob("*.gguf"):
            metadata_file = backup_file.with_suffix('.json')
            metadata = {}
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            backups.append({
                'path': str(backup_file),
                'filename': backup_file.name,
                'size_gb': backup_file.stat().st_size / (1024**3),
                'metadata': metadata
            })
        
        return sorted(backups, key=lambda x: x['metadata'].get('backup_time', ''), reverse=True)
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """古いバックアップクリーンアップ"""
        backups = self.list_backups()
        if len(backups) > keep_count:
            for backup in backups[keep_count:]:
                try:
                    Path(backup['path']).unlink()
                    metadata_path = Path(backup['path']).with_suffix('.json')
                    if metadata_path.exists():
                        metadata_path.unlink()
                    print(f"🗑️ 古いバックアップ削除: {backup['filename']}")
                except Exception as e:
                    print(f"⚠️ バックアップ削除エラー: {e}")

class AdvancedNKATGUI:
    """高機能NKAT-GGUF GUI"""
    
    def __init__(self):
        self.converter = None
        self.config = NKATConfig()
        self.downloader = HuggingFaceDownloader()
        self.file_history = FileHistory()
        self.backup_manager = BackupManager()
        self.drive_mounted = False
        self.selected_file_path = None
        
        self._create_advanced_interface()
    
    def _create_advanced_interface(self):
        """高機能インターフェース作成"""
        # スタイル定義
        display.display(HTML("""
        <style>
        .nkat-container { background: #f8f9fa; border-radius: 10px; padding: 20px; margin: 10px 0; }
        .nkat-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; padding: 20px; text-align: center; margin-bottom: 20px; }
        .nkat-section { background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 10px 0; }
        .nkat-status-success { background: #d4edda; border: 1px solid #c3e6cb; color: #155724; border-radius: 5px; padding: 10px; }
        .nkat-status-error { background: #f8d7da; border: 1px solid #f5c6cb; color: #721c24; border-radius: 5px; padding: 10px; }
        .nkat-status-warning { background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; border-radius: 5px; padding: 10px; }
        .nkat-drag-drop { border: 2px dashed #007bff; background: #f8f9fa; padding: 30px; text-align: center; border-radius: 10px; cursor: pointer; transition: all 0.3s; }
        .nkat-drag-drop:hover { background: #e3f2fd; border-color: #0056b3; }
        .nkat-drag-active { background: #e8f5e8; border-color: #28a745; }
        </style>
        """))
        
        # メインタイトル
        self.title_html = HTML("""
        <div class="nkat-header">
            <h1>🎨 高機能NKAT-GGUF変換システム</h1>
            <p>非可換コルモゴロフアーノルド表現理論による高度なGGUF最適化</p>
            <p>🚀 DND対応 | 📚 履歴管理 | 💾 自動バックアップ | 🤗 HF連携</p>
        </div>
        """)
        
        # タブ構成
        self._create_main_tabs()
        
        # ドラッグ&ドロップエリア
        self._create_drag_drop_area()
        
        # ステータス表示
        self.status_display = widgets.HTML(value="<div class='nkat-status-warning'>待機中 - ファイルを選択してください</div>")
        
        # インターフェース表示
        self._display_advanced_interface()
    
    def _create_main_tabs(self):
        """メインタブ作成"""
        # 1. ファイル選択タブ
        self.file_tab = self._create_file_selection_tab()
        
        # 2. 設定タブ
        self.config_tab = self._create_advanced_config_tab()
        
        # 3. 履歴タブ
        self.history_tab = self._create_history_tab()
        
        # 4. バックアップタブ
        self.backup_tab = self._create_backup_tab()
        
        # タブウィジェット
        self.main_tabs = widgets.Tab(children=[
            self.file_tab,
            self.config_tab,
            self.history_tab,
            self.backup_tab
        ])
        
        self.main_tabs.set_title(0, "📁 ファイル選択")
        self.main_tabs.set_title(1, "⚙️ 詳細設定")
        self.main_tabs.set_title(2, "📚 履歴管理")
        self.main_tabs.set_title(3, "💾 バックアップ")
    
    def _create_file_selection_tab(self):
        """ファイル選択タブ作成"""
        # Google Drive連携
        self.drive_button = widgets.Button(
            description='📁 Google Drive接続',
            button_style='info',
            layout=widgets.Layout(width='200px', height='40px')
        )
        self.drive_button.on_click(self._mount_drive)
        self.drive_status = widgets.HTML(value="⚠️ Google Drive未接続")
        
        # Hugging Face URL入力（改良版）
        self.hf_url_input = widgets.Textarea(
            value='',
            placeholder='🤗 Hugging Face URL入力:\n• https://huggingface.co/username/model-name\n• username/model-name\n• 複数URL対応（改行区切り）',
            description='HF URL:',
            layout=widgets.Layout(width='100%', height='100px'),
            style={'description_width': 'initial'}
        )
        
        self.hf_download_button = widgets.Button(
            description='📥 一括ダウンロード',
            button_style='primary',
            layout=widgets.Layout(width='200px', height='40px'),
            disabled=True
        )
        self.hf_download_button.on_click(self._batch_download_from_hf)
        
        self.hf_status = widgets.HTML(value="🤗 Hugging Face URLを入力してください")
        
        # ファイルアップロード
        self.file_upload = widgets.FileUpload(
            accept='.gguf',
            multiple=True,
            description='複数ファイル選択'
        )
        
        # 最近使用したファイル
        self.recent_files_dropdown = widgets.Dropdown(
            options=[('ファイルを選択...', '')],
            description='最近使用:',
            layout=widgets.Layout(width='100%')
        )
        self.recent_files_dropdown.observe(self._on_recent_file_selected, names='value')
        
        # ファイル情報表示
        self.file_info_html = widgets.HTML(value="")
        
        return widgets.VBox([
            widgets.HTML("<h3>🌐 Google Drive連携</h3>"),
            widgets.HBox([self.drive_button, self.drive_status]),
            
            widgets.HTML("<h3>🤗 Hugging Face ダウンロード</h3>"),
            self.hf_url_input,
            widgets.HBox([self.hf_download_button, self.hf_status]),
            
            widgets.HTML("<h3>📁 ファイルアップロード</h3>"),
            self.file_upload,
            
            widgets.HTML("<h3>🕒 最近使用したファイル</h3>"),
            self.recent_files_dropdown,
            
            widgets.HTML("<h3>📊 選択ファイル情報</h3>"),
            self.file_info_html
        ])
    
    def _create_drag_drop_area(self):
        """ドラッグ&ドロップエリア作成"""
        self.drag_drop_area = widgets.HTML(value="""
        <div class="nkat-drag-drop" id="drag-drop-area">
            <h3>📋 ドラッグ&ドロップエリア</h3>
            <p>🎯 GGUFファイルをここにドラッグ&ドロップ</p>
            <p>または</p>
            <p>📁 クリックしてファイルを選択</p>
            <small>複数ファイル対応・自動バックアップ作成</small>
        </div>
        """)
        
        # JavaScript for drag and drop
        display.display(Javascript("""
        function setupDragDrop() {
            const dropArea = document.getElementById('drag-drop-area');
            if (!dropArea) return;
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight(e) {
                dropArea.classList.add('nkat-drag-active');
            }
            
            function unhighlight(e) {
                dropArea.classList.remove('nkat-drag-active');
            }
            
            dropArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                // ファイル情報を表示
                for (let file of files) {
                    if (file.name.endsWith('.gguf')) {
                        console.log('GGUF file dropped:', file.name);
                        // Python側での処理をトリガー
                    }
                }
            }
        }
        
        // DOM読み込み後に実行
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', setupDragDrop);
        } else {
            setupDragDrop();
        }
        """))
    
    def _create_advanced_config_tab(self):
        """詳細設定タブ作成"""
        # NKAT理論設定
        nkat_section = widgets.VBox([
            widgets.HTML("<h4>🧠 NKAT理論設定</h4>"),
            widgets.Checkbox(value=self.config.enable_ka_operators, description='Kolmogorov-Arnold演算子有効'),
            widgets.IntSlider(value=self.config.ka_grid_size, min=4, max=16, step=2, description='グリッドサイズ'),
            widgets.IntSlider(value=self.config.lie_algebra_dim, min=2, max=8, description='リー代数次元'),
            widgets.FloatSlider(value=self.config.noncommutative_strength, min=0.01, max=1.0, step=0.01, description='非可換強度')
        ])
        
        # パフォーマンス設定
        performance_section = widgets.VBox([
            widgets.HTML("<h4>⚡ パフォーマンス設定</h4>"),
            widgets.Checkbox(value=self.config.use_64bit_precision, description='64bit精度有効'),
            widgets.Checkbox(value=self.config.enable_cuda_optimization, description='CUDA最適化'),
            widgets.FloatSlider(value=self.config.max_memory_gb, min=1.0, max=15.0, step=0.5, description='最大メモリ(GB)'),
            widgets.IntSlider(value=self.config.chunk_size_mb, min=128, max=2048, step=128, description='チャンクサイズ(MB)')
        ])
        
        # バックアップ設定
        backup_section = widgets.VBox([
            widgets.HTML("<h4>💾 バックアップ設定</h4>"),
            widgets.Checkbox(value=True, description='変換前自動バックアップ'),
            widgets.Checkbox(value=True, description='Google Driveバックアップ'),
            widgets.IntSlider(value=10, min=1, max=50, description='バックアップ保持数'),
            widgets.Checkbox(value=self.config.enable_checkpoint, description='チェックポイント有効')
        ])
        
        # 設定プリセット
        preset_section = widgets.VBox([
            widgets.HTML("<h4>🎯 設定プリセット</h4>"),
            widgets.Dropdown(
                options=[
                    ('デフォルト', 'default'),
                    ('高速処理', 'fast'),
                    ('高品質', 'quality'),
                    ('省メモリ', 'memory_efficient'),
                    ('RTX3080最適化', 'rtx3080')
                ],
                description='プリセット:'
            ),
            widgets.Button(description='プリセット適用', button_style='info')
        ])
        
        return widgets.VBox([nkat_section, performance_section, backup_section, preset_section])
    
    def _create_history_tab(self):
        """履歴タブ作成"""
        # 履歴更新ボタン
        self.refresh_history_button = widgets.Button(
            description='🔄 履歴更新',
            button_style='info'
        )
        self.refresh_history_button.on_click(self._refresh_history)
        
        # 履歴表示
        self.history_output = widgets.Output()
        
        # 履歴クリアボタン
        self.clear_history_button = widgets.Button(
            description='🗑️ 履歴クリア',
            button_style='warning'
        )
        self.clear_history_button.on_click(self._clear_history)
        
        return widgets.VBox([
            widgets.HBox([self.refresh_history_button, self.clear_history_button]),
            self.history_output
        ])
    
    def _create_backup_tab(self):
        """バックアップタブ作成"""
        # バックアップ更新ボタン
        self.refresh_backup_button = widgets.Button(
            description='🔄 バックアップ更新',
            button_style='info'
        )
        self.refresh_backup_button.on_click(self._refresh_backups)
        
        # バックアップ表示
        self.backup_output = widgets.Output()
        
        # バックアップクリーンアップボタン
        self.cleanup_backup_button = widgets.Button(
            description='🧹 古いバックアップ削除',
            button_style='warning'
        )
        self.cleanup_backup_button.on_click(self._cleanup_backups)
        
        return widgets.VBox([
            widgets.HBox([self.refresh_backup_button, self.cleanup_backup_button]),
            self.backup_output
        ])
    
    def _display_advanced_interface(self):
        """高機能インターフェース表示"""
        # 実行ボタン
        self.batch_convert_button = widgets.Button(
            description='🚀 一括NKAT変換実行',
            button_style='success',
            layout=widgets.Layout(width='300px', height='50px'),
            disabled=True
        )
        self.batch_convert_button.on_click(self._start_batch_conversion)
        
        # 進捗表示
        self.progress = widgets.IntProgress(
            value=0, min=0, max=100,
            description='進捗:',
            layout=widgets.Layout(width='100%')
        )
        
        # ログ出力
        self.log_output = widgets.Output()
        
        # メインレイアウト
        main_layout = widgets.VBox([
            self.title_html,
            self.drag_drop_area,
            self.main_tabs,
            self.status_display,
            self.batch_convert_button,
            self.progress,
            widgets.HTML("<h3>📋 変換ログ</h3>"),
            self.log_output
        ])
        
        display.display(main_layout)
        
        # 初期履歴読み込み
        self._refresh_history()
        self._refresh_recent_files()
    
    def _refresh_recent_files(self):
        """最近使用したファイル更新"""
        recent_files = self.file_history.get_recent_files(10)
        options = [('ファイルを選択...', '')]
        
        for file_info in recent_files:
            display_name = f"{file_info['filename']} ({file_info['size_gb']:.2f}GB) - {file_info['timestamp'][:10]}"
            options.append((display_name, file_info['path']))
        
        self.recent_files_dropdown.options = options
    
    def _mount_drive(self, b):
        """Google Drive接続"""
        with self.log_output:
            try:
                if not self.drive_mounted:
                    print("📁 Google Driveに接続中...")
                    drive.mount('/content/drive')
                    self.drive_mounted = True
                    self.drive_status.value = "✅ Google Drive接続済み"
                    self.drive_button.description = "✅ Drive接続済み"
                    self.drive_button.button_style = 'success'
                    print("✅ Google Drive接続完了")
                else:
                    print("ℹ️ 既にGoogle Driveに接続済みです")
            except Exception as e:
                print(f"❌ Drive接続エラー: {e}")
                self.drive_status.value = f"❌ 接続エラー: {e}"
    
    def _batch_download_from_hf(self, b):
        """一括Hugging Faceダウンロード"""
        urls = [url.strip() for url in self.hf_url_input.value.split('\n') if url.strip()]
        
        if not urls:
            self.hf_status.value = "❌ URLが入力されていません"
            return
        
        self.hf_download_button.disabled = True
        downloaded_files = []
        
        with self.log_output:
            print(f"🤗 一括ダウンロード開始: {len(urls)}個のURL")
            
            for i, url in enumerate(urls):
                print(f"\n[{i+1}/{len(urls)}] {url}")
                
                repo_id, filename = self.downloader.parse_hf_url(url)
                if not repo_id:
                    print(f"❌ 無効なURL: {url}")
                    continue
                
                try:
                    downloaded_path = self.downloader.download_gguf(repo_id, filename)
                    if downloaded_path:
                        downloaded_files.append(downloaded_path)
                        
                        # 履歴に追加
                        file_info = {
                            'filename': Path(downloaded_path).name,
                            'path': downloaded_path,
                            'size_gb': Path(downloaded_path).stat().st_size / (1024**3),
                            'source': 'hf_download',
                            'hf_repo': repo_id
                        }
                        self.file_history.add_entry(file_info)
                        
                        print(f"✅ ダウンロード完了: {Path(downloaded_path).name}")
                except Exception as e:
                    print(f"❌ ダウンロードエラー: {e}")
            
            print(f"\n🎉 一括ダウンロード完了: {len(downloaded_files)}ファイル")
            
            if downloaded_files:
                self.batch_convert_button.disabled = False
                self.hf_status.value = f"✅ {len(downloaded_files)}ファイルダウンロード完了"
                self._refresh_recent_files()
            else:
                self.hf_status.value = "❌ ダウンロードに失敗しました"
        
        self.hf_download_button.disabled = False
    
    def _on_recent_file_selected(self, change):
        """最近使用したファイル選択時"""
        if change['new']:
            self.selected_file_path = change['new']
            self._update_file_info(change['new'])
            self.batch_convert_button.disabled = False
    
    def _update_file_info(self, file_path: str):
        """ファイル情報更新"""
        try:
            path = Path(file_path)
            if path.exists():
                size_gb = path.stat().st_size / (1024**3)
                self.file_info_html.value = f"""
                <div class="nkat-status-success">
                    <strong>📁 選択ファイル:</strong> {path.name}<br>
                    <strong>📊 サイズ:</strong> {size_gb:.2f}GB<br>
                    <strong>📍 パス:</strong> {file_path}
                </div>
                """
                self.status_display.value = "<div class='nkat-status-success'>ファイル選択完了 - 変換準備完了</div>"
            else:
                self.file_info_html.value = "<div class='nkat-status-error'>ファイルが見つかりません</div>"
                self.status_display.value = "<div class='nkat-status-error'>ファイルエラー</div>"
        except Exception as e:
            self.file_info_html.value = f"<div class='nkat-status-error'>エラー: {e}</div>"
    
    def _start_batch_conversion(self, b):
        """一括変換開始"""
        # TODO: 一括変換処理の実装
        with self.log_output:
            print("🚀 一括変換機能は開発中です")
    
    def _refresh_history(self, b=None):
        """履歴更新"""
        with self.history_output:
            clear_output(wait=True)
            recent_files = self.file_history.get_recent_files(20)
            
            if not recent_files:
                print("📝 履歴はありません")
                return
            
            print("📚 ファイル履歴 (最新20件)")
            print("=" * 60)
            
            for i, file_info in enumerate(recent_files, 1):
                status_icon = "🤗" if file_info['source'] == 'hf_download' else "📁"
                print(f"{i:2d}. {status_icon} {file_info['filename']}")
                print(f"     サイズ: {file_info['size_gb']:.2f}GB | {file_info['timestamp'][:16]}")
                if file_info.get('hf_repo'):
                    print(f"     HFリポジトリ: {file_info['hf_repo']}")
                print()
    
    def _clear_history(self, b):
        """履歴クリア"""
        self.file_history.history = []
        self.file_history._save_history()
        self._refresh_history()
        self._refresh_recent_files()
    
    def _refresh_backups(self, b=None):
        """バックアップ更新"""
        with self.backup_output:
            clear_output(wait=True)
            backups = self.backup_manager.list_backups()
            
            if not backups:
                print("💾 バックアップはありません")
                return
            
            print("💾 バックアップ一覧")
            print("=" * 60)
            
            for i, backup in enumerate(backups, 1):
                metadata = backup['metadata']
                print(f"{i:2d}. {backup['filename']}")
                print(f"     サイズ: {backup['size_gb']:.2f}GB")
                print(f"     作成日時: {metadata.get('backup_time', 'N/A')[:16]}")
                print(f"     種類: {metadata.get('backup_type', 'unknown')}")
                print()
    
    def _cleanup_backups(self, b):
        """バックアップクリーンアップ"""
        with self.log_output:
            print("🧹 古いバックアップをクリーンアップ中...")
            self.backup_manager.cleanup_old_backups(keep_count=10)
            self._refresh_backups()
            print("✅ クリーンアップ完了")

def main():
    """メイン関数"""
    print("🎨 高機能NKAT-GGUF変換システムを起動中...")
    gui = AdvancedNKATGUI()
    print("✅ システム起動完了！")

if __name__ == "__main__":
    main() 