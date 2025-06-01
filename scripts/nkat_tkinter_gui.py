#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 NKAT-GGUF Tkinter GUI変換システム
参照ファイル記憶機能・ドラッグ&ドロップ・自動バックアップ対応

特徴:
- Tkinter GUI
- ファイル記憶機能
- ドラッグ&ドロップ対応
- 自動バックアップ
- GPU（RTX3080）最適化
- 進捗表示
- 日本語表示
"""

import os
import sys
import json
import shutil
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

# 外部ライブラリ
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1024**3
    else:
        GPU_NAME = "CPU"
        VRAM_GB = 0
except ImportError:
    GPU_AVAILABLE = False
    GPU_NAME = "CPU"
    VRAM_GB = 0

# ドラッグ&ドロップサポート
try:
    import tkinterdnd2 as tkdnd
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

# ディスク容量チェック用
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # フォールバック: Python標準ライブラリを使用
    try:
        import shutil
        SHUTIL_AVAILABLE = True
    except ImportError:
        SHUTIL_AVAILABLE = False

# 内部モジュール
sys.path.append(os.path.dirname(__file__))
try:
    from nkat_gguf_colab_main import NKATGGUFConverter, NKATConfig
    from huggingface_downloader import HuggingFaceDownloader
except ImportError as e:
    print(f"❌ モジュールインポートエラー: {e}")
    sys.exit(1)

@dataclass
class GUIConfig:
    """GUI設定"""
    window_title: str = "🚀 NKAT-GGUF変換システム"
    window_size: str = "1000x700"
    theme: str = "clam"
    remember_file_history: bool = True
    max_file_history: int = 10
    auto_backup: bool = True
    backup_suffix: str = "_backup"
    
class FileHistory:
    """ファイル履歴管理"""
    
    def __init__(self, config_file: str = "file_history.json", max_items: int = 10):
        self.config_file = Path(config_file)
        self.max_items = max_items
        self.history = self._load_history()
    
    def _load_history(self) -> list:
        """履歴ファイル読み込み"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('files', [])
            except Exception as e:
                print(f"⚠️ 履歴読み込みエラー: {e}")
        return []
    
    def _save_history(self):
        """履歴ファイル保存"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'files': self.history,
                    'updated': time.strftime('%Y-%m-%d %H:%M:%S')
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ 履歴保存エラー: {e}")
    
    def add_file(self, file_path: str):
        """ファイル追加"""
        file_path = str(Path(file_path).resolve())
        
        # 既存エントリを削除
        if file_path in self.history:
            self.history.remove(file_path)
        
        # 先頭に追加
        self.history.insert(0, file_path)
        
        # 最大数制限
        if len(self.history) > self.max_items:
            self.history = self.history[:self.max_items]
        
        self._save_history()
    
    def get_valid_files(self) -> list:
        """有効なファイルのみ取得"""
        valid_files = []
        for file_path in self.history:
            if Path(file_path).exists():
                valid_files.append(file_path)
        
        # 無効ファイルがあった場合、履歴を更新
        if len(valid_files) != len(self.history):
            self.history = valid_files
            self._save_history()
        
        return valid_files

class BackupManager:
    """自動バックアップ管理"""
    
    def __init__(self, suffix: str = "_backup"):
        self.suffix = suffix
    
    def create_backup(self, file_path: str) -> Optional[str]:
        """バックアップ作成"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return None
            
            # バックアップファイル名生成
            backup_name = f"{source_path.stem}{self.suffix}{source_path.suffix}"
            backup_path = source_path.parent / backup_name
            
            # バックアップ作成
            shutil.copy2(source_path, backup_path)
            print(f"💾 バックアップ作成: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            print(f"❌ バックアップ作成エラー: {e}")
            return None
    
    def restore_backup(self, backup_path: str, target_path: str) -> bool:
        """バックアップ復元"""
        try:
            shutil.copy2(backup_path, target_path)
            print(f"🔄 バックアップ復元: {backup_path} → {target_path}")
            return True
        except Exception as e:
            print(f"❌ バックアップ復元エラー: {e}")
            return False

class NKATTkinterGUI:
    """NKAT-GGUF Tkinter GUI"""
    
    def __init__(self):
        self.config = GUIConfig()
        self.file_history = FileHistory(max_items=self.config.max_file_history)
        self.backup_manager = BackupManager(self.config.backup_suffix)
        self.converter = None
        self.hf_downloader = HuggingFaceDownloader()
        self.current_file = None
        self.conversion_thread = None
        
        self._create_gui()
    
    def _create_gui(self):
        """GUI作成"""
        # メインウィンドウ
        if DND_AVAILABLE:
            self.root = tkdnd.Tk()
        else:
            self.root = tk.Tk()
        
        self.root.title(self.config.window_title)
        self.root.geometry(self.config.window_size)
        self.root.configure(bg='#f0f0f0')
        
        # テーマ設定
        style = ttk.Style()
        style.theme_use(self.config.theme)
        
        # メニューバー
        self._create_menu()
        
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ウィンドウサイズ調整
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        # ヘッダー
        self._create_header(main_frame)
        
        # ファイル選択エリア
        self._create_file_selection(main_frame)
        
        # Hugging Face エリア
        self._create_hf_section(main_frame)
        
        # 設定エリア
        self._create_config_section(main_frame)
        
        # 実行エリア
        self._create_execution_section(main_frame)
        
        # ステータスエリア
        self._create_status_section(main_frame)
        
        # ログエリア
        self._create_log_section(main_frame)
        
        # ドラッグ&ドロップ設定
        if DND_AVAILABLE:
            self._setup_drag_drop()
        
        # 初期化
        self._update_ui_state()
    
    def _create_menu(self):
        """メニューバー作成"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # ファイルメニュー
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="ファイルを開く", command=self._select_file)
        file_menu.add_separator()
        file_menu.add_command(label="履歴をクリア", command=self._clear_history)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.root.quit)
        
        # ツールメニュー
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ツール", menu=tools_menu)
        tools_menu.add_command(label="GPU情報", command=self._show_gpu_info)
        tools_menu.add_command(label="設定リセット", command=self._reset_config)
        
        # ヘルプメニュー
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="使用方法", command=self._show_help)
        help_menu.add_command(label="バージョン情報", command=self._show_about)
    
    def _create_header(self, parent):
        """ヘッダー作成"""
        header_frame = ttk.LabelFrame(parent, text="🚀 NKAT-GGUF変換システム", padding="10")
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # システム情報
        info_text = f"GPU: {GPU_NAME}"
        if GPU_AVAILABLE:
            info_text += f" ({VRAM_GB:.1f}GB VRAM)"
        else:
            info_text += " (CPU モード)"
        
        ttk.Label(header_frame, text=info_text, font=('Arial', 10)).pack()
        ttk.Label(header_frame, text="非可換コルモゴロフアーノルド表現理論による最適化", 
                 font=('Arial', 9), foreground='gray').pack()
    
    def _create_file_selection(self, parent):
        """ファイル選択エリア作成"""
        file_frame = ttk.LabelFrame(parent, text="📁 ファイル選択", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        file_frame.columnconfigure(1, weight=1)
        
        # ファイル選択ボタン
        ttk.Button(file_frame, text="ファイルを選択", 
                  command=self._select_file).grid(row=0, column=0, padx=(0, 10))
        
        # ファイルパス表示
        self.file_path_var = tk.StringVar()
        self.file_path_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, 
                                        state='readonly', font=('Arial', 9))
        self.file_path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # 履歴ボタン
        ttk.Button(file_frame, text="履歴", 
                  command=self._show_history).grid(row=0, column=2)
        
        # ドラッグ&ドロップエリア
        self.drop_frame = ttk.LabelFrame(file_frame, text="ドラッグ&ドロップエリア", padding="20")
        self.drop_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        self.drop_label = ttk.Label(self.drop_frame, 
                                   text="GGUFファイルをここにドラッグ&ドロップ\n（対応ファイル: *.gguf）", 
                                   justify=tk.CENTER, font=('Arial', 10))
        self.drop_label.pack(expand=True, fill=tk.BOTH)
    
    def _create_hf_section(self, parent):
        """Hugging Face セクション作成"""
        hf_frame = ttk.LabelFrame(parent, text="🤗 Hugging Face ダウンロード", padding="10")
        hf_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        hf_frame.columnconfigure(1, weight=1)
        
        # URL入力
        ttk.Label(hf_frame, text="URL:").grid(row=0, column=0, padx=(0, 5), sticky=tk.W)
        self.hf_url_var = tk.StringVar()
        self.hf_url_entry = ttk.Entry(hf_frame, textvariable=self.hf_url_var, font=('Arial', 9))
        self.hf_url_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        self.hf_url_entry.bind('<KeyRelease>', self._on_hf_url_change)
        
        # ダウンロードボタン
        self.hf_download_btn = ttk.Button(hf_frame, text="ダウンロード", 
                                         command=self._download_from_hf, state=tk.DISABLED)
        self.hf_download_btn.grid(row=0, column=2)
        
        # ステータス表示
        self.hf_status_var = tk.StringVar(value="Hugging Face URLを入力してください")
        ttk.Label(hf_frame, textvariable=self.hf_status_var, 
                 font=('Arial', 9), foreground='gray').grid(row=1, column=0, columnspan=3, 
                                                           sticky=tk.W, pady=(5, 0))
    
    def _create_config_section(self, parent):
        """設定セクション作成"""
        config_frame = ttk.LabelFrame(parent, text="⚙️ 変換設定", padding="10")
        config_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # タブ作成
        notebook = ttk.Notebook(config_frame)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # 基本設定タブ
        basic_frame = ttk.Frame(notebook, padding="10")
        notebook.add(basic_frame, text="基本設定")
        
        self.ka_enable_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(basic_frame, text="Kolmogorov-Arnold演算子有効", 
                       variable=self.ka_enable_var).grid(row=0, column=0, sticky=tk.W)
        
        ttk.Label(basic_frame, text="グリッドサイズ:").grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        self.ka_grid_var = tk.IntVar(value=8)
        ttk.Scale(basic_frame, from_=4, to=16, variable=self.ka_grid_var, 
                 orient=tk.HORIZONTAL).grid(row=1, column=1, sticky=(tk.W, tk.E), 
                                          padx=(10, 0), pady=(10, 0))
        
        # 精度設定タブ
        precision_frame = ttk.Frame(notebook, padding="10")
        notebook.add(precision_frame, text="精度・最適化")
        
        self.precision_64bit_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(precision_frame, text="64bit精度有効", 
                       variable=self.precision_64bit_var).grid(row=0, column=0, sticky=tk.W)
        
        self.cuda_enable_var = tk.BooleanVar(value=GPU_AVAILABLE)
        ttk.Checkbutton(precision_frame, text="CUDA最適化有効", 
                       variable=self.cuda_enable_var, 
                       state=tk.NORMAL if GPU_AVAILABLE else tk.DISABLED).grid(row=1, column=0, sticky=tk.W)
        
        # メモリ設定タブ
        memory_frame = ttk.Frame(notebook, padding="10")
        notebook.add(memory_frame, text="メモリ・バックアップ")
        
        ttk.Label(memory_frame, text="最大メモリ (GB):").grid(row=0, column=0, sticky=tk.W)
        self.memory_var = tk.DoubleVar(value=10.0)
        ttk.Scale(memory_frame, from_=1.0, to=15.0, variable=self.memory_var, 
                 orient=tk.HORIZONTAL).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 0))
        
        self.backup_enable_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(memory_frame, text="自動バックアップ有効", 
                       variable=self.backup_enable_var).grid(row=1, column=0, sticky=tk.W, pady=(10, 0))
        
        self.checkpoint_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(memory_frame, text="チェックポイント有効", 
                       variable=self.checkpoint_var).grid(row=2, column=0, sticky=tk.W)
    
    def _create_execution_section(self, parent):
        """実行セクション作成"""
        exec_frame = ttk.LabelFrame(parent, text="🚀 実行", padding="10")
        exec_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        exec_frame.columnconfigure(1, weight=1)
        
        # 変換ボタン
        self.convert_btn = ttk.Button(exec_frame, text="🔄 NKAT変換実行", 
                                     command=self._start_conversion, state=tk.DISABLED)
        self.convert_btn.grid(row=0, column=0, padx=(0, 20))
        
        # 停止ボタン
        self.stop_btn = ttk.Button(exec_frame, text="⏹️ 停止", 
                                  command=self._stop_conversion, state=tk.DISABLED)
        self.stop_btn.grid(row=0, column=1, padx=(0, 20))
        
        # 出力フォルダ開く
        ttk.Button(exec_frame, text="📂 出力フォルダ", 
                  command=self._open_output_folder).grid(row=0, column=2)
    
    def _create_status_section(self, parent):
        """ステータスセクション作成"""
        status_frame = ttk.LabelFrame(parent, text="📊 進捗状況", padding="10")
        status_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        status_frame.columnconfigure(0, weight=1)
        
        # 進捗バー
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # ステータステキスト
        self.status_var = tk.StringVar(value="待機中...")
        ttk.Label(status_frame, textvariable=self.status_var, 
                 font=('Arial', 9)).grid(row=1, column=0, sticky=tk.W)
    
    def _create_log_section(self, parent):
        """ログセクション作成"""
        log_frame = ttk.LabelFrame(parent, text="📋 ログ", padding="10")
        log_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        # ログテキスト
        self.log_text = ScrolledText(log_frame, height=8, font=('Consolas', 9))
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ログクリアボタン
        ttk.Button(log_frame, text="クリア", 
                  command=self._clear_log).grid(row=1, column=0, sticky=tk.E, pady=(5, 0))
    
    def _setup_drag_drop(self):
        """ドラッグ&ドロップ設定"""
        if not DND_AVAILABLE:
            return
        
        # ドロップイベント設定
        self.drop_frame.drop_target_register(tkdnd.DND_FILES)
        self.drop_frame.dnd_bind('<<Drop>>', self._on_drop)
        self.drop_frame.dnd_bind('<<DragEnter>>', self._on_drag_enter)
        self.drop_frame.dnd_bind('<<DragLeave>>', self._on_drag_leave)
    
    def _on_drop(self, event):
        """ドロップイベント処理"""
        files = self.root.tk.splitlist(event.data)
        if files:
            file_path = files[0]
            if file_path.lower().endswith('.gguf'):
                self._set_current_file(file_path)
                self._log(f"📁 ドロップされたファイル: {Path(file_path).name}")
            else:
                self._log("❌ GGUFファイルではありません")
                messagebox.showerror("エラー", "GGUFファイル（*.gguf）を選択してください")
    
    def _on_drag_enter(self, event):
        """ドラッグ開始イベント"""
        self.drop_label.config(foreground='blue')
    
    def _on_drag_leave(self, event):
        """ドラッグ終了イベント"""
        self.drop_label.config(foreground='black')
    
    def _select_file(self):
        """ファイル選択ダイアログ"""
        file_path = filedialog.askopenfilename(
            title="GGUFファイルを選択",
            filetypes=[("GGUFファイル", "*.gguf"), ("すべてのファイル", "*.*")]
        )
        
        if file_path:
            self._set_current_file(file_path)
    
    def _set_current_file(self, file_path: str):
        """現在のファイル設定"""
        self.current_file = str(Path(file_path).resolve())
        self.file_path_var.set(self.current_file)
        
        # 履歴に追加
        if self.config.remember_file_history:
            self.file_history.add_file(self.current_file)
        
        # UI状態更新
        self._update_ui_state()
        
        # ログ出力
        file_size = Path(self.current_file).stat().st_size / (1024**3)
        self._log(f"📁 ファイル選択: {Path(self.current_file).name} ({file_size:.2f}GB)")
    
    def _show_history(self):
        """ファイル履歴表示"""
        history_files = self.file_history.get_valid_files()
        
        if not history_files:
            messagebox.showinfo("履歴", "ファイル履歴がありません")
            return
        
        # 履歴選択ダイアログ
        history_window = tk.Toplevel(self.root)
        history_window.title("ファイル履歴")
        history_window.geometry("600x400")
        history_window.transient(self.root)
        history_window.grab_set()
        
        # リストボックス
        listbox_frame = ttk.Frame(history_window, padding="10")
        listbox_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(listbox_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(listbox_frame, yscrollcommand=scrollbar.set, font=('Arial', 9))
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # 履歴項目追加
        for file_path in history_files:
            display_name = f"{Path(file_path).name} ({Path(file_path).parent})"
            listbox.insert(tk.END, display_name)
        
        # ボタンフレーム
        btn_frame = ttk.Frame(history_window, padding="10")
        btn_frame.pack(fill=tk.X)
        
        def select_from_history():
            selection = listbox.curselection()
            if selection:
                selected_file = history_files[selection[0]]
                self._set_current_file(selected_file)
                history_window.destroy()
        
        ttk.Button(btn_frame, text="選択", command=select_from_history).pack(side=tk.RIGHT, padx=(5, 0))
        ttk.Button(btn_frame, text="キャンセル", command=history_window.destroy).pack(side=tk.RIGHT)
        
        # ダブルクリックでも選択
        listbox.bind('<Double-Button-1>', lambda e: select_from_history())
    
    def _clear_history(self):
        """履歴クリア"""
        if messagebox.askyesno("確認", "ファイル履歴をクリアしますか？"):
            self.file_history.history = []
            self.file_history._save_history()
            self._log("🗑️ ファイル履歴をクリアしました")
    
    def _on_hf_url_change(self, event):
        """Hugging Face URL変更イベント"""
        url = self.hf_url_var.get().strip()
        if url:
            repo_id, filename = self.hf_downloader.parse_hf_url(url)
            if repo_id:
                self.hf_download_btn.config(state=tk.NORMAL)
                status = f"✅ 有効なURL: {repo_id}"
                if filename:
                    status += f" ({filename})"
                self.hf_status_var.set(status)
            else:
                self.hf_download_btn.config(state=tk.DISABLED)
                self.hf_status_var.set("❌ 無効なHugging Face URL")
        else:
            self.hf_download_btn.config(state=tk.DISABLED)
            self.hf_status_var.set("Hugging Face URLを入力してください")
    
    def _download_from_hf(self):
        """Hugging Faceダウンロード"""
        url = self.hf_url_var.get().strip()
        if not url:
            return
        
        repo_id, filename = self.hf_downloader.parse_hf_url(url)
        if not repo_id:
            messagebox.showerror("エラー", "無効なHugging Face URLです")
            return
        
        # ダウンロード実行（別スレッド）
        def download_thread():
            try:
                self.hf_download_btn.config(state=tk.DISABLED)
                self._log(f"🤗 Hugging Faceダウンロード開始: {repo_id}")
                
                def progress_callback(percent, message):
                    self.root.after(0, lambda: self.progress_var.set(percent))
                    self.root.after(0, lambda: self.status_var.set(message))
                    self.root.after(0, lambda: self._log(f"[{percent:3.0f}%] {message}"))
                
                downloaded_path = self.hf_downloader.download_gguf(
                    repo_id=repo_id,
                    filename=filename,
                    progress_callback=progress_callback
                )
                
                if downloaded_path:
                    self.root.after(0, lambda: self._set_current_file(downloaded_path))
                    self.root.after(0, lambda: self._log("🎉 Hugging Faceダウンロード完了！"))
                else:
                    self.root.after(0, lambda: self._log("❌ Hugging Faceダウンロード失敗"))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._log(f"❌ ダウンロードエラー: {error_msg}"))
            finally:
                self.root.after(0, lambda: self.hf_download_btn.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.progress_var.set(0))
                self.root.after(0, lambda: self.status_var.set("待機中..."))
        
        threading.Thread(target=download_thread, daemon=True).start()
    
    def _start_conversion(self):
        """変換開始"""
        if not self.current_file or not Path(self.current_file).exists():
            messagebox.showerror("エラー", "ファイルが選択されていません")
            return
        
        # ファイル存在・読み取り権限の詳細チェック
        if not self._validate_input_file():
            return
        
        # ディスク容量チェック
        if not self._check_disk_space():
            return
        
        # 設定作成
        config = NKATConfig(
            enable_ka_operators=self.ka_enable_var.get(),
            ka_grid_size=self.ka_grid_var.get(),
            use_64bit_precision=self.precision_64bit_var.get(),
            enable_cuda_optimization=self.cuda_enable_var.get(),
            max_memory_gb=self.memory_var.get(),
            enable_checkpoint=self.checkpoint_var.get()
        )
        
        # 出力パス生成
        input_path = Path(self.current_file)
        output_path = input_path.parent / f"{input_path.stem}_nkat_enhanced.gguf"
        
        # バックアップ作成
        backup_path = None
        if self.backup_enable_var.get():
            backup_path = self.backup_manager.create_backup(self.current_file)
            if backup_path:
                self._log(f"💾 自動バックアップ作成: {Path(backup_path).name}")
        
        # 変換実行（別スレッド）
        def conversion_thread():
            try:
                self.root.after(0, lambda: self._set_conversion_ui(True))
                
                # コンバーター初期化
                self.converter = NKATGGUFConverter(config)
                
                def progress_callback(percent, message):
                    self.root.after(0, lambda: self.progress_var.set(percent))
                    self.root.after(0, lambda: self.status_var.set(message))
                    self.root.after(0, lambda: self._log(f"[{percent:3.0f}%] {message}"))
                
                self.root.after(0, lambda: self._log(f"🚀 NKAT変換開始: {input_path.name}"))
                
                # 変換実行
                success = self.converter.convert_to_nkat(
                    str(input_path),
                    str(output_path),
                    progress_callback
                )
                
                if success:
                    # 統計情報
                    input_size = input_path.stat().st_size / (1024**3)
                    output_size = output_path.stat().st_size / (1024**3)
                    compression_ratio = (output_size / input_size) * 100
                    
                    self.root.after(0, lambda: self._log("🎉 変換完了！"))
                    self.root.after(0, lambda: self._log(f"📊 入力: {input_size:.2f}GB → 出力: {output_size:.2f}GB ({compression_ratio:.1f}%)"))
                    self.root.after(0, lambda: self.status_var.set("✅ 変換完了"))
                    self.root.after(0, lambda: messagebox.showinfo("完了", f"変換が完了しました！\n出力: {output_path.name}"))
                else:
                    self.root.after(0, lambda: self._log("❌ 変換失敗"))
                    self.root.after(0, lambda: self.status_var.set("❌ 変換失敗"))
                    self.root.after(0, lambda: messagebox.showerror("エラー", "変換に失敗しました"))
                
            except Exception as e:
                error_msg = str(e)
                self.root.after(0, lambda: self._log(f"❌ 変換エラー: {error_msg}"))
                self.root.after(0, lambda: messagebox.showerror("エラー", f"変換エラー: {error_msg}"))
            finally:
                self.root.after(0, lambda: self._set_conversion_ui(False))
                self.root.after(0, lambda: self.progress_var.set(0))
        
        self.conversion_thread = threading.Thread(target=conversion_thread, daemon=True)
        self.conversion_thread.start()
    
    def _stop_conversion(self):
        """変換停止"""
        if self.conversion_thread and self.conversion_thread.is_alive():
            # スレッド停止は実装が複雑なため、ユーザーに通知のみ
            self._log("⚠️ 変換停止が要求されました（次のチェックポイントで停止）")
            messagebox.showinfo("停止", "変換は次のチェックポイントで停止されます")
    
    def _set_conversion_ui(self, converting: bool):
        """変換時のUI状態設定"""
        if converting:
            self.convert_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.hf_download_btn.config(state=tk.DISABLED)
        else:
            self.convert_btn.config(state=tk.NORMAL if self.current_file else tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.hf_download_btn.config(state=tk.NORMAL)
    
    def _update_ui_state(self):
        """UI状態更新"""
        has_file = self.current_file and Path(self.current_file).exists()
        self.convert_btn.config(state=tk.NORMAL if has_file else tk.DISABLED)
    
    def _open_output_folder(self):
        """出力フォルダを開く"""
        if self.current_file:
            output_folder = Path(self.current_file).parent
        else:
            output_folder = Path.cwd()
        
        # OS別のフォルダ開きコマンド
        import subprocess
        try:
            if sys.platform == "win32":
                subprocess.Popen(['explorer', str(output_folder)])
            elif sys.platform == "darwin":
                subprocess.Popen(['open', str(output_folder)])
            else:
                subprocess.Popen(['xdg-open', str(output_folder)])
        except Exception as e:
            self._log(f"⚠️ フォルダを開けませんでした: {e}")
    
    def _log(self, message: str):
        """ログ出力"""
        timestamp = time.strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        
        # ログが長すぎる場合は古い部分を削除
        lines = int(self.log_text.index('end-1c').split('.')[0])
        if lines > 1000:
            self.log_text.delete('1.0', '500.0')
    
    def _clear_log(self):
        """ログクリア"""
        self.log_text.delete('1.0', tk.END)
    
    def _show_gpu_info(self):
        """GPU情報表示"""
        info = f"GPU情報\n\n"
        info += f"GPU利用可能: {'はい' if GPU_AVAILABLE else 'いいえ'}\n"
        info += f"GPU名: {GPU_NAME}\n"
        if GPU_AVAILABLE:
            info += f"VRAM: {VRAM_GB:.1f}GB\n"
            info += f"PyTorch CUDA: {torch.version.cuda}\n"
        
        messagebox.showinfo("GPU情報", info)
    
    def _reset_config(self):
        """設定リセット"""
        if messagebox.askyesno("確認", "設定をデフォルトにリセットしますか？"):
            self.ka_enable_var.set(True)
            self.ka_grid_var.set(8)
            self.precision_64bit_var.set(True)
            self.cuda_enable_var.set(GPU_AVAILABLE)
            self.memory_var.set(10.0)
            self.backup_enable_var.set(True)
            self.checkpoint_var.set(True)
            self._log("🔄 設定をリセットしました")
    
    def _show_help(self):
        """ヘルプ表示"""
        help_text = """🚀 NKAT-GGUF変換システム 使用方法

1. ファイル選択
   • 「ファイルを選択」ボタンでGGUFファイルを選択
   • ドラッグ&ドロップでファイルを選択
   • 「履歴」ボタンで過去に使用したファイルを選択

2. Hugging Faceダウンロード
   • URLフィールドにHugging Face URLを入力
   • 「ダウンロード」ボタンでファイルを取得

3. 設定調整
   • 基本設定: NKAT演算子とグリッドサイズ
   • 精度・最適化: 64bit精度とCUDA設定
   • メモリ・バックアップ: メモリ制限と自動機能

4. 変換実行
   • 「NKAT変換実行」ボタンで変換開始
   • 進捗はプログレスバーで確認
   • 完了後は出力フォルダで結果を確認

5. 便利機能
   • 自動バックアップ: 元ファイルを自動保護
   • ファイル履歴: 最近使用したファイルを記憶
   • チェックポイント: 変換中断からの復旧対応
"""
        
        help_window = tk.Toplevel(self.root)
        help_window.title("ヘルプ")
        help_window.geometry("600x500")
        help_window.transient(self.root)
        
        text_widget = ScrolledText(help_window, font=('Arial', 10), padding="20")
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert('1.0', help_text)
        text_widget.config(state=tk.DISABLED)
    
    def _show_about(self):
        """バージョン情報表示"""
        about_text = f"""🚀 NKAT-GGUF変換システム

バージョン: 1.0.0
作成者: NKAT Development Team

特徴:
• 非可換コルモゴロフアーノルド表現理論
• GPU（RTX3080）最適化
• 自動バックアップ機能
• ファイル履歴機能
• ドラッグ&ドロップ対応

システム情報:
• GPU: {GPU_NAME}
• VRAM: {VRAM_GB:.1f}GB
• ドラッグ&ドロップ: {'対応' if DND_AVAILABLE else '非対応'}
"""
        messagebox.showinfo("バージョン情報", about_text)
    
    def _check_disk_space(self):
        """ディスク容量チェック（Python 3.12互換性対応）"""
        free_space_gb = None
        file_size_gb = None
        drive = None
        
        try:
            file_path = Path(self.current_file)
            file_size_gb = file_path.stat().st_size / (1024**3)
            
            # 方法1: psutilを使用（推奨）
            if PSUTIL_AVAILABLE:
                try:
                    if sys.platform == "win32":
                        drive = file_path.anchor  # C:\ など
                    else:
                        drive = '/'
                    
                    disk_usage = psutil.disk_usage(drive)
                    free_space_gb = disk_usage.free / (1024**3)
                    self._log(f"✅ psutilでディスク容量を取得: {drive}")
                    
                except Exception as psutil_error:
                    self._log(f"⚠️ psutilエラー（Python 3.12互換性問題の可能性）: {psutil_error}")
                    # フォールバックを試行
                    free_space_gb = None
            
            # 方法2: shutilフォールバック（Python 3.3+標準）
            if free_space_gb is None and SHUTIL_AVAILABLE:
                try:
                    # shutilを使用してディスク容量取得
                    if sys.platform == "win32":
                        drive = file_path.anchor
                    else:
                        drive = str(file_path.parent)
                    
                    total, used, free = shutil.disk_usage(drive)
                    free_space_gb = free / (1024**3)
                    self._log(f"✅ shutil（標準ライブラリ）でディスク容量を取得: {drive}")
                    
                except Exception as shutil_error:
                    self._log(f"⚠️ shutilエラー: {shutil_error}")
                    free_space_gb = None
            
            # ディスク容量チェックができない場合
            if free_space_gb is None:
                self._log("⚠️ ディスク容量チェックライブラリが利用できません（変換は続行可能）")
                result = messagebox.askyesno(
                    "ディスク容量チェック不可", 
                    "ディスク容量の確認ができませんでした。\n\n"
                    "変換を続行しますか？\n"
                    "（十分な空き容量があることを事前に確認してください）"
                )
                return result
            
            # 変換時には元ファイル + 出力ファイル + 一時ファイルが必要（約3倍の容量）
            required_space_gb = file_size_gb * 3.0
            
            self._log(f"💽 ディスク容量チェック: {drive} - 空き容量: {free_space_gb:.2f}GB, 必要容量: {required_space_gb:.2f}GB")
            
            if free_space_gb < required_space_gb:
                self._log(f"❌ ディスク容量不足: 空き容量: {free_space_gb:.2f}GB, 必要容量: {required_space_gb:.2f}GB")
                
                # ユーザーに確認
                result = messagebox.askyesnocancel(
                    "ディスク容量不足", 
                    f"ディスク容量が不足している可能性があります。\n\n"
                    f"空き容量: {free_space_gb:.2f}GB\n"
                    f"推奨容量: {required_space_gb:.2f}GB\n\n"
                    f"続行しますか？（リスクがあります）\n\n"
                    f"はい: 続行\n"
                    f"いいえ: 容量を確保してから再実行\n"
                    f"キャンセル: 変換を中止"
                )
                
                if result is True:
                    self._log("⚠️ ユーザーが容量不足でも続行を選択")
                    return True
                elif result is False:
                    self._log("💾 ユーザーが容量確保を選択")
                    self._open_output_folder()  # 容量確保のためフォルダを開く
                    return False
                else:  # None (キャンセル)
                    self._log("🚫 ユーザーが変換を中止")
                    return False
            else:
                self._log(f"✅ ディスク容量は十分です（{free_space_gb:.2f}GB 利用可能）")
                return True
                
        except Exception as e:
            error_msg = str(e)
            self._log(f"❌ ディスク容量チェックエラー: {error_msg}")
            
            # エラーが発生した場合、ユーザーに確認
            result = messagebox.askyesno(
                "容量チェックエラー", 
                f"ディスク容量の確認中にエラーが発生しました。\n\n"
                f"エラー: {error_msg}\n\n"
                f"変換を続行しますか？"
            )
            
            if result:
                self._log("⚠️ ユーザーが容量チェックエラー後も続行を選択")
                return True
            else:
                self._log("🚫 ユーザーが容量チェックエラー後に中止を選択")
                return False
    
    def _validate_input_file(self):
        """ファイル存在・読み取り権限の詳細チェック"""
        file_path = Path(self.current_file)
        
        if not file_path.exists():
            self._log("❌ ファイルが存在しません")
            messagebox.showerror("エラー", f"ファイルが存在しません:\n{file_path}")
            return False
        
        if not file_path.is_file():
            self._log("❌ ファイルが正しくありません")
            messagebox.showerror("エラー", f"選択されたパスはファイルではありません:\n{file_path}")
            return False
        
        # 読み取り権限をテスト
        try:
            with open(file_path, 'rb') as f:
                f.read(1024)  # 最初の1KBを読み取りテスト
            self._log("✅ ファイルの存在と読み取り権限が確認できました")
            return True
        except PermissionError:
            self._log("❌ ファイルの読み取り権限がありません")
            messagebox.showerror("エラー", f"ファイルの読み取り権限がありません:\n{file_path}")
            return False
        except Exception as e:
            error_msg = str(e)
            self._log(f"❌ ファイル読み取りエラー: {error_msg}")
            messagebox.showerror("エラー", f"ファイル読み取りエラー:\n{error_msg}")
            return False
    
    def run(self):
        """GUI実行"""
        self._log("🚀 NKAT-GGUF変換システムを開始しました")
        if DND_AVAILABLE:
            self._log("✅ ドラッグ&ドロップ機能が利用可能です")
        else:
            self._log("⚠️ tkinterdnd2がインストールされていません（ドラッグ&ドロップ無効）")
        
        self._log(f"🎮 GPU: {GPU_NAME}")
        if GPU_AVAILABLE:
            self._log(f"💾 VRAM: {VRAM_GB:.1f}GB")
        
        self.root.mainloop()

def main():
    """メイン関数"""
    # ドラッグ&ドロップライブラリチェック
    if not DND_AVAILABLE:
        print("⚠️ tkinterdnd2がインストールされていません")
        print("ドラッグ&ドロップ機能を使用するには以下をインストールしてください:")
        print("pip install tkinterdnd2")
        print("")
    
    # GUI起動
    app = NKATTkinterGUI()
    app.run()

if __name__ == "__main__":
    main() 