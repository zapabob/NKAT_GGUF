#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 NKAT-GGUF GUI拡張機能
- リアルタイムファイル検出
- ドラッグ&ドロップ詳細処理
- プログレス表示拡張
- エラーハンドリング強化
"""

import asyncio
import threading
import traceback
from pathlib import Path
from typing import List, Dict, Any, Callable
import json
import time
from datetime import datetime

# Google Colab環境検出
try:
    from google.colab import files
    import IPython.display as display
    from IPython.display import Javascript, HTML
    import ipywidgets as widgets
    from tqdm.notebook import tqdm
    COLAB_ENV = True
except ImportError:
    from tqdm import tqdm
    COLAB_ENV = False

class FileDropHandler:
    """ファイルドラッグ&ドロップ処理"""
    
    def __init__(self, callback: Callable = None):
        self.callback = callback
        self.dropped_files = []
        self._setup_file_drop()
    
    def _setup_file_drop(self):
        """ファイルドロップJavaScript設定"""
        if not COLAB_ENV:
            return
        
        js_code = """
        window.setupAdvancedFileDrop = function() {
            // グローバル変数でファイル情報を保持
            window.droppedFileInfo = [];
            
            const dropArea = document.getElementById('drag-drop-area');
            if (!dropArea) {
                console.log('Drag drop area not found, will retry...');
                setTimeout(window.setupAdvancedFileDrop, 1000);
                return;
            }
            
            console.log('Setting up advanced file drop...');
            
            // イベントリスナー設定
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
                document.body.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            // ビジュアルフィードバック
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight(e) {
                dropArea.classList.add('nkat-drag-active');
                dropArea.innerHTML = `
                    <h3>🎯 ファイルをドロップしてください</h3>
                    <p>✨ NKAT変換準備完了</p>
                `;
            }
            
            function unhighlight(e) {
                dropArea.classList.remove('nkat-drag-active');
                dropArea.innerHTML = `
                    <h3>📋 ドラッグ&ドロップエリア</h3>
                    <p>🎯 GGUFファイルをここにドラッグ&ドロップ</p>
                    <p>または</p>
                    <p>📁 クリックしてファイルを選択</p>
                    <small>複数ファイル対応・自動バックアップ作成</small>
                `;
            }
            
            // ドロップ処理
            dropArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                console.log('Files dropped:', files.length);
                window.droppedFileInfo = [];
                
                // ファイル情報収集
                Array.from(files).forEach((file, index) => {
                    if (file.name.toLowerCase().endsWith('.gguf')) {
                        const fileInfo = {
                            name: file.name,
                            size: file.size,
                            sizeGB: (file.size / (1024*1024*1024)).toFixed(2),
                            type: file.type,
                            lastModified: new Date(file.lastModified).toISOString(),
                            index: index
                        };
                        
                        window.droppedFileInfo.push(fileInfo);
                        console.log('GGUF file info:', fileInfo);
                    }
                });
                
                // 結果表示
                if (window.droppedFileInfo.length > 0) {
                    dropArea.innerHTML = `
                        <h3>✅ ${window.droppedFileInfo.length}個のGGUFファイル検出</h3>
                        <div style="text-align: left; margin: 10px 0;">
                            ${window.droppedFileInfo.map(f => `
                                <div style="background: #e8f5e8; padding: 5px; margin: 2px; border-radius: 3px;">
                                    📁 ${f.name} (${f.sizeGB}GB)
                                </div>
                            `).join('')}
                        </div>
                        <p>🚀 変換準備完了</p>
                    `;
                    
                    // Python側にファイル情報を通知
                    window.notifyPythonFilesDrop && window.notifyPythonFilesDrop(window.droppedFileInfo);
                } else {
                    dropArea.innerHTML = `
                        <h3>⚠️ GGUFファイルが見つかりません</h3>
                        <p>GGUFファイルをドロップしてください</p>
                    `;
                }
            }
            
            // クリック時のファイル選択
            dropArea.addEventListener('click', () => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.gguf';
                input.multiple = true;
                
                input.onchange = (e) => {
                    const files = e.target.files;
                    if (files.length > 0) {
                        // ドロップ処理と同じ処理を実行
                        const fakeEvent = { dataTransfer: { files: files } };
                        handleDrop(fakeEvent);
                    }
                };
                
                input.click();
            });
            
            console.log('Advanced file drop setup complete');
        };
        
        // 初期化実行
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', window.setupAdvancedFileDrop);
        } else {
            window.setupAdvancedFileDrop();
        }
        """
        
        display.display(Javascript(js_code))
    
    def set_callback(self, callback: Callable):
        """コールバック設定"""
        self.callback = callback
        
        # Python側通知関数設定
        if COLAB_ENV:
            js_notify = f"""
            window.notifyPythonFilesDrop = function(fileInfo) {{
                console.log('Notifying Python about dropped files:', fileInfo);
                // IPython.notebook.kernel.execute を使用してPython側にデータ送信
                const pythonCode = `
dropped_files_info = {json.dumps('__FILE_INFO__')}
if hasattr(window, 'file_drop_handler') and window.file_drop_handler.callback:
    window.file_drop_handler.callback(dropped_files_info)
`.replace('__FILE_INFO__', JSON.stringify(fileInfo));
                
                IPython.notebook.kernel.execute(pythonCode);
            }};
            """
            display.display(Javascript(js_notify))
    
    def get_dropped_files(self) -> List[Dict]:
        """ドロップされたファイル情報取得"""
        return self.dropped_files

class ProgressTracker:
    """プログレス追跡システム"""
    
    def __init__(self):
        self.progress_widget = None
        self.status_widget = None
        self.detail_widget = None
        self.start_time = None
        self.current_step = 0
        self.total_steps = 0
        
    def create_progress_display(self) -> widgets.VBox:
        """プログレス表示ウィジェット作成"""
        self.progress_widget = widgets.IntProgress(
            value=0, min=0, max=100,
            description='進捗:',
            bar_style='info',
            layout=widgets.Layout(width='100%')
        )
        
        self.status_widget = widgets.HTML(
            value="<div style='text-align: center; padding: 10px;'>待機中</div>"
        )
        
        self.detail_widget = widgets.HTML(
            value="<div style='font-family: monospace; font-size: 12px; background: #f8f9fa; padding: 10px; border-radius: 5px;'>詳細情報がここに表示されます</div>"
        )
        
        self.time_widget = widgets.HTML(
            value="<div style='text-align: center; color: #666;'>経過時間: --:--</div>"
        )
        
        return widgets.VBox([
            self.status_widget,
            self.progress_widget,
            self.time_widget,
            self.detail_widget
        ])
    
    def start(self, total_steps: int, description: str = "処理中"):
        """プログレス開始"""
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
        
        if self.progress_widget:
            self.progress_widget.value = 0
            self.progress_widget.max = total_steps
            self.progress_widget.description = description
            self.progress_widget.bar_style = 'info'
        
        if self.status_widget:
            self.status_widget.value = f"<div style='text-align: center; padding: 10px; background: #d1ecf1; border-radius: 5px;'>🚀 {description}を開始</div>"
    
    def update(self, step: int = None, description: str = None, details: str = None):
        """プログレス更新"""
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # 進捗率計算
        progress_percent = int((self.current_step / self.total_steps) * 100) if self.total_steps > 0 else 0
        
        # 経過時間計算
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        elapsed_str = f"{int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
        
        # 予想残り時間
        if self.current_step > 0 and elapsed_time > 0:
            avg_time_per_step = elapsed_time / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = avg_time_per_step * remaining_steps
            eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"
        else:
            eta_str = "--:--"
        
        # ウィジェット更新
        if self.progress_widget:
            self.progress_widget.value = self.current_step
            if description:
                self.progress_widget.description = description
        
        if self.status_widget:
            status_html = f"""
            <div style='text-align: center; padding: 10px; background: #d1ecf1; border-radius: 5px;'>
                📊 ステップ {self.current_step}/{self.total_steps} ({progress_percent}%)
                {f'<br>🔄 {description}' if description else ''}
            </div>
            """
            self.status_widget.value = status_html
        
        if self.time_widget:
            self.time_widget.value = f"<div style='text-align: center; color: #666;'>経過時間: {elapsed_str} | 予想残り: {eta_str}</div>"
        
        if self.detail_widget and details:
            detail_html = f"""
            <div style='font-family: monospace; font-size: 12px; background: #f8f9fa; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto;'>
                <strong>📋 詳細情報:</strong><br>
                {details.replace('\n', '<br>')}
            </div>
            """
            self.detail_widget.value = detail_html
    
    def complete(self, success: bool = True, message: str = "完了"):
        """プログレス完了"""
        if self.progress_widget:
            self.progress_widget.value = self.total_steps
            self.progress_widget.bar_style = 'success' if success else 'danger'
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        elapsed_str = f"{int(elapsed_time // 60):02d}:{int(elapsed_time % 60):02d}"
        
        if self.status_widget:
            icon = "✅" if success else "❌"
            bg_color = "#d4edda" if success else "#f8d7da"
            text_color = "#155724" if success else "#721c24"
            
            status_html = f"""
            <div style='text-align: center; padding: 15px; background: {bg_color}; color: {text_color}; border-radius: 5px; font-weight: bold;'>
                {icon} {message}
            </div>
            """
            self.status_widget.value = status_html
        
        if self.time_widget:
            self.time_widget.value = f"<div style='text-align: center; color: #666;'>総実行時間: {elapsed_str}</div>"

class RealTimeMonitor:
    """リアルタイム監視システム"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        self.monitor_data = {}
    
    def add_callback(self, callback: Callable):
        """監視コールバック追加"""
        self.callbacks.append(callback)
    
    def start_monitoring(self, monitor_paths: List[str] = None):
        """監視開始"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_paths = monitor_paths or ["/content"]
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """監視停止"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
    
    def _monitor_loop(self):
        """監視ループ"""
        last_check = {}
        
        while self.monitoring:
            try:
                current_data = {}
                
                # ファイルシステム監視
                for path in self.monitor_paths:
                    path_obj = Path(path)
                    if path_obj.exists():
                        current_data[path] = {
                            'files': list(path_obj.rglob("*.gguf")),
                            'last_modified': max(
                                [f.stat().st_mtime for f in path_obj.rglob("*") if f.is_file()],
                                default=0
                            )
                        }
                
                # 変更検出
                for path, data in current_data.items():
                    if path not in last_check or last_check[path]['last_modified'] != data['last_modified']:
                        for callback in self.callbacks:
                            try:
                                callback(path, data)
                            except Exception as e:
                                print(f"Monitor callback error: {e}")
                
                last_check = current_data.copy()
                time.sleep(2)  # 2秒間隔で監視
                
            except Exception as e:
                print(f"Monitor loop error: {e}")
                time.sleep(5)

class ErrorHandler:
    """エラーハンドリングシステム"""
    
    def __init__(self):
        self.error_log = []
        self.error_display = None
    
    def create_error_display(self) -> widgets.Output:
        """エラー表示ウィジェット作成"""
        self.error_display = widgets.Output()
        return self.error_display
    
    def handle_error(self, error: Exception, context: str = "", user_message: str = None):
        """エラー処理"""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'user_message': user_message
        }
        
        self.error_log.append(error_info)
        
        # エラー表示
        if self.error_display:
            with self.error_display:
                print(f"\n❌ エラー発生 [{error_info['timestamp'][:19]}]")
                print(f"📍 コンテキスト: {context}")
                if user_message:
                    print(f"💬 メッセージ: {user_message}")
                print(f"🔍 エラー種別: {error_info['error_type']}")
                print(f"📝 詳細: {error_info['error_message']}")
                print("-" * 50)
    
    def get_error_summary(self) -> str:
        """エラーサマリー取得"""
        if not self.error_log:
            return "エラーはありません"
        
        recent_errors = self.error_log[-5:]  # 最新5件
        summary = f"最新エラー {len(recent_errors)}件:\n"
        
        for error in recent_errors:
            summary += f"• [{error['timestamp'][:19]}] {error['error_type']}: {error['context']}\n"
        
        return summary
    
    def clear_errors(self):
        """エラーログクリア"""
        self.error_log.clear()
        if self.error_display:
            self.error_display.clear_output()

# グローバルインスタンス
file_drop_handler = FileDropHandler()
progress_tracker = ProgressTracker()
real_time_monitor = RealTimeMonitor()
error_handler = ErrorHandler()

def setup_gui_extensions():
    """GUI拡張機能セットアップ"""
    print("🔧 GUI拡張機能をセットアップ中...")
    
    # リアルタイム監視開始
    real_time_monitor.start_monitoring(["/content", "/content/drive"])
    
    print("✅ GUI拡張機能セットアップ完了")
    return {
        'file_drop': file_drop_handler,
        'progress': progress_tracker,
        'monitor': real_time_monitor,
        'error_handler': error_handler
    } 