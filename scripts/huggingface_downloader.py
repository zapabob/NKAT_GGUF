"""
Hugging Face Hub からGGUFファイルをダウンロードするためのユーティリティ
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

try:
    from huggingface_hub import HfApi, hf_hub_download, list_repo_files
    from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError
except ImportError:
    print("❌ Hugging Face Hub未インストール")
    sys.exit(1)

class HuggingFaceDownloader:
    """Hugging Face URLからGGUFファイルダウンロード"""
    
    def __init__(self, download_dir: str = "./downloads"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.api = HfApi()
    
    def parse_hf_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Hugging Face URLをパース"""
        # URL形式: https://huggingface.co/username/repo-name
        # または: https://huggingface.co/username/repo-name/blob/main/filename.gguf
        patterns = [
            r'https://huggingface\.co/([^/]+/[^/]+)',
            r'huggingface\.co/([^/]+/[^/]+)',
            r'^([^/]+/[^/]+)$'  # 直接のrepo名
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url.strip())
            if match:
                repo_id = match.group(1)
                # ファイル名を抽出（URLに含まれている場合）
                filename_match = re.search(r'/blob/[^/]+/(.+\.gguf)', url)
                filename = filename_match.group(1) if filename_match else None
                return repo_id, filename
        
        return None, None
    
    def find_gguf_files(self, repo_id: str) -> List[str]:
        """リポジトリ内のGGUFファイルを検索"""
        try:
            files = list_repo_files(repo_id)
            gguf_files = [f for f in files if f.endswith('.gguf')]
            return sorted(gguf_files)
        except Exception as e:
            print(f"❌ リポジトリアクセスエラー: {e}")
            return []
    
    def download_gguf(self, repo_id: str, filename: str = None, progress_callback=None) -> Optional[str]:
        """GGUFファイルをダウンロード"""
        try:
            if progress_callback:
                progress_callback(5, f"🔍 リポジトリ {repo_id} を検索中...")
            
            # ファイル名が指定されていない場合、GGUFファイルを検索
            if not filename:
                gguf_files = self.find_gguf_files(repo_id)
                if not gguf_files:
                    raise ValueError(f"リポジトリ {repo_id} にGGUFファイルが見つかりません")
                
                # 複数ある場合は最初のものを選択（後でUI選択機能を追加可能）
                filename = gguf_files[0]
                if len(gguf_files) > 1:
                    print(f"⚠️ 複数のGGUFファイルが見つかりました。{filename} をダウンロードします")
                    print(f"利用可能ファイル: {', '.join(gguf_files)}")
            
            if progress_callback:
                progress_callback(15, f"📥 {filename} をダウンロード中...")
            
            # ダウンロード実行
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=str(self.download_dir),
                local_dir_use_symlinks=False
            )
            
            if progress_callback:
                progress_callback(80, f"✅ ダウンロード完了: {filename}")
            
            print(f"✅ ダウンロード完了: {downloaded_path}")
            return downloaded_path
            
        except Exception as e:
            error_msg = f"❌ ダウンロードエラー: {e}"
            print(error_msg)
            if progress_callback:
                progress_callback(0, error_msg)
            return None
    
    def get_model_info(self, repo_id: str) -> Dict[str, Any]:
        """モデル情報を取得"""
        try:
            model_info = self.api.model_info(repo_id)
            return {
                'model_name': model_info.modelId,
                'downloads': getattr(model_info, 'downloads', 0),
                'likes': getattr(model_info, 'likes', 0),
                'tags': getattr(model_info, 'tags', []),
                'library_name': getattr(model_info, 'library_name', 'unknown'),
                'pipeline_tag': getattr(model_info, 'pipeline_tag', 'unknown')
            }
        except Exception as e:
            print(f"⚠️ モデル情報取得エラー: {e}")
            return {} 