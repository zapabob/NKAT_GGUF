#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""バックアップファイル削除スクリプト"""

import shutil
from pathlib import Path

def cleanup_integrity_backups():
    """古いバックアップファイルを削除"""
    integrity_dir = Path('integrity_backups')
    if not integrity_dir.exists():
        print("integrity_backupsディレクトリが見つかりません")
        return
    
    backup_files = list(integrity_dir.glob('*.gguf'))
    backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"バックアップファイル: {len(backup_files)}個")
    
    # 最新1つ以外削除
    deleted_size = 0
    for file_path in backup_files[1:]:
        size = file_path.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f'削除: {file_path.name} ({size_mb:.1f}MB)')
        file_path.unlink()
        deleted_size += size
    
    print(f'完了: {len(backup_files[1:])}ファイル削除')
    print(f'削減容量: {deleted_size/(1024*1024*1024):.2f}GB')

if __name__ == "__main__":
    cleanup_integrity_backups() 