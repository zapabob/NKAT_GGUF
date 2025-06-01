#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗂️ NKAT-GGUF リポジトリ整理システム
リポジトリの構造を最適化し、不要ファイルを削除します
"""

import os
import shutil
import json
import hashlib
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

class RepositoryOrganizer:
    """リポジトリ整理システム"""
    
    def __init__(self, repo_path="."):
        self.repo_path = Path(repo_path)
        self.backup_root = self.repo_path / "emergency_backups"
        self.backup_root.mkdir(exist_ok=True)
        
        # 整理統計
        self.stats = {
            "deleted_files": 0,
            "moved_files": 0,
            "duplicates_removed": 0,
            "space_saved_mb": 0,
            "errors": []
        }
        
    def create_backup(self):
        """整理前のバックアップ作成"""
        print("📦 整理前バックアップ作成中...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.backup_root / f"pre_organization_{timestamp}"
        
        try:
            # 重要ファイルのみバックアップ
            important_files = [
                "README.md", "PROJECT_SUMMARY.md", "requirements.txt",
                "file_history.json", ".gitignore"
            ]
            
            backup_dir.mkdir(exist_ok=True)
            for file_name in important_files:
                file_path = self.repo_path / file_name
                if file_path.exists():
                    shutil.copy2(file_path, backup_dir)
                    
            print(f"✅ バックアップ作成完了: {backup_dir}")
            return backup_dir
        except Exception as e:
            print(f"❌ バックアップ作成エラー: {e}")
            return None
    
    def analyze_duplicates(self):
        """重複ファイル検出"""
        print("🔍 重複ファイル検出中...")
        file_hashes = {}
        duplicates = []
        
        # 除外するディレクトリ
        exclude_dirs = {".git", ".specstory", "__pycache__", "emergency_backups"}
        
        for file_path in self.repo_path.rglob("*"):
            if (file_path.is_file() and 
                not any(excluded in str(file_path) for excluded in exclude_dirs)):
                
                try:
                    # ファイルハッシュ計算
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if file_hash in file_hashes:
                        duplicates.append((file_path, file_hashes[file_hash]))
                    else:
                        file_hashes[file_hash] = file_path
                        
                except Exception as e:
                    self.stats["errors"].append(f"ハッシュ計算エラー {file_path}: {e}")
        
        return duplicates
    
    def remove_temporary_files(self):
        """一時ファイル・キャッシュファイル削除"""
        print("🧹 一時ファイル削除中...")
        
        temp_patterns = [
            "*.tmp", "*.temp", "*.pyc", "*.log", "*.bak",
            "Thumbs.db", ".DS_Store", "*.swp", "*.swo"
        ]
        
        deleted_count = 0
        deleted_size = 0
        
        for pattern in temp_patterns:
            for file_path in self.repo_path.rglob(pattern):
                try:
                    if file_path.is_file():
                        size = file_path.stat().st_size
                        file_path.unlink()
                        deleted_count += 1
                        deleted_size += size
                        
                except Exception as e:
                    self.stats["errors"].append(f"削除エラー {file_path}: {e}")
        
        self.stats["deleted_files"] += deleted_count
        self.stats["space_saved_mb"] += deleted_size / (1024 * 1024)
        
        print(f"🗑️ 一時ファイル削除: {deleted_count}個, {deleted_size/(1024*1024):.2f}MB")
    
    def organize_models_directory(self):
        """modelsディレクトリの整理"""
        print("🤖 モデルディレクトリ整理中...")
        
        models_dir = self.repo_path / "models"
        if not models_dir.exists():
            return
            
        # サブディレクトリ作成
        test_models = models_dir / "test"
        demo_models = models_dir / "demo"
        integrated_models = models_dir / "integrated"
        
        for subdir in [test_models, demo_models, integrated_models]:
            subdir.mkdir(exist_ok=True)
        
        # ファイル分類と移動
        for file_path in models_dir.iterdir():
            if file_path.is_file() and file_path.suffix == ".gguf":
                name = file_path.name.lower()
                
                try:
                    if "test" in name:
                        target = test_models / file_path.name
                    elif "demo" in name:
                        target = demo_models / file_path.name
                    elif "integrated" in name:
                        target = integrated_models / file_path.name
                    else:
                        continue  # ルートに残す
                    
                    if not target.exists():
                        shutil.move(str(file_path), str(target))
                        self.stats["moved_files"] += 1
                        
                except Exception as e:
                    self.stats["errors"].append(f"移動エラー {file_path}: {e}")
    
    def clean_integrity_backups(self):
        """整合性バックアップの重複削除"""
        print("💾 バックアップファイル整理中...")
        
        integrity_dir = self.repo_path / "integrity_backups"
        if not integrity_dir.exists():
            return
        
        # 日付でソートして最新3つ以外削除
        backup_files = list(integrity_dir.glob("*.gguf"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # 古いバックアップを削除（最新3つを保持）
        for file_path in backup_files[3:]:
            try:
                size = file_path.stat().st_size
                file_path.unlink()
                self.stats["deleted_files"] += 1
                self.stats["space_saved_mb"] += size / (1024 * 1024)
                print(f"🗑️ 古いバックアップ削除: {file_path.name}")
                
            except Exception as e:
                self.stats["errors"].append(f"バックアップ削除エラー {file_path}: {e}")
    
    def update_gitignore(self):
        """gitignoreファイルの更新"""
        print("📝 .gitignore更新中...")
        
        gitignore_path = self.repo_path / ".gitignore"
        
        additional_ignores = [
            "# Python キャッシュ",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "",
            "# 一時ファイル",
            "*.tmp",
            "*.temp",
            "*.log",
            "*.bak",
            "",
            "# OS固有",
            "Thumbs.db",
            ".DS_Store",
            "",
            "# IDE",
            ".vscode/",
            ".idea/",
            "",
            "# 大容量モデルファイル（オプション）",
            "# *.gguf",
            "",
            "# バックアップディレクトリ",
            "emergency_backups/",
            "integrity_backups/*.gguf"
        ]
        
        try:
            if gitignore_path.exists():
                with open(gitignore_path, 'r', encoding='utf-8') as f:
                    current_content = f.read()
            else:
                current_content = ""
            
            # 新しいエントリを追加
            new_entries = []
            for entry in additional_ignores:
                if entry not in current_content:
                    new_entries.append(entry)
            
            if new_entries:
                with open(gitignore_path, 'a', encoding='utf-8') as f:
                    f.write("\n" + "\n".join(new_entries))
                print("✅ .gitignore更新完了")
                
        except Exception as e:
            self.stats["errors"].append(f".gitignore更新エラー: {e}")
    
    def create_organization_report(self):
        """整理レポート作成"""
        print("📊 整理レポート作成中...")
        
        report = {
            "organization_date": datetime.now().isoformat(),
            "statistics": self.stats,
            "directory_structure": self.get_directory_structure(),
            "recommendations": [
                "定期的な一時ファイル削除を実行してください",
                "大きなモデルファイルはGit LFSの使用を検討してください",
                "開発中は emergency_backups/ を確認してください"
            ]
        }
        
        report_path = self.repo_path / "ORGANIZATION_REPORT.md"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# 🗂️ リポジトリ整理レポート\n\n")
                f.write(f"**整理日時**: {report['organization_date']}\n\n")
                
                f.write("## 📊 整理統計\n\n")
                f.write(f"- 削除ファイル数: {self.stats['deleted_files']}\n")
                f.write(f"- 移動ファイル数: {self.stats['moved_files']}\n")
                f.write(f"- 重複削除数: {self.stats['duplicates_removed']}\n")
                f.write(f"- 節約容量: {self.stats['space_saved_mb']:.2f}MB\n")
                
                if self.stats["errors"]:
                    f.write("\n## ⚠️ エラー\n\n")
                    for error in self.stats["errors"]:
                        f.write(f"- {error}\n")
                
                f.write("\n## 📁 整理後ディレクトリ構造\n\n")
                f.write("```\n")
                f.write(self.get_directory_tree())
                f.write("```\n")
                
            print(f"✅ レポート作成完了: {report_path}")
            
        except Exception as e:
            print(f"❌ レポート作成エラー: {e}")
    
    def get_directory_structure(self):
        """ディレクトリ構造取得"""
        structure = {}
        exclude_dirs = {".git", ".specstory", "__pycache__", "emergency_backups"}
        
        for item in self.repo_path.iterdir():
            if item.is_dir() and item.name not in exclude_dirs:
                file_count = len(list(item.rglob("*")))
                structure[item.name] = file_count
        
        return structure
    
    def get_directory_tree(self):
        """ディレクトリツリー文字列生成"""
        tree_lines = ["NKAT_GGUF/"]
        exclude_dirs = {".git", ".specstory", "__pycache__", "emergency_backups"}
        
        dirs = [d for d in self.repo_path.iterdir() 
               if d.is_dir() and d.name not in exclude_dirs]
        dirs.sort()
        
        for i, dir_path in enumerate(dirs):
            is_last = i == len(dirs) - 1
            prefix = "└── " if is_last else "├── "
            file_count = len([f for f in dir_path.rglob("*") if f.is_file()])
            tree_lines.append(f"{prefix}{dir_path.name}/ ({file_count} files)")
        
        # ルートファイル
        root_files = [f for f in self.repo_path.iterdir() if f.is_file()]
        if root_files:
            tree_lines.append("├── [root files]")
            for file_path in sorted(root_files):
                tree_lines.append(f"│   ├── {file_path.name}")
        
        return "\n".join(tree_lines)
    
    def run_full_organization(self):
        """フル整理実行"""
        print("🚀 NKAT-GGUF リポジトリ整理開始")
        print("=" * 50)
        
        # 1. バックアップ作成
        backup_dir = self.create_backup()
        if not backup_dir:
            print("❌ バックアップ作成失敗。整理を中止します。")
            return False
        
        try:
            # 2. 重複ファイル分析
            duplicates = self.analyze_duplicates()
            if duplicates:
                print(f"⚠️ 重複ファイル検出: {len(duplicates)}組")
                for dup_pair in duplicates:
                    print(f"  - {dup_pair[0]} ←→ {dup_pair[1]}")
            
            # 3. 一時ファイル削除
            self.remove_temporary_files()
            
            # 4. モデルディレクトリ整理
            self.organize_models_directory()
            
            # 5. バックアップファイル整理
            self.clean_integrity_backups()
            
            # 6. gitignore更新
            self.update_gitignore()
            
            # 7. レポート作成
            self.create_organization_report()
            
            print("\n" + "=" * 50)
            print("✅ リポジトリ整理完了!")
            print(f"📊 削除: {self.stats['deleted_files']}ファイル")
            print(f"📁 移動: {self.stats['moved_files']}ファイル")
            print(f"💾 節約: {self.stats['space_saved_mb']:.2f}MB")
            
            if self.stats["errors"]:
                print(f"⚠️ エラー: {len(self.stats['errors'])}件")
                print("詳細は ORGANIZATION_REPORT.md を確認してください")
            
            return True
            
        except Exception as e:
            print(f"❌ 整理中にエラーが発生: {e}")
            print(f"📦 バックアップから復旧可能: {backup_dir}")
            return False

def main():
    """メイン実行"""
    print("🗂️ NKAT-GGUF リポジトリ整理システム")
    
    organizer = RepositoryOrganizer()
    
    # 確認プロンプト
    response = input("\n整理を実行しますか？ (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("整理をキャンセルしました。")
        return
    
    # 整理実行
    success = organizer.run_full_organization()
    
    if success:
        print("\n🎉 整理が正常に完了しました！")
        print("📝 詳細レポート: ORGANIZATION_REPORT.md")
    else:
        print("\n❌ 整理中にエラーが発生しました。")
        print("📦 emergency_backups/ から復旧してください。")

if __name__ == "__main__":
    main() 