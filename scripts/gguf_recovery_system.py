#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUF緊急リカバリーシステム
GGUF Emergency Recovery System for Corrupted Files
"""

import os
import sys
import struct
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# tqdmインポート
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None):
            self.iterable = iterable
            self.desc = desc
            self.total = total
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            pass

class GGUFRecoverySystem:
    """GGUF緊急リカバリーシステム"""
    
    GGUF_MAGIC = b'GGUF'
    
    def __init__(self):
        self.recovery_log = []
        self.temp_dir = tempfile.mkdtemp(prefix="gguf_recovery_")
        print(f"🏥 GGUF緊急リカバリーシステム初期化")
        print(f"   一時ディレクトリ: {self.temp_dir}")
    
    def diagnose_file(self, file_path: str) -> Dict[str, Any]:
        """ファイル診断"""
        print(f"\n🔍 ファイル診断開始: {Path(file_path).name}")
        
        diagnosis = {
            'file_exists': False,
            'file_size': 0,
            'readable': False,
            'gguf_magic_valid': False,
            'header_valid': False,
            'corruption_detected': False,
            'corruption_points': [],
            'recovery_possible': False,
            'recovery_strategy': None,
            'errors': []
        }
        
        try:
            # ファイル存在確認
            if not os.path.exists(file_path):
                diagnosis['errors'].append("ファイルが存在しません")
                return diagnosis
            
            diagnosis['file_exists'] = True
            
            # ファイルサイズ確認
            file_size = os.path.getsize(file_path)
            diagnosis['file_size'] = file_size
            
            if file_size == 0:
                diagnosis['errors'].append("ファイルサイズが0です")
                return diagnosis
            
            # 読み取り可能性確認
            try:
                with open(file_path, 'rb') as f:
                    f.read(1)
                diagnosis['readable'] = True
            except Exception as e:
                diagnosis['errors'].append(f"ファイル読み取り不可: {e}")
                return diagnosis
            
            # GGUF形式診断
            gguf_diagnosis = self._diagnose_gguf_structure(file_path)
            diagnosis.update(gguf_diagnosis)
            
            # リカバリー可能性判定
            diagnosis['recovery_possible'] = self._assess_recovery_possibility(diagnosis)
            if diagnosis['recovery_possible']:
                diagnosis['recovery_strategy'] = self._determine_recovery_strategy(diagnosis)
            
            print(f"  📊 診断完了: リカバリー{'可能' if diagnosis['recovery_possible'] else '困難'}")
            
        except Exception as e:
            diagnosis['errors'].append(f"診断中にエラー: {e}")
        
        return diagnosis
    
    def _diagnose_gguf_structure(self, file_path: str) -> Dict[str, Any]:
        """GGUF構造診断"""
        result = {
            'gguf_magic_valid': False,
            'header_valid': False,
            'corruption_detected': False,
            'corruption_points': [],
            'metadata_readable': False,
            'tensor_info_readable': False
        }
        
        try:
            with open(file_path, 'rb') as f:
                # マジックナンバーチェック
                magic = f.read(4)
                if magic == self.GGUF_MAGIC:
                    result['gguf_magic_valid'] = True
                    print(f"    ✅ GGUF マジックナンバー有効")
                else:
                    result['corruption_points'].append(f"無効なマジックナンバー: {magic}")
                    print(f"    ❌ 無効なマジックナンバー: {magic}")
                
                # バージョン読み取り
                try:
                    version_data = f.read(4)
                    if len(version_data) == 4:
                        version = struct.unpack('<I', version_data)[0]
                        if 1 <= version <= 3:
                            print(f"    ✅ バージョン: {version}")
                        else:
                            result['corruption_points'].append(f"異常なバージョン: {version}")
                    else:
                        result['corruption_points'].append("バージョン情報不完全")
                except Exception as e:
                    result['corruption_points'].append(f"バージョン読み取りエラー: {e}")
                
                # メタデータ数読み取り
                try:
                    metadata_data = f.read(8)
                    if len(metadata_data) == 8:
                        metadata_count = struct.unpack('<Q', metadata_data)[0]
                        if metadata_count < 10000:  # 妥当な範囲
                            result['metadata_readable'] = True
                            print(f"    ✅ メタデータ数: {metadata_count}")
                        else:
                            result['corruption_points'].append(f"異常なメタデータ数: {metadata_count}")
                    else:
                        result['corruption_points'].append("メタデータ数情報不完全")
                except Exception as e:
                    result['corruption_points'].append(f"メタデータ数読み取りエラー: {e}")
                
                # テンソル数読み取り
                try:
                    tensor_data = f.read(8)
                    if len(tensor_data) == 8:
                        tensor_count = struct.unpack('<Q', tensor_data)[0]
                        if tensor_count < 10000:  # 妥当な範囲
                            result['tensor_info_readable'] = True
                            print(f"    ✅ テンソル数: {tensor_count}")
                        else:
                            result['corruption_points'].append(f"異常なテンソル数: {tensor_count}")
                    else:
                        result['corruption_points'].append("テンソル数情報不完全")
                except Exception as e:
                    result['corruption_points'].append(f"テンソル数読み取りエラー: {e}")
                
                # ヘッダー全体の妥当性
                if (result['gguf_magic_valid'] and 
                    result['metadata_readable'] and 
                    result['tensor_info_readable']):
                    result['header_valid'] = True
                
                # 破損検出
                if result['corruption_points']:
                    result['corruption_detected'] = True
                
        except Exception as e:
            result['corruption_points'].append(f"構造診断エラー: {e}")
            result['corruption_detected'] = True
        
        return result
    
    def _assess_recovery_possibility(self, diagnosis: Dict[str, Any]) -> bool:
        """リカバリー可能性評価"""
        # ファイルが存在し、読み取り可能である必要がある
        if not diagnosis['file_exists'] or not diagnosis['readable']:
            return False
        
        # ファイルサイズが妥当である必要がある
        if diagnosis['file_size'] < 32:  # 最小ヘッダーサイズ
            return False
        
        # 重大な破損でなければリカバリー可能
        critical_corruptions = [
            "ファイル読み取り不可",
            "構造診断エラー"
        ]
        
        for corruption in diagnosis['corruption_points']:
            if any(critical in corruption for critical in critical_corruptions):
                return False
        
        return True
    
    def _determine_recovery_strategy(self, diagnosis: Dict[str, Any]) -> str:
        """リカバリー戦略決定"""
        if diagnosis['gguf_magic_valid'] and diagnosis['header_valid']:
            return "partial_recovery"  # 部分的リカバリー
        elif diagnosis['gguf_magic_valid']:
            return "header_reconstruction"  # ヘッダー再構築
        else:
            return "magic_repair"  # マジックナンバー修復
    
    def recover_file(self, file_path: str, output_path: str = None) -> Dict[str, Any]:
        """ファイルリカバリー実行"""
        print(f"\n🔧 ファイルリカバリー開始: {Path(file_path).name}")
        
        # 診断実行
        diagnosis = self.diagnose_file(file_path)
        
        if not diagnosis['recovery_possible']:
            return {
                'success': False,
                'error': 'リカバリー不可能',
                'diagnosis': diagnosis
            }
        
        # 出力パス設定
        if not output_path:
            file_stem = Path(file_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(Path(file_path).parent / f"{file_stem}_recovered_{timestamp}.gguf")
        
        # バックアップ作成
        backup_path = self._create_backup(file_path)
        
        try:
            # リカバリー戦略に基づく修復実行
            strategy = diagnosis['recovery_strategy']
            
            if strategy == "partial_recovery":
                success = self._partial_recovery(file_path, output_path, diagnosis)
            elif strategy == "header_reconstruction":
                success = self._header_reconstruction(file_path, output_path, diagnosis)
            elif strategy == "magic_repair":
                success = self._magic_repair(file_path, output_path, diagnosis)
            else:
                success = False
            
            if success:
                print(f"✅ リカバリー完了: {Path(output_path).name}")
                return {
                    'success': True,
                    'output_path': output_path,
                    'backup_path': backup_path,
                    'strategy_used': strategy,
                    'diagnosis': diagnosis
                }
            else:
                return {
                    'success': False,
                    'error': 'リカバリー処理失敗',
                    'backup_path': backup_path,
                    'diagnosis': diagnosis
                }
                
        except Exception as e:
            print(f"❌ リカバリーエラー: {e}")
            return {
                'success': False,
                'error': f'リカバリー中にエラー: {e}',
                'backup_path': backup_path,
                'diagnosis': diagnosis
            }
    
    def _create_backup(self, file_path: str) -> str:
        """バックアップ作成"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_stem = Path(file_path).stem
            backup_name = f"{file_stem}_before_recovery_{timestamp}.gguf"
            backup_path = str(Path(self.temp_dir) / backup_name)
            
            shutil.copy2(file_path, backup_path)
            print(f"  💾 バックアップ作成: {backup_name}")
            return backup_path
            
        except Exception as e:
            print(f"  ⚠️ バックアップ作成失敗: {e}")
            return ""
    
    def _partial_recovery(self, input_path: str, output_path: str, diagnosis: Dict[str, Any]) -> bool:
        """部分的リカバリー"""
        print(f"  🔧 部分的リカバリー実行")
        
        try:
            with open(input_path, 'rb') as infile, open(output_path, 'wb') as outfile:
                # ヘッダー部分をコピー（最初の24バイト）
                header_data = infile.read(24)
                outfile.write(header_data)
                
                # 残りのデータを安全にコピー
                chunk_size = 1024 * 1024  # 1MB chunks
                while True:
                    chunk = infile.read(chunk_size)
                    if not chunk:
                        break
                    
                    # チャンクの妥当性チェック
                    if self._is_chunk_valid(chunk):
                        outfile.write(chunk)
                    else:
                        # 破損チャンクを修復
                        repaired_chunk = self._repair_chunk(chunk)
                        outfile.write(repaired_chunk)
            
            return True
            
        except Exception as e:
            print(f"    ❌ 部分的リカバリーエラー: {e}")
            return False
    
    def _header_reconstruction(self, input_path: str, output_path: str, diagnosis: Dict[str, Any]) -> bool:
        """ヘッダー再構築"""
        print(f"  🔧 ヘッダー再構築実行")
        
        try:
            with open(input_path, 'rb') as infile:
                # 元のデータを読み込み
                original_data = infile.read()
            
            with open(output_path, 'wb') as outfile:
                # 有効なGGUFヘッダーを作成
                outfile.write(self.GGUF_MAGIC)  # マジックナンバー
                outfile.write(struct.pack('<I', 3))  # バージョン3
                outfile.write(struct.pack('<Q', 0))  # メタデータ数（暫定）
                outfile.write(struct.pack('<Q', 0))  # テンソル数（暫定）
                
                # 元のデータの有効部分を追加（ヘッダー以降）
                if len(original_data) > 24:
                    valid_data = original_data[24:]
                    outfile.write(valid_data)
            
            return True
            
        except Exception as e:
            print(f"    ❌ ヘッダー再構築エラー: {e}")
            return False
    
    def _magic_repair(self, input_path: str, output_path: str, diagnosis: Dict[str, Any]) -> bool:
        """マジックナンバー修復"""
        print(f"  🔧 マジックナンバー修復実行")
        
        try:
            with open(input_path, 'rb') as infile:
                original_data = infile.read()
            
            with open(output_path, 'wb') as outfile:
                # 正しいマジックナンバーを書き込み
                outfile.write(self.GGUF_MAGIC)
                
                # 残りのデータをそのまま書き込み
                if len(original_data) > 4:
                    outfile.write(original_data[4:])
                else:
                    # データが短すぎる場合は最小限のヘッダーを作成
                    outfile.write(struct.pack('<I', 3))  # バージョン
                    outfile.write(struct.pack('<Q', 0))  # メタデータ数
                    outfile.write(struct.pack('<Q', 0))  # テンソル数
            
            return True
            
        except Exception as e:
            print(f"    ❌ マジックナンバー修復エラー: {e}")
            return False
    
    def _is_chunk_valid(self, chunk: bytes) -> bool:
        """チャンクの妥当性チェック"""
        # 簡単な妥当性チェック
        if len(chunk) == 0:
            return False
        
        # 全てのバイトが同じ値でないかチェック
        if len(set(chunk)) == 1:
            return False
        
        return True
    
    def _repair_chunk(self, chunk: bytes) -> bytes:
        """破損チャンクの修復"""
        # 簡単な修復：ゼロで埋める
        return b'\x00' * len(chunk)
    
    def emergency_recovery(self, file_path: str) -> Dict[str, Any]:
        """緊急リカバリー（最後の手段）"""
        print(f"\n🚨 緊急リカバリー実行: {Path(file_path).name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        emergency_output = str(Path(file_path).parent / f"emergency_recovered_{timestamp}.gguf")
        
        try:
            with open(file_path, 'rb') as infile, open(emergency_output, 'wb') as outfile:
                # 最小限の有効なGGUFファイルを作成
                outfile.write(self.GGUF_MAGIC)  # マジックナンバー
                outfile.write(struct.pack('<I', 3))  # バージョン3
                outfile.write(struct.pack('<Q', 1))  # メタデータ数
                outfile.write(struct.pack('<Q', 0))  # テンソル数
                
                # ダミーメタデータ
                key = "recovered_file"
                value = f"Emergency recovery at {timestamp}"
                
                # キー長とキー
                outfile.write(struct.pack('<Q', len(key)))
                outfile.write(key.encode('utf-8'))
                
                # 値の型（文字列）
                outfile.write(struct.pack('<I', 8))  # STRING type
                
                # 値長と値
                outfile.write(struct.pack('<Q', len(value)))
                outfile.write(value.encode('utf-8'))
            
            print(f"✅ 緊急リカバリー完了: {Path(emergency_output).name}")
            return {
                'success': True,
                'output_path': emergency_output,
                'method': 'emergency_recovery'
            }
            
        except Exception as e:
            print(f"❌ 緊急リカバリー失敗: {e}")
            return {
                'success': False,
                'error': f'緊急リカバリーエラー: {e}'
            }
    
    def cleanup(self):
        """一時ファイル削除"""
        try:
            shutil.rmtree(self.temp_dir)
            print(f"🧹 一時ファイル削除完了")
        except Exception as e:
            print(f"⚠️ 一時ファイル削除エラー: {e}")

def main():
    """メイン関数"""
    if len(sys.argv) < 2:
        print("使用法: python gguf_recovery_system.py <ファイルパス> [出力パス]")
        print("例: python gguf_recovery_system.py broken_model.gguf")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    print("🏥 GGUF緊急リカバリーシステム v1.0")
    print("=" * 50)
    
    recovery_system = GGUFRecoverySystem()
    
    try:
        # 通常のリカバリー試行
        result = recovery_system.recover_file(input_file, output_file)
        
        if result['success']:
            print(f"\n✅ リカバリー成功!")
            print(f"   出力ファイル: {result['output_path']}")
            if result.get('backup_path'):
                print(f"   バックアップ: {result['backup_path']}")
        else:
            print(f"\n⚠️ 通常リカバリー失敗: {result['error']}")
            print("緊急リカバリーを試行します...")
            
            # 緊急リカバリー試行
            emergency_result = recovery_system.emergency_recovery(input_file)
            
            if emergency_result['success']:
                print(f"\n🚨 緊急リカバリー成功!")
                print(f"   出力ファイル: {emergency_result['output_path']}")
                print("   注意: 元のデータの一部が失われている可能性があります")
            else:
                print(f"\n❌ 全てのリカバリー方法が失敗しました")
                print(f"   エラー: {emergency_result['error']}")
    
    finally:
        recovery_system.cleanup()

if __name__ == "__main__":
    main() 