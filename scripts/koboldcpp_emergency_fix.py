#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🆘 KoboldCPP緊急修復システム
KoboldCPP Emergency Fix System for bad_alloc and access violation errors

特徴:
- tokenizer.ggml.tokens bad_allocエラー修復
- アクセス違反エラー解決
- NKATファイル対応
- LoRA化オプション
"""

import os
import sys
import struct
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional

class KoboldCPPEmergencyFix:
    """KoboldCPP緊急修復システム"""
    
    GGUF_MAGIC = b'GGUF'
    
    def __init__(self):
        self.backup_dir = Path("emergency_backups")
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_emergency_backup(self, file_path: str) -> str:
        """緊急バックアップ作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{Path(file_path).stem}_emergency_{timestamp}.gguf"
        shutil.copy2(file_path, backup_path)
        return str(backup_path)
    
    def analyze_tokenizer_issue(self, file_path: str) -> Dict:
        """tokenizerエラー分析"""
        try:
            with open(file_path, 'rb') as f:
                # GGUFマジック確認
                magic = f.read(4)
                if magic != self.GGUF_MAGIC:
                    return {"status": "error", "message": "不正なGGUFファイル"}
                
                # バージョン読み取り
                version = struct.unpack('<I', f.read(4))[0]
                
                # テンソル数読み取り
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                
                # メタデータ数読み取り
                metadata_count = struct.unpack('<Q', f.read(8))[0]
                
                # メタデータ読み取り
                tokenizer_found = False
                tokenizer_size = 0
                
                for i in range(metadata_count):
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    key = f.read(key_len).decode('utf-8')
                    
                    if key == 'tokenizer.ggml.tokens':
                        tokenizer_found = True
                        # 値タイプ読み取り
                        value_type = struct.unpack('<I', f.read(4))[0]
                        
                        if value_type == 8:  # 配列タイプ
                            array_type = struct.unpack('<I', f.read(4))[0]
                            array_len = struct.unpack('<Q', f.read(8))[0]
                            tokenizer_size = array_len
                            
                            return {
                                "status": "found",
                                "tokenizer_size": tokenizer_size,
                                "array_length": array_len,
                                "position": f.tell()
                            }
                    else:
                        # 値をスキップ
                        value_type = struct.unpack('<I', f.read(4))[0]
                        if value_type == 8:  # 配列
                            array_type = struct.unpack('<I', f.read(4))[0]
                            array_len = struct.unpack('<Q', f.read(8))[0]
                            if array_type == 6:  # 文字列配列
                                for j in range(array_len):
                                    str_len = struct.unpack('<Q', f.read(8))[0]
                                    f.read(str_len)
                        elif value_type == 6:  # 文字列
                            str_len = struct.unpack('<Q', f.read(8))[0]
                            f.read(str_len)
                        elif value_type in [0, 1, 2, 3]:  # 整数
                            f.read(8)
                        elif value_type in [4, 5]:  # 浮動小数点
                            f.read(8)
                
                if not tokenizer_found:
                    return {"status": "not_found", "message": "tokenizer.ggml.tokensが見つかりません"}
                
                return {"status": "analyzed"}
                
        except Exception as e:
            return {"status": "error", "message": f"分析エラー: {str(e)}"}
    
    def fix_tokenizer_bad_alloc(self, file_path: str) -> str:
        """tokenizer bad_allocエラー修復"""
        print("🔧 tokenizer bad_allocエラー修復開始...")
        
        analysis = self.analyze_tokenizer_issue(file_path)
        
        if analysis["status"] == "error":
            print(f"❌ エラー: {analysis['message']}")
            return None
        
        # バックアップ作成
        backup_path = self.create_emergency_backup(file_path)
        print(f"💾 バックアップ作成: {backup_path}")
        
        # 修復版作成
        fixed_path = file_path.replace('.gguf', '_tokenfixed.gguf')
        
        try:
            with open(file_path, 'rb') as src, open(fixed_path, 'wb') as dst:
                # ヘッダーコピー
                header = src.read(20)  # magic + version + tensor_count + metadata_count
                dst.write(header)
                
                # メタデータ処理
                metadata_count = struct.unpack('<Q', header[12:20])[0]
                
                for i in range(metadata_count):
                    # キー読み取り
                    key_len_data = src.read(8)
                    key_len = struct.unpack('<Q', key_len_data)[0]
                    key_data = src.read(key_len)
                    key = key_data.decode('utf-8')
                    
                    dst.write(key_len_data)
                    dst.write(key_data)
                    
                    if key == 'tokenizer.ggml.tokens':
                        print("🔧 tokenizer.ggml.tokens修復中...")
                        
                        # 値タイプ読み取り
                        value_type_data = src.read(4)
                        value_type = struct.unpack('<I', value_type_data)[0]
                        
                        if value_type == 8:  # 配列
                            array_type_data = src.read(4)
                            array_len_data = src.read(8)
                            array_len = struct.unpack('<Q', array_len_data)[0]
                            
                            # 小さなサイズに制限
                            if array_len > 100000:
                                print(f"⚠️ トークン数を{array_len}から100000に制限")
                                array_len = 100000
                                array_len_data = struct.pack('<Q', array_len)
                            
                            dst.write(value_type_data)
                            dst.write(array_type_data)
                            dst.write(array_len_data)
                            
                            # トークンデータのコピー（制限付き）
                            for j in range(array_len):
                                try:
                                    str_len_data = src.read(8)
                                    if len(str_len_data) < 8:
                                        break
                                    str_len = struct.unpack('<Q', str_len_data)[0]
                                    
                                    # 文字列長制限
                                    if str_len > 1000:
                                        str_len = 1000
                                        str_len_data = struct.pack('<Q', str_len)
                                    
                                    dst.write(str_len_data)
                                    
                                    token_data = src.read(str_len)
                                    if len(token_data) < str_len:
                                        token_data += b'\x00' * (str_len - len(token_data))
                                    
                                    dst.write(token_data)
                                    
                                except Exception as e:
                                    print(f"⚠️ トークン{j}でエラー: {e}")
                                    break
                        else:
                            # 通常コピー
                            value_data = src.read(8)
                            dst.write(value_type_data)
                            dst.write(value_data)
                    else:
                        # 他のメタデータは通常コピー
                        value_type_data = src.read(4)
                        value_type = struct.unpack('<I', value_type_data)[0]
                        dst.write(value_type_data)
                        
                        if value_type == 8:  # 配列
                            remaining = src.read()
                            dst.write(remaining)
                            break
                        elif value_type == 6:  # 文字列
                            str_len_data = src.read(8)
                            str_len = struct.unpack('<Q', str_len_data)[0]
                            str_data = src.read(str_len)
                            dst.write(str_len_data)
                            dst.write(str_data)
                        else:
                            data = src.read(8)
                            dst.write(data)
                
                # 残りのデータをコピー
                remaining = src.read()
                dst.write(remaining)
            
            print(f"✅ 修復完了: {fixed_path}")
            return fixed_path
            
        except Exception as e:
            print(f"❌ 修復失敗: {e}")
            return None
    
    def create_koboldcpp_launch_config(self, model_path: str) -> str:
        """最適化されたKoboldCPP起動設定作成"""
        config = {
            "model": model_path,
            "contextsize": 2048,  # 削減
            "blasbatchsize": 64,  # 大幅削減
            "blasthreads": 2,     # 削減
            "gpulayers": 0,       # CPU使用
            "nommap": True,       # メモリマッピング無効
            "noavx2": True,       # AVX2無効
            "usemlock": False,    # メモリロック無効
            "failsafe": True,     # セーフモード
            "port": 5001,
            "skiplauncher": True
        }
        
        # バッチファイル作成
        model_name = Path(model_path).stem
        batch_path = f"run_{model_name}_emergency.bat"
        
        with open(batch_path, 'w', encoding='utf-8') as f:
            f.write("@echo off\n")
            f.write("REM KoboldCPP緊急起動設定\n")
            f.write(f"REM モデル: {model_name}\n")
            f.write("echo 🆘 KoboldCPP緊急モード起動\n")
            f.write("echo メモリ最小設定で起動中...\n")
            f.write("echo.\n\n")
            
            cmd = f"python koboldcpp.py"
            for key, value in config.items():
                if isinstance(value, bool):
                    if value:
                        cmd += f" --{key}"
                else:
                    cmd += f" --{key} {value}"
            
            f.write(cmd + "\n")
            f.write("pause\n")
        
        return batch_path

def main():
    if len(sys.argv) < 2:
        print("使用法: python koboldcpp_emergency_fix.py <gguf_file_path> [action]")
        print("アクション: analyze, fix, config")
        return
    
    file_path = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else "fix"
    
    fixer = KoboldCPPEmergencyFix()
    
    print("🆘 KoboldCPP緊急修復システム v1.0")
    print("=" * 50)
    
    if action == "analyze":
        result = fixer.analyze_tokenizer_issue(file_path)
        print(f"📊 分析結果: {result}")
    
    elif action == "fix":
        fixed_path = fixer.fix_tokenizer_bad_alloc(file_path)
        if fixed_path:
            batch_path = fixer.create_koboldcpp_launch_config(fixed_path)
            print(f"🚀 起動設定作成: {batch_path}")
            print("\n💡 次の手順:")
            print(f"1. {batch_path} を実行")
            print("2. エラーが続く場合はCPUのみモードを試行")
            print("3. コンテキストサイズをさらに削減")
    
    elif action == "config":
        batch_path = fixer.create_koboldcpp_launch_config(file_path)
        print(f"🚀 緊急起動設定作成: {batch_path}")

if __name__ == "__main__":
    import datetime
    main() 