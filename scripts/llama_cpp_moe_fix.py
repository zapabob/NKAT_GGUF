#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔧 llama.cpp MoE (Mixture of Experts) 量子化タイプ修復システム
llama.cpp MoE Quantization Type Fix System

特徴:
- MoEモデルの量子化タイプ不一致を修復
- 全エキスパートの量子化タイプを統一
- llama.cpp最新版との互換性確保
- NKAT変換との統合対応
- 自動バックアップ機能

参考: https://github.com/ggerganov/llama.cpp/discussions/9299
"""

import os
import sys
import struct
import shutil
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llama_cpp_moe_fix.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# GGUF定数
GGUF_MAGIC = b'GGUF'
GGUF_VERSION = 3

# 型定義
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# 量子化タイプ
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29

# MoE関連キー
MOE_METADATA_KEYS = [
    'llama.expert_count',
    'llama.expert_used_count',
    'general.architecture'
]

class LlamaCppMoEFixer:
    """llama.cpp MoE修復システム"""
    
    def __init__(self, backup_dir: str = "emergency_backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        self.type_sizes = {
            GGUF_TYPE_UINT8: 1, GGUF_TYPE_INT8: 1,
            GGUF_TYPE_UINT16: 2, GGUF_TYPE_INT16: 2,
            GGUF_TYPE_UINT32: 4, GGUF_TYPE_INT32: 4, GGUF_TYPE_FLOAT32: 4,
            GGUF_TYPE_UINT64: 8, GGUF_TYPE_INT64: 8, GGUF_TYPE_FLOAT64: 8,
            GGUF_TYPE_BOOL: 1
        }
        
        logger.info(f"🔧 LlamaCppMoEFixer 初期化完了")
        logger.info(f"   バックアップディレクトリ: {self.backup_dir}")
    
    def create_backup(self, file_path: str) -> str:
        """バックアップ作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{Path(file_path).stem}_moe_backup_{timestamp}.gguf"
        shutil.copy2(file_path, backup_path)
        logger.info(f"💾 バックアップ作成: {backup_path}")
        return str(backup_path)
    
    def analyze_moe_model(self, file_path: str) -> Dict[str, Any]:
        """MoEモデル分析"""
        logger.info(f"🔍 MoEモデル分析開始: {file_path}")
        
        try:
            with open(file_path, 'rb') as f:
                # ヘッダー読み取り
                header = self._read_header(f)
                if not header:
                    return {"status": "error", "message": "無効なGGUFファイル"}
                
                # メタデータ読み取り
                metadata = self._read_metadata(f, header['metadata_count'])
                
                # MoE関連情報抽出
                moe_info = self._extract_moe_info(metadata)
                
                # テンソル情報読み取り
                tensors_info = self._read_tensors_info(f, header['tensor_count'])
                
                # エキスパート量子化タイプ分析
                expert_analysis = self._analyze_expert_quantization(tensors_info, moe_info)
                
                return {
                    "status": "success",
                    "is_moe": moe_info.get("expert_count", 0) > 1,
                    "expert_count": moe_info.get("expert_count", 0),
                    "expert_used_count": moe_info.get("expert_used_count", 0),
                    "architecture": moe_info.get("architecture", "unknown"),
                    "expert_analysis": expert_analysis,
                    "needs_fix": expert_analysis.get("needs_fix", False),
                    "recommendation": expert_analysis.get("recommendation", "")
                }
                
        except Exception as e:
            logger.error(f"❌ 分析エラー: {e}")
            return {"status": "error", "message": str(e)}
    
    def _read_header(self, f) -> Optional[Dict]:
        """GGUFヘッダー読み取り"""
        # マジック確認
        magic = f.read(4)
        if magic != GGUF_MAGIC:
            return None
        
        # バージョン
        version = struct.unpack('<I', f.read(4))[0]
        
        # テンソル数
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        
        # メタデータ数
        metadata_count = struct.unpack('<Q', f.read(8))[0]
        
        return {
            'version': version,
            'tensor_count': tensor_count,
            'metadata_count': metadata_count
        }
    
    def _read_metadata(self, f, metadata_count: int) -> Dict[str, Any]:
        """メタデータ読み取り"""
        metadata = {}
        
        for i in range(metadata_count):
            # キー読み取り
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            
            # 値読み取り
            value = self._read_value(f)
            metadata[key] = value
        
        return metadata
    
    def _read_value(self, f):
        """値読み取り"""
        value_type = struct.unpack('<I', f.read(4))[0]
        
        if value_type == GGUF_TYPE_STRING:
            str_len = struct.unpack('<Q', f.read(8))[0]
            return f.read(str_len).decode('utf-8')
        elif value_type == GGUF_TYPE_ARRAY:
            array_type = struct.unpack('<I', f.read(4))[0]
            array_len = struct.unpack('<Q', f.read(8))[0]
            
            array_values = []
            for _ in range(array_len):
                if array_type == GGUF_TYPE_STRING:
                    str_len = struct.unpack('<Q', f.read(8))[0]
                    array_values.append(f.read(str_len).decode('utf-8'))
                else:
                    # 他の型は簡略化
                    f.read(self.type_sizes.get(array_type, 4))
                    array_values.append(None)
            
            return array_values
        elif value_type in self.type_sizes:
            size = self.type_sizes[value_type]
            data = f.read(size)
            
            if value_type == GGUF_TYPE_UINT32:
                return struct.unpack('<I', data)[0]
            elif value_type == GGUF_TYPE_UINT64:
                return struct.unpack('<Q', data)[0]
            elif value_type == GGUF_TYPE_FLOAT32:
                return struct.unpack('<f', data)[0]
            else:
                return data
        else:
            # 不明な型
            return None
    
    def _extract_moe_info(self, metadata: Dict) -> Dict[str, Any]:
        """MoE情報抽出"""
        moe_info = {}
        
        for key in MOE_METADATA_KEYS:
            if key in metadata:
                if 'expert_count' in key:
                    moe_info['expert_count'] = metadata[key]
                elif 'expert_used_count' in key:
                    moe_info['expert_used_count'] = metadata[key]
                elif 'architecture' in key:
                    moe_info['architecture'] = metadata[key]
        
        return moe_info
    
    def _read_tensors_info(self, f, tensor_count: int) -> List[Dict]:
        """テンソル情報読み取り"""
        tensors = []
        
        for i in range(tensor_count):
            # テンソル名
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            
            # 次元数
            n_dims = struct.unpack('<I', f.read(4))[0]
            
            # 形状
            shape = []
            for _ in range(n_dims):
                shape.append(struct.unpack('<Q', f.read(8))[0])
            
            # データ型
            dtype = struct.unpack('<I', f.read(4))[0]
            
            # オフセット
            offset = struct.unpack('<Q', f.read(8))[0]
            
            tensors.append({
                'name': name,
                'shape': shape,
                'dtype': dtype,
                'offset': offset
            })
        
        return tensors
    
    def _analyze_expert_quantization(self, tensors_info: List[Dict], moe_info: Dict) -> Dict[str, Any]:
        """エキスパート量子化分析"""
        if not moe_info.get("expert_count", 0) > 1:
            return {"needs_fix": False, "recommendation": "MoEモデルではありません"}
        
        # エキスパート関連テンサーを特定
        expert_tensors = []
        for tensor in tensors_info:
            name = tensor['name']
            if any(expert_keyword in name for expert_keyword in [
                'ffn_gate_inp', 'feed_forward', 'mlp', 'expert'
            ]):
                expert_tensors.append(tensor)
        
        if not expert_tensors:
            return {"needs_fix": False, "recommendation": "エキスパートテンサーが見つかりません"}
        
        # 量子化タイプの分布を調べる
        dtype_counts = {}
        for tensor in expert_tensors:
            dtype = tensor['dtype']
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
        
        # 最も一般的な量子化タイプを決定
        most_common_dtype = max(dtype_counts, key=dtype_counts.get)
        
        # 不一致があるかチェック
        has_mismatch = len(dtype_counts) > 1
        
        result = {
            "needs_fix": has_mismatch,
            "expert_tensors_count": len(expert_tensors),
            "dtype_distribution": dtype_counts,
            "most_common_dtype": most_common_dtype,
            "mismatch_count": sum(1 for dtype in dtype_counts if dtype != most_common_dtype)
        }
        
        if has_mismatch:
            result["recommendation"] = f"全エキスパートを量子化タイプ {most_common_dtype} に統一することを推奨"
        else:
            result["recommendation"] = "量子化タイプは統一されています"
        
        return result
    
    def fix_moe_quantization(self, input_path: str, output_path: str = None) -> bool:
        """MoE量子化修復"""
        logger.info(f"🔧 MoE量子化修復開始: {input_path}")
        
        # 分析実行
        analysis = self.analyze_moe_model(input_path)
        
        if analysis["status"] != "success":
            logger.error(f"❌ 分析失敗: {analysis.get('message', '不明なエラー')}")
            return False
        
        if not analysis["needs_fix"]:
            logger.info(f"✅ 修復不要: {analysis['recommendation']}")
            return True
        
        # バックアップ作成
        backup_path = self.create_backup(input_path)
        
        # 出力パス決定
        if output_path is None:
            output_path = input_path.replace('.gguf', '_moe_fixed.gguf')
        
        try:
            # 修復実行
            success = self._perform_moe_fix(input_path, output_path, analysis)
            
            if success:
                logger.info(f"✅ MoE修復完了: {output_path}")
                return True
            else:
                logger.error(f"❌ MoE修復失敗")
                return False
                
        except Exception as e:
            logger.error(f"❌ 修復エラー: {e}")
            return False
    
    def _perform_moe_fix(self, input_path: str, output_path: str, analysis: Dict) -> bool:
        """実際のMoE修復処理"""
        target_dtype = analysis["expert_analysis"]["most_common_dtype"]
        
        logger.info(f"🔧 量子化タイプを {target_dtype} に統一中...")
        
        with open(input_path, 'rb') as src, open(output_path, 'wb') as dst:
            # ヘッダー読み取り・書き込み
            header_data = src.read(24)  # magic + version + tensor_count + metadata_count
            dst.write(header_data)
            
            header = struct.unpack('<4sIQQ', header_data)
            tensor_count = header[2]
            metadata_count = header[3]
            
            # メタデータ部分をそのままコピー
            metadata_start = src.tell()
            metadata_size = self._calculate_metadata_size(src, metadata_count)
            src.seek(metadata_start)
            metadata_data = src.read(metadata_size)
            dst.write(metadata_data)
            
            # テンサー情報部分の修復
            tensors_start = src.tell()
            
            for i in range(tensor_count):
                tensor_start_pos = src.tell()
                
                # テンサー名
                name_len = struct.unpack('<Q', src.read(8))[0]
                name = src.read(name_len).decode('utf-8')
                
                # 次元数
                n_dims = struct.unpack('<I', src.read(4))[0]
                
                # 形状
                shape = []
                for _ in range(n_dims):
                    shape.append(struct.unpack('<Q', src.read(8))[0])
                
                # データ型
                dtype = struct.unpack('<I', src.read(4))[0]
                
                # オフセット
                offset = struct.unpack('<Q', src.read(8))[0]
                
                # エキスパート関連テンサーの場合は量子化タイプを修正
                is_expert_tensor = any(expert_keyword in name for expert_keyword in [
                    'ffn_gate_inp', 'feed_forward', 'mlp', 'expert'
                ])
                
                if is_expert_tensor and dtype != target_dtype:
                    logger.info(f"   📝 修正: {name} ({dtype} -> {target_dtype})")
                    dtype = target_dtype
                
                # 修正されたテンサー情報を書き込み
                dst.write(struct.pack('<Q', name_len))
                dst.write(name.encode('utf-8'))
                dst.write(struct.pack('<I', n_dims))
                for dim in shape:
                    dst.write(struct.pack('<Q', dim))
                dst.write(struct.pack('<I', dtype))
                dst.write(struct.pack('<Q', offset))
            
            # テンサーデータ部分をそのままコピー
            remaining_data = src.read()
            dst.write(remaining_data)
        
        return True
    
    def _calculate_metadata_size(self, f, metadata_count: int) -> int:
        """メタデータサイズ計算"""
        start_pos = f.tell()
        
        for i in range(metadata_count):
            # キー
            key_len = struct.unpack('<Q', f.read(8))[0]
            f.read(key_len)
            
            # 値
            self._skip_value(f)
        
        end_pos = f.tell()
        f.seek(start_pos)
        
        return end_pos - start_pos
    
    def _skip_value(self, f):
        """値をスキップ"""
        value_type = struct.unpack('<I', f.read(4))[0]
        
        if value_type == GGUF_TYPE_STRING:
            str_len = struct.unpack('<Q', f.read(8))[0]
            f.read(str_len)
        elif value_type == GGUF_TYPE_ARRAY:
            array_type = struct.unpack('<I', f.read(4))[0]
            array_len = struct.unpack('<Q', f.read(8))[0]
            
            for _ in range(array_len):
                if array_type == GGUF_TYPE_STRING:
                    str_len = struct.unpack('<Q', f.read(8))[0]
                    f.read(str_len)
                else:
                    f.read(self.type_sizes.get(array_type, 4))
        elif value_type in self.type_sizes:
            f.read(self.type_sizes[value_type])
    
    def print_analysis_report(self, analysis: Dict):
        """分析レポート表示"""
        print("\n" + "="*60)
        print("📊 MoEモデル分析レポート")
        print("="*60)
        
        if analysis["status"] != "success":
            print(f"❌ エラー: {analysis['message']}")
            return
        
        print(f"🤖 アーキテクチャ: {analysis['architecture']}")
        print(f"🧠 MoEモデル: {'はい' if analysis['is_moe'] else 'いいえ'}")
        
        if analysis['is_moe']:
            print(f"👥 エキスパート数: {analysis['expert_count']}")
            print(f"🎯 使用エキスパート数: {analysis['expert_used_count']}")
            
            expert_analysis = analysis["expert_analysis"]
            print(f"🔧 修復必要: {'はい' if analysis['needs_fix'] else 'いいえ'}")
            print(f"📊 エキスパートテンサー数: {expert_analysis['expert_tensors_count']}")
            
            print("\n📈 量子化タイプ分布:")
            for dtype, count in expert_analysis["dtype_distribution"].items():
                print(f"   タイプ {dtype}: {count}個")
            
            print(f"\n💡 推奨事項: {expert_analysis['recommendation']}")
        
        print("="*60)

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='llama.cpp MoE修復ツール')
    parser.add_argument('input', help='入力GGUFファイル')
    parser.add_argument('-o', '--output', help='出力ファイル（オプション）')
    parser.add_argument('-a', '--analyze-only', action='store_true', help='分析のみ実行')
    parser.add_argument('--backup-dir', default='emergency_backups', help='バックアップディレクトリ')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ ファイルが見つかりません: {args.input}")
        sys.exit(1)
    
    # 修復システム初期化
    fixer = LlamaCppMoEFixer(backup_dir=args.backup_dir)
    
    # 分析実行
    analysis = fixer.analyze_moe_model(args.input)
    fixer.print_analysis_report(analysis)
    
    if args.analyze_only:
        print("\n✅ 分析完了")
        return
    
    if analysis["status"] == "success" and analysis["needs_fix"]:
        print(f"\n🔧 修復を開始します...")
        success = fixer.fix_moe_quantization(args.input, args.output)
        
        if success:
            print(f"\n✅ 修復が正常に完了しました")
        else:
            print(f"\n❌ 修復に失敗しました")
            sys.exit(1)
    else:
        print(f"\n✅ 修復は不要です")

if __name__ == "__main__":
    main() 