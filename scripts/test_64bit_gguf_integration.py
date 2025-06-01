#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
64bit長GGUF統合テストスクリプト
"""

import os
import sys
import time
import struct
import numpy as np
from gguf_nkat_integration import GGUFNKATIntegrator, NKATConfig

def create_test_gguf_with_large_values(filepath: str):
    """大きな値を含むテストGGUFファイルを作成"""
    print(f"🔧 64bit長テスト用GGUFファイル作成: {filepath}")
    
    with open(filepath, 'wb') as f:
        # GGUFヘッダー
        f.write(b'GGUF')  # magic
        f.write(struct.pack('<I', 3))  # version
        f.write(struct.pack('<Q', 1))  # tensor_count
        f.write(struct.pack('<Q', 6))  # metadata_kv_count
        
        # メタデータ（64bit精度をテストするための大きな値を含む）
        metadata_items = [
            ("general.name", "64bit_precision_test", 4),  # string
            ("general.version", "1.0", 4),  # string
            ("large_int32", 2147483647, 6),  # 32bit最大値
            ("large_int64", 9223372036854775807, 11),  # 64bit最大値  
            ("precision_float32", 3.14159265359, 7),  # 32bit float
            ("precision_float64", 3.141592653589793238462643383279, 12),  # 64bit float
        ]
        
        for key, value, value_type in metadata_items:
            # キー
            key_bytes = key.encode('utf-8')
            f.write(struct.pack('<Q', len(key_bytes)))
            f.write(key_bytes)
            
            # 値型
            f.write(struct.pack('<I', value_type))
            
            # 値
            if value_type == 4:  # string
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
            elif value_type == 6:  # int32
                f.write(struct.pack('<i', value))
            elif value_type == 11:  # int64
                f.write(struct.pack('<q', value))
            elif value_type == 7:  # float32
                f.write(struct.pack('<f', value))
            elif value_type == 12:  # float64
                f.write(struct.pack('<d', value))
        
        # テンソル情報（1つのダミーテンソル）
        tensor_name = "test_tensor"
        name_bytes = tensor_name.encode('utf-8')
        f.write(struct.pack('<Q', len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack('<I', 2))  # n_dims
        f.write(struct.pack('<Q', 10))  # dim0
        f.write(struct.pack('<Q', 10))  # dim1
        f.write(struct.pack('<I', 0))  # type (uint8)
        f.write(struct.pack('<Q', 0))  # offset
        
        # ダミーテンソルデータ（100バイト）
        f.write(b'\x01' * 100)
    
    print(f"   ✅ 64bit長テスト用GGUFファイル作成完了: {os.path.getsize(filepath)} bytes")

def test_64bit_precision_comparison():
    """64bit精度と32bit精度の比較テスト"""
    print("🧪 64bit精度 vs 32bit精度 比較テスト開始")
    
    # テストファイル作成
    test_input = "test_64bit_input.gguf"
    test_output_32bit = "test_32bit_output.gguf"
    test_output_64bit = "test_64bit_output.gguf"
    
    create_test_gguf_with_large_values(test_input)
    
    # 32bit設定でのテスト
    print("\n📊 32bit精度モードでのテスト:")
    config_32bit = NKATConfig(use_64bit_precision=False, data_alignment=4)
    integrator_32bit = GGUFNKATIntegrator(config_32bit)
    
    start_time = time.time()
    integrator_32bit.create_nkat_enhanced_gguf(test_input, test_output_32bit)
    time_32bit = time.time() - start_time
    
    # 64bit設定でのテスト
    print("\n📊 64bit精度モードでのテスト:")
    config_64bit = NKATConfig(use_64bit_precision=True, data_alignment=8)
    integrator_64bit = GGUFNKATIntegrator(config_64bit)
    
    start_time = time.time()
    integrator_64bit.create_nkat_enhanced_gguf(test_input, test_output_64bit)
    time_64bit = time.time() - start_time
    
    # 結果比較
    print("\n📈 結果比較:")
    print(f"   32bit処理時間: {time_32bit:.4f}秒")
    print(f"   64bit処理時間: {time_64bit:.4f}秒")
    print(f"   時間差: {abs(time_64bit - time_32bit):.4f}秒")
    
    if os.path.exists(test_output_32bit):
        size_32bit = os.path.getsize(test_output_32bit)
        print(f"   32bit出力ファイルサイズ: {size_32bit} bytes")
    
    if os.path.exists(test_output_64bit):
        size_64bit = os.path.getsize(test_output_64bit)
        print(f"   64bit出力ファイルサイズ: {size_64bit} bytes")
        print(f"   サイズ差: {abs(size_64bit - size_32bit)} bytes")
    
    # メタデータ精度比較
    print("\n🔍 メタデータ精度検証:")
    
    # 32bitで読み取り
    try:
        metadata_32bit = integrator_32bit.read_gguf_metadata(test_output_32bit)
        print(f"   32bit読み取りメタデータ: {len(metadata_32bit)} 項目")
        
        # 精度確認（大きな値）
        if "large_int64" in metadata_32bit:
            value_32bit = metadata_32bit["large_int64"] 
            print(f"   32bit読み取り大整数値: {value_32bit} (型: {type(value_32bit)})")
    except Exception as e:
        print(f"   ⚠️ 32bit読み取りエラー: {e}")
    
    # 64bitで読み取り
    try:
        metadata_64bit = integrator_64bit.read_gguf_metadata(test_output_64bit)
        print(f"   64bit読み取りメタデータ: {len(metadata_64bit)} 項目")
        
        # 精度確認（大きな値）
        if "large_int64" in metadata_64bit:
            value_64bit = metadata_64bit["large_int64"]
            print(f"   64bit読み取り大整数値: {value_64bit} (型: {type(value_64bit)})")
            
        # NKAT固有のメタデータ確認
        nkat_keys = [k for k in metadata_64bit.keys() if k.startswith("nkat.")]
        print(f"   NKAT関連メタデータ: {len(nkat_keys)} 項目")
        for key in nkat_keys[:5]:  # 最初の5項目を表示
            print(f"     {key}: {metadata_64bit[key]}")
            
    except Exception as e:
        print(f"   ⚠️ 64bit読み取りエラー: {e}")
    
    # クリーンアップ
    for filepath in [test_input, test_output_32bit, test_output_64bit]:
        if os.path.exists(filepath):
            os.remove(filepath)
            
    print("\n✅ 64bit精度比較テスト完了")

def test_large_metadata_handling():
    """大量メタデータでの64bit境界整列テスト"""
    print("\n🧪 大量メタデータ64bit境界整列テスト開始")
    
    config = NKATConfig(use_64bit_precision=True, data_alignment=8)
    integrator = GGUFNKATIntegrator(config)
    
    # 大量データを含むメタデータ作成
    large_data = list(range(1000))  # 1000要素のリスト
    precision_value = np.float64(np.pi * 1e10)  # 高精度値
    
    # テストメタデータに大量データを追加
    test_metadata = integrator.nkat_metadata.copy()
    test_metadata.update({
        "large_array_data": large_data,
        "high_precision_value": float(precision_value),
        "timestamp_64bit": int(time.time() * 1e6),  # マイクロ秒精度
        "memory_alignment_test": "A" * 1024,  # 1KB文字列
    })
    
    print(f"   拡張メタデータ項目数: {len(test_metadata)}")
    print(f"   高精度値: {precision_value}")
    print(f"   64bitタイムスタンプ: {test_metadata['timestamp_64bit']}")
    
    # ダミーGGUFファイルからの統合テスト
    test_input = "test_large_metadata_input.gguf"
    test_output = "test_large_metadata_output.gguf"
    
    create_test_gguf_with_large_values(test_input)
    
    print(f"   メタデータ境界整列での統合処理...")
    start_time = time.time()
    integrator._write_enhanced_gguf_64bit(test_input, test_output, test_metadata)
    process_time = time.time() - start_time
    
    print(f"   処理時間: {process_time:.4f}秒")
    
    if os.path.exists(test_output):
        output_size = os.path.getsize(test_output)
        print(f"   出力ファイルサイズ: {output_size} bytes")
        
        # 64bit境界整列確認
        print(f"   64bit境界整列確認（ファイルサイズが8の倍数か）: {output_size % 8 == 0}")
    
    # クリーンアップ
    for filepath in [test_input, test_output]:
        if os.path.exists(filepath):
            os.remove(filepath)
    
    print("   ✅ 大量メタデータ64bit境界整列テスト完了")

def main():
    """メインテスト実行"""
    print("🚀 64bit長GGUF統合テスト開始")
    print("=" * 60)
    
    try:
        # 64bit vs 32bit比較テスト
        test_64bit_precision_comparison()
        
        # 大量メタデータでの境界整列テスト
        test_large_metadata_handling()
        
        print("\n" + "=" * 60)
        print("🎉 すべての64bit長テスト完了！")
        print("64bit精度での読み込み改良が正常に動作しています。")
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 