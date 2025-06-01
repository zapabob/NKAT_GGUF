#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GGUF NKAT統合テスト（2KB問題修正確認）
"""

import os
import sys
from pathlib import Path
import colab_gguf_nkat_integration as nkat
import struct

def test_gguf_integration():
    """GGUF統合テスト"""
    print("🔄 GGUF NKAT統合テスト開始...")
    
    # 現在のディレクトリ確認
    current_dir = Path('.')
    print(f"📁 作業ディレクトリ: {current_dir.resolve()}")
    
    # GGUFファイルを検索（再帰的）
    print("🔍 GGUFファイル検索中...")
    gguf_files = list(current_dir.glob('*.gguf'))
    
    # ルートディレクトリにない場合はサブディレクトリも検索
    if not gguf_files:
        print("   ルートディレクトリに無し、サブディレクトリ検索中...")
        gguf_files = list(current_dir.glob('**/*.gguf'))
    
    print(f"   発見されたGGUFファイル: {len(gguf_files)}個")
    
    for i, gguf_file in enumerate(gguf_files):
        size_mb = os.path.getsize(gguf_file) / (1024 * 1024)
        print(f"   {i+1}. {gguf_file} ({size_mb:.2f}MB)")
    
    if not gguf_files:
        print("❌ GGUFファイルが見つかりません")
        print("🔄 テスト用GGUFファイルを作成中...")
        
        # テスト用の大きなGGUFファイル作成
        test_gguf = create_test_gguf_file()
        if test_gguf:
            gguf_files = [test_gguf]
            print(f"✅ テスト用GGUFファイル作成: {test_gguf}")
        else:
            return False
    
    # 最大のファイルを選択（2KB問題のファイルは除外）
    valid_files = [f for f in gguf_files if os.path.getsize(f) > 1024 * 1024]  # 1MB以上
    
    if not valid_files:
        print("⚠️ 有効な大きさのGGUFファイルがありません")
        # 最初のファイルでもテスト
        input_file = str(gguf_files[0])
        print(f"   最小サイズファイルでテスト: {input_file}")
    else:
        input_file = str(max(valid_files, key=lambda f: os.path.getsize(f)))
        print(f"   最大ファイル選択: {input_file}")
    
    # test_modelsディレクトリ作成
    output_dir = Path('test_models')
    output_dir.mkdir(exist_ok=True)
    print(f"📁 出力ディレクトリ: {output_dir.resolve()}")
    
    output_file = output_dir / f"{Path(input_file).stem}_size_fixed.gguf"
    
    print(f"📁 入力ファイル: {input_file}")
    print(f"📁 出力ファイル: {output_file}")
    
    # 入力ファイルサイズ確認
    input_size_mb = os.path.getsize(input_file) / (1024 * 1024)
    print(f"📊 入力サイズ: {input_size_mb:.2f}MB")
    
    try:
        # NKAT統合実行
        print("🔄 NKAT統合実行中...")
        integrator = nkat.GGUFNKATIntegrator()
        integrator.create_nkat_enhanced_gguf(input_file, str(output_file))
        
        # 出力ファイル確認
        print("📋 出力ファイル確認中...")
        if output_file.exists():
            output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"📊 出力サイズ: {output_size_mb:.2f}MB")
            
            # サイズ比較
            size_ratio = output_size_mb / input_size_mb if input_size_mb > 0 else 0
            print(f"📈 サイズ比: {size_ratio:.3f}")
            
            if output_size_mb < 0.005:  # 5KB未満なら問題
                print("❌ 出力ファイルが小さすぎます（2KB問題未解決）")
                return False
            elif input_size_mb > 1 and size_ratio > 0.8:  # 80%以上なら成功
                print("✅ テンソルデータが正常に保持されています")
                return True
            elif input_size_mb <= 1:  # 小さい入力ファイルの場合
                print("⚠️ 入力ファイルが小さいため判定困難、出力サイズで判定")
                if output_size_mb > 0.01:  # 10KB以上なら一応成功
                    print("✅ メタデータ処理は正常です")
                    return True
                else:
                    print("❌ 出力が小さすぎます")
                    return False
            else:
                print("⚠️ 出力ファイルが予想より小さいです")
                return False
        else:
            print("❌ 出力ファイルが作成されませんでした")
            return False
            
    except Exception as e:
        print(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_gguf_file():
    """テスト用の大きなGGUFファイルを作成"""
    try:
        test_file = Path('test_large.gguf')
        
        # 簡易的なGGUFファイル構造を作成
        with open(test_file, 'wb') as f:
            # GGUF ヘッダー
            f.write(b'GGUF')  # Magic
            f.write(struct.pack('<I', 3))  # Version
            f.write(struct.pack('<Q', 2))  # Tensor count
            f.write(struct.pack('<Q', 3))  # Metadata count
            
            # サンプルメタデータ
            metadata = {
                'general.architecture': 'test_model',
                'general.name': 'Test Model',
                'general.description': 'Test GGUF file for debugging'
            }
            
            for key, value in metadata.items():
                # キー
                key_bytes = key.encode('utf-8')
                f.write(struct.pack('<Q', len(key_bytes)))
                f.write(key_bytes)
                
                # 値（string type）
                f.write(struct.pack('<I', 4))  # string type
                value_bytes = value.encode('utf-8')
                f.write(struct.pack('<Q', len(value_bytes)))
                f.write(value_bytes)
            
            # ダミーテンソル情報（2個）
            for i in range(2):
                # テンソル名
                tensor_name = f'tensor_{i}'.encode('utf-8')
                f.write(struct.pack('<Q', len(tensor_name)))
                f.write(tensor_name)
                
                # 次元数
                f.write(struct.pack('<I', 2))
                
                # 各次元のサイズ
                f.write(struct.pack('<Q', 1024))  # 次元1
                f.write(struct.pack('<Q', 512))   # 次元2
                
                # データ型（float32）
                f.write(struct.pack('<I', 1))
                
                # オフセット
                f.write(struct.pack('<Q', f.tell() + 1000))
            
            # ダミーテンソルデータ（約10MB）
            dummy_data = b'\x00' * (10 * 1024 * 1024)
            f.write(dummy_data)
        
        print(f"   ✅ テスト用GGUFファイル作成: {test_file} ({os.path.getsize(test_file)/(1024*1024):.2f}MB)")
        return test_file
        
    except Exception as e:
        print(f"   ❌ テスト用ファイル作成失敗: {e}")
        return None

if __name__ == "__main__":
    success = test_gguf_integration()
    if success:
        print("\n🎉 テスト成功：2KB問題が修正されました！")
    else:
        print("\n💥 テスト失敗：2KB問題が未解決です")
    
    sys.exit(0 if success else 1) 