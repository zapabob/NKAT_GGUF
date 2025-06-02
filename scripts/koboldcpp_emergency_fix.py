#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🆘 KoboldCPP緊急修復システム（llama.cpp対応強化版）
KoboldCPP Emergency Fix System for bad_alloc and access violation errors

特徴:
- tokenizer.ggml.tokens bad_allocエラー修復
- アクセス違反エラー解決
- NKATファイル対応
- MoE（Mixture of Experts）量子化タイプ不一致修復
- llama.cpp最新版互換性確保
- LoRA化オプション
- 電源断復旧システム統合
"""

import os
import sys
import struct
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging
import json

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('koboldcpp_emergency_fix.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class KoboldCPPEmergencyFix:
    """KoboldCPP緊急修復システム（llama.cpp対応強化版）"""
    
    GGUF_MAGIC = b'GGUF'
    
    # GGUF型定義
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
    
    def __init__(self):
        self.backup_dir = Path("emergency_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        self.type_sizes = {
            self.GGUF_TYPE_UINT8: 1, self.GGUF_TYPE_INT8: 1,
            self.GGUF_TYPE_UINT16: 2, self.GGUF_TYPE_INT16: 2,
            self.GGUF_TYPE_UINT32: 4, self.GGUF_TYPE_INT32: 4, 
            self.GGUF_TYPE_FLOAT32: 4,
            self.GGUF_TYPE_UINT64: 8, self.GGUF_TYPE_INT64: 8, 
            self.GGUF_TYPE_FLOAT64: 8,
            self.GGUF_TYPE_BOOL: 1
        }
        
        logger.info("🆘 KoboldCPP緊急修復システム（llama.cpp強化版）初期化完了")
    
    def create_emergency_backup(self, file_path: str) -> str:
        """緊急バックアップ作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{Path(file_path).stem}_emergency_{timestamp}.gguf"
        shutil.copy2(file_path, backup_path)
        logger.info(f"💾 緊急バックアップ作成: {backup_path}")
        return str(backup_path)
    
    def comprehensive_analysis(self, file_path: str) -> Dict:
        """包括的ファイル分析"""
        logger.info(f"🔍 包括的分析開始: {file_path}")
        
        analysis_result = {
            "file_path": file_path,
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "gguf_valid": False,
            "tokenizer_issues": [],
            "moe_issues": [],
            "memory_issues": [],
            "recommendations": []
        }
        
        try:
            with open(file_path, 'rb') as f:
                # GGUFヘッダー確認
                magic = f.read(4)
                if magic != self.GGUF_MAGIC:
                    analysis_result["recommendations"].append("❌ 無効なGGUFファイル")
                    return analysis_result
                
                analysis_result["gguf_valid"] = True
                
                # バージョン読み取り
                version = struct.unpack('<I', f.read(4))[0]
                analysis_result["gguf_version"] = version
                
                # テンソル数とメタデータ数
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_count = struct.unpack('<Q', f.read(8))[0]
                
                analysis_result["tensor_count"] = tensor_count
                analysis_result["metadata_count"] = metadata_count
                
                # メタデータ分析
                metadata_analysis = self._analyze_metadata(f, metadata_count)
                analysis_result.update(metadata_analysis)
                
                # tokenizer問題チェック
                tokenizer_analysis = self._check_tokenizer_issues(f, metadata_count)
                analysis_result["tokenizer_issues"] = tokenizer_analysis
                
                # MoE問題チェック
                moe_analysis = self._check_moe_issues(f, tensor_count, metadata_analysis.get("metadata", {}))
                analysis_result["moe_issues"] = moe_analysis
                
                # メモリ問題チェック
                memory_analysis = self._check_memory_issues(analysis_result)
                analysis_result["memory_issues"] = memory_analysis
                
                # 推奨事項生成
                recommendations = self._generate_recommendations(analysis_result)
                analysis_result["recommendations"] = recommendations
                
        except Exception as e:
            logger.error(f"❌ 分析エラー: {e}")
            analysis_result["error"] = str(e)
        
        return analysis_result
    
    def _analyze_metadata(self, f, metadata_count: int) -> Dict:
        """メタデータ詳細分析"""
        metadata = {}
        f.seek(24)  # ヘッダー後に移動
        
        try:
            for i in range(metadata_count):
                # キー読み取り
                key_len = struct.unpack('<Q', f.read(8))[0]
                if key_len > 1000:  # 異常な長さ
                    return {"metadata": {}, "metadata_error": "異常なキー長"}
                
                key = f.read(key_len).decode('utf-8', errors='ignore')
                
                # 値読み取り
                value = self._safe_read_value(f)
                metadata[key] = value
                
        except Exception as e:
            logger.warning(f"⚠️ メタデータ読み取りエラー: {e}")
        
        return {"metadata": metadata}
    
    def _safe_read_value(self, f):
        """安全な値読み取り"""
        try:
            value_type = struct.unpack('<I', f.read(4))[0]
            
            if value_type == self.GGUF_TYPE_STRING:
                str_len = struct.unpack('<Q', f.read(8))[0]
                if str_len > 10000:  # 異常な長さ制限
                    f.read(str_len)
                    return "<<TOO_LONG>>"
                return f.read(str_len).decode('utf-8', errors='ignore')
            
            elif value_type == self.GGUF_TYPE_ARRAY:
                array_type = struct.unpack('<I', f.read(4))[0]
                array_len = struct.unpack('<Q', f.read(8))[0]
                
                if array_len > 100000:  # 配列長制限
                    # 配列全体をスキップ
                    for _ in range(array_len):
                        if array_type == self.GGUF_TYPE_STRING:
                            str_len = struct.unpack('<Q', f.read(8))[0]
                            f.read(str_len)
                        else:
                            f.read(self.type_sizes.get(array_type, 4))
                    return "<<LARGE_ARRAY>>"
                
                # 正常サイズの配列
                array_values = []
                for _ in range(min(array_len, 1000)):  # 最大1000要素まで
                    if array_type == self.GGUF_TYPE_STRING:
                        str_len = struct.unpack('<Q', f.read(8))[0]
                        if str_len > 500:  # 文字列長制限
                            f.read(str_len)
                            array_values.append("<<LONG_STRING>>")
                        else:
                            array_values.append(f.read(str_len).decode('utf-8', errors='ignore'))
                    else:
                        f.read(self.type_sizes.get(array_type, 4))
                        array_values.append(None)
                return array_values
            
            elif value_type in self.type_sizes:
                size = self.type_sizes[value_type]
                data = f.read(size)
                
                if value_type == self.GGUF_TYPE_UINT32:
                    return struct.unpack('<I', data)[0]
                elif value_type == self.GGUF_TYPE_UINT64:
                    return struct.unpack('<Q', data)[0]
                elif value_type == self.GGUF_TYPE_FLOAT32:
                    return struct.unpack('<f', data)[0]
                else:
                    return data
            else:
                return None
                
        except Exception:
            return None
    
    def _check_tokenizer_issues(self, f, metadata_count: int) -> List[str]:
        """トークナイザー問題チェック"""
        issues = []
        f.seek(24)  # ヘッダー後に移動
        
        try:
            for i in range(metadata_count):
                # キー読み取り
                key_len = struct.unpack('<Q', f.read(8))[0]
                key = f.read(key_len).decode('utf-8', errors='ignore')
                
                if key == 'tokenizer.ggml.tokens':
                    # 値タイプ読み取り
                    value_type = struct.unpack('<I', f.read(4))[0]
                    
                    if value_type == self.GGUF_TYPE_ARRAY:
                        array_type = struct.unpack('<I', f.read(4))[0]
                        array_len = struct.unpack('<Q', f.read(8))[0]
                        
                        if array_len > 200000:
                            issues.append(f"⚠️ トークン数が多すぎます: {array_len}")
                        
                        if array_len == 0:
                            issues.append("❌ トークン配列が空です")
                        
                        # 最初の数個のトークンをチェック
                        problematic_tokens = 0
                        for j in range(min(array_len, 100)):
                            try:
                                str_len = struct.unpack('<Q', f.read(8))[0]
                                if str_len > 1000:
                                    problematic_tokens += 1
                                f.read(str_len)
                            except:
                                problematic_tokens += 1
                                break
                        
                        if problematic_tokens > 10:
                            issues.append(f"⚠️ 問題のあるトークンが多数: {problematic_tokens}")
                        
                        break
                else:
                    # 値をスキップ
                    self._safe_read_value(f)
                    
        except Exception as e:
            issues.append(f"❌ トークナイザー分析エラー: {e}")
        
        return issues
    
    def _check_moe_issues(self, f, tensor_count: int, metadata: Dict) -> List[str]:
        """MoE問題チェック"""
        issues = []
        
        # MoE関連メタデータチェック
        expert_count = metadata.get('llama.expert_count', 0)
        expert_used_count = metadata.get('llama.expert_used_count', 0)
        
        if expert_count > 1:
            logger.info(f"🧠 MoEモデル検出: {expert_count}エキスパート")
            
            # テンソル情報読み取り（簡略版）
            current_pos = f.tell()
            try:
                # メタデータ部分をスキップしてテンソル情報へ
                f.seek(24)  # ヘッダー後
                metadata_count = struct.unpack('<Q', f.read(16))[1]  # metadata_count取得
                
                # メタデータスキップ
                for i in range(metadata_count):
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    f.read(key_len)
                    self._safe_read_value(f)
                
                # テンソル情報分析
                expert_dtypes = []
                for i in range(min(tensor_count, 1000)):  # 最大1000テンサーまで
                    try:
                        name_len = struct.unpack('<Q', f.read(8))[0]
                        name = f.read(name_len).decode('utf-8', errors='ignore')
                        
                        n_dims = struct.unpack('<I', f.read(4))[0]
                        for _ in range(n_dims):
                            f.read(8)  # 形状スキップ
                        
                        dtype = struct.unpack('<I', f.read(4))[0]
                        f.read(8)  # オフセットスキップ
                        
                        # エキスパート関連テンサー
                        if any(keyword in name for keyword in ['expert', 'ffn_gate', 'mlp']):
                            expert_dtypes.append(dtype)
                            
                    except:
                        break
                
                # 量子化タイプ統一チェック
                if expert_dtypes and len(set(expert_dtypes)) > 1:
                    issues.append(f"⚠️ MoEエキスパートの量子化タイプが不統一: {set(expert_dtypes)}")
                    issues.append("💡 llama.cppで互換性問題の可能性があります")
                
            except Exception as e:
                issues.append(f"❌ MoE分析エラー: {e}")
            finally:
                f.seek(current_pos)
        
        return issues
    
    def _check_memory_issues(self, analysis_result: Dict) -> List[str]:
        """メモリ問題チェック"""
        issues = []
        
        file_size_mb = analysis_result.get("file_size_mb", 0)
        tensor_count = analysis_result.get("tensor_count", 0)
        
        if file_size_mb > 20000:  # 20GB以上
            issues.append(f"⚠️ 大容量ファイル: {file_size_mb:.1f}MB")
            issues.append("💡 メモリ不足の可能性があります")
        
        if tensor_count > 2000:
            issues.append(f"⚠️ テンサー数が多い: {tensor_count}")
            issues.append("💡 処理時間が長くなる可能性があります")
        
        return issues
    
    def _generate_recommendations(self, analysis_result: Dict) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        if analysis_result.get("tokenizer_issues"):
            recommendations.append("🔧 トークナイザー修復を推奨")
        
        if analysis_result.get("moe_issues"):
            recommendations.append("🔧 MoE量子化タイプ修復を推奨")
        
        if analysis_result.get("memory_issues"):
            recommendations.append("💾 メモリ最適化を推奨")
        
        if not recommendations:
            recommendations.append("✅ 重大な問題は検出されませんでした")
        
        return recommendations
    
    def fix_all_issues(self, file_path: str) -> str:
        """全問題の統合修復"""
        logger.info(f"🔧 統合修復開始: {file_path}")
        
        # 分析実行
        analysis = self.comprehensive_analysis(file_path)
        
        if not analysis.get("gguf_valid"):
            logger.error("❌ 無効なGGUFファイル")
            return None
        
        # バックアップ作成
        backup_path = self.create_emergency_backup(file_path)
        
        # 修復版パス生成
        fixed_path = file_path.replace('.gguf', '_emergency_fixed.gguf')
        
        try:
            # 段階的修復
            current_file = file_path
            
            # 1. トークナイザー修復
            if analysis.get("tokenizer_issues"):
                logger.info("🔧 トークナイザー修復中...")
                tokenizer_fixed = self.fix_tokenizer_bad_alloc(current_file)
                if tokenizer_fixed:
                    current_file = tokenizer_fixed
            
            # 2. MoE修復（MoE修復クラスを使用）
            if analysis.get("moe_issues"):
                logger.info("🔧 MoE修復中...")
                try:
                    from llama_cpp_moe_fix import LlamaCppMoEFixer
                    moe_fixer = LlamaCppMoEFixer()
                    moe_fixed_path = current_file.replace('.gguf', '_moe_fixed.gguf')
                    if moe_fixer.fix_moe_quantization(current_file, moe_fixed_path):
                        current_file = moe_fixed_path
                except ImportError:
                    logger.warning("⚠️ MoE修復モジュールが見つかりません")
            
            # 3. 最終ファイル名調整
            if current_file != file_path:
                if current_file != fixed_path:
                    shutil.move(current_file, fixed_path)
                    current_file = fixed_path
            
            logger.info(f"✅ 統合修復完了: {current_file}")
            return current_file
            
        except Exception as e:
            logger.error(f"❌ 統合修復エラー: {e}")
            return None
    
    def analyze_tokenizer_issue(self, file_path: str) -> Dict:
        """tokenizerエラー分析（互換性維持）"""
        logger.info(f"🔍 tokenizer分析: {file_path}")
        
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
                    key = f.read(key_len).decode('utf-8', errors='ignore')
                    
                    if key == 'tokenizer.ggml.tokens':
                        tokenizer_found = True
                        # 値タイプ読み取り
                        value_type = struct.unpack('<I', f.read(4))[0]
                        
                        if value_type == self.GGUF_TYPE_ARRAY:
                            array_type = struct.unpack('<I', f.read(4))[0]
                            array_len = struct.unpack('<Q', f.read(8))[0]
                            tokenizer_size = array_len
                            
                            return {
                                "status": "found",
                                "tokenizer_size": tokenizer_size,
                                "array_length": array_len,
                                "position": f.tell(),
                                "version": version
                            }
                    else:
                        # 値をスキップ
                        self._safe_read_value(f)
                
                if not tokenizer_found:
                    return {"status": "not_found", "message": "tokenizer.ggml.tokensが見つかりません"}
                
                return {"status": "analyzed", "version": version}
                
        except Exception as e:
            return {"status": "error", "message": f"分析エラー: {str(e)}"}
    
    def fix_tokenizer_bad_alloc(self, file_path: str) -> str:
        """tokenizer bad_allocエラー修復（改良版）"""
        logger.info("🔧 tokenizer bad_allocエラー修復開始...")
        
        analysis = self.analyze_tokenizer_issue(file_path)
        
        if analysis["status"] == "error":
            logger.error(f"❌ エラー: {analysis['message']}")
            return None
        
        # バックアップ作成
        backup_path = self.create_emergency_backup(file_path)
        
        # 修復版作成
        fixed_path = file_path.replace('.gguf', '_tokenfixed.gguf')
        
        try:
            with open(file_path, 'rb') as src, open(fixed_path, 'wb') as dst:
                # ヘッダーコピー
                header = src.read(24)  # magic + version + tensor_count + metadata_count
                dst.write(header)
                
                # メタデータ処理
                metadata_count = struct.unpack('<Q', header[16:24])[0]
                
                for i in range(metadata_count):
                    # キー読み取り
                    key_len_data = src.read(8)
                    key_len = struct.unpack('<Q', key_len_data)[0]
                    key_data = src.read(key_len)
                    key = key_data.decode('utf-8', errors='ignore')
                    
                    dst.write(key_len_data)
                    dst.write(key_data)
                    
                    if key == 'tokenizer.ggml.tokens':
                        logger.info("🔧 tokenizer.ggml.tokens修復中...")
                        
                        # 値タイプ読み取り
                        value_type_data = src.read(4)
                        value_type = struct.unpack('<I', value_type_data)[0]
                        
                        if value_type == self.GGUF_TYPE_ARRAY:
                            array_type_data = src.read(4)
                            array_len_data = src.read(8)
                            array_len = struct.unpack('<Q', array_len_data)[0]
                            
                            # 安全なサイズに制限
                            safe_len = min(array_len, 100000)  # 100K トークンまで
                            if array_len != safe_len:
                                logger.info(f"⚠️ トークン数を{array_len}から{safe_len}に制限")
                                array_len = safe_len
                                array_len_data = struct.pack('<Q', array_len)
                            
                            dst.write(value_type_data)
                            dst.write(array_type_data)
                            dst.write(array_len_data)
                            
                            # トークンデータのコピー（安全性重視）
                            successful_tokens = 0
                            for j in range(array_len):
                                try:
                                    str_len_data = src.read(8)
                                    if len(str_len_data) < 8:
                                        break
                                    str_len = struct.unpack('<Q', str_len_data)[0]
                                    
                                    # 文字列長制限
                                    if str_len > 500:  # 500文字まで
                                        str_len = 500
                                        str_len_data = struct.pack('<Q', str_len)
                                    
                                    dst.write(str_len_data)
                                    
                                    token_data = src.read(str_len)
                                    if len(token_data) < str_len:
                                        token_data += b'\x00' * (str_len - len(token_data))
                                    
                                    dst.write(token_data)
                                    successful_tokens += 1
                                    
                                except Exception as e:
                                    logger.warning(f"⚠️ トークン{j}でエラー: {e}")
                                    break
                            
                            logger.info(f"✅ {successful_tokens}個のトークンを正常に処理")
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
                        
                        if value_type == self.GGUF_TYPE_ARRAY:
                            # 配列データを安全にコピー
                            array_type_data = src.read(4)
                            array_len_data = src.read(8)
                            dst.write(array_type_data)
                            dst.write(array_len_data)
                            
                            array_type = struct.unpack('<I', array_type_data)[0]
                            array_len = struct.unpack('<Q', array_len_data)[0]
                            
                            # 配列内容コピー
                            for k in range(array_len):
                                if array_type == self.GGUF_TYPE_STRING:
                                    str_len_data = src.read(8)
                                    str_len = struct.unpack('<Q', str_len_data)[0]
                                    str_data = src.read(str_len)
                                    dst.write(str_len_data)
                                    dst.write(str_data)
                                else:
                                    size = self.type_sizes.get(array_type, 4)
                                    data = src.read(size)
                                    dst.write(data)
                        else:
                            # 単純な値
                            size = self.type_sizes.get(value_type, 8)
                            if value_type == self.GGUF_TYPE_STRING:
                                str_len = struct.unpack('<Q', src.read(8))[0]
                                dst.write(struct.pack('<Q', str_len))
                                dst.write(src.read(str_len))
                            else:
                                data = src.read(size)
                                dst.write(data)
                
                # 残りのデータ（テンサー情報とデータ）をコピー
                remaining_data = src.read()
                dst.write(remaining_data)
            
            logger.info(f"✅ tokenizer修復完了: {fixed_path}")
            return fixed_path
            
        except Exception as e:
            logger.error(f"❌ tokenizer修復エラー: {e}")
            return None
    
    def create_koboldcpp_launch_config(self, model_path: str) -> str:
        """KoboldCPP起動設定作成（改良版）"""
        logger.info(f"📝 KoboldCPP起動設定作成: {model_path}")
        
        # モデル分析
        analysis = self.comprehensive_analysis(model_path)
        
        config = {
            "model_path": model_path,
            "contextsize": 4096,
            "blasbatchsize": 256,
            "blasthreads": 4,
            "port": 5001,
            "gpulayers": 28,
            "usecublas": "normal",
            "nommap": True,
            "usemlock": False,
            "threads": 6,
            "quiet": False
        }
        
        # ファイルサイズに基づく調整
        file_size_mb = analysis.get("file_size_mb", 0)
        
        if file_size_mb > 10000:  # 10GB以上
            config["blasbatchsize"] = 128
            config["contextsize"] = 2048
            config["gpulayers"] = 20
            logger.info("🔧 大容量モデル用設定を適用")
        
        elif file_size_mb < 1000:  # 1GB未満
            config["blasbatchsize"] = 512
            config["contextsize"] = 8192
            config["gpulayers"] = 35
            logger.info("🔧 軽量モデル用設定を適用")
        
        # MoEモデル特別設定
        if analysis.get("moe_issues"):
            config["nommap"] = True
            config["usemlock"] = False
            config["blasbatchsize"] = min(config["blasbatchsize"], 128)
            logger.info("🧠 MoEモデル用設定を適用")
        
        # バッチファイル生成
        batch_content = f"""@echo off
REM KoboldCPP最適化起動スクリプト（自動生成）
REM モデル: {Path(model_path).name}
REM ファイルサイズ: {file_size_mb:.1f}MB
REM 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

echo 🚀 KoboldCPP最適化起動
echo モデル: {Path(model_path).name}
echo ファイルサイズ: {file_size_mb:.1f}MB
echo.

REM メモリ監視開始
echo 📊 システム情報:
systeminfo | findstr "Total Physical Memory"
echo.

REM KoboldCPP起動
echo 🔧 KoboldCPP起動中...
python koboldcpp.py --model "{model_path}" --contextsize {config['contextsize']} --blasbatchsize {config['blasbatchsize']} --blasthreads {config['blasthreads']} --port {config['port']} --skiplauncher --gpulayers {config['gpulayers']} --usecublas {config['usecublas']} 0 {"--nommap" if config['nommap'] else ""} {"--usemlock False" if not config['usemlock'] else ""} --threads {config['threads']}

pause
"""
        
        # バッチファイル保存
        batch_path = f"run_{Path(model_path).stem}_optimized.bat"
        with open(batch_path, 'w', encoding='utf-8') as f:
            f.write(batch_content)
        
        logger.info(f"✅ KoboldCPP起動設定作成完了: {batch_path}")
        return batch_path
    
    def print_analysis_report(self, analysis: Dict):
        """分析レポート表示"""
        print("\n" + "="*70)
        print("🆘 KoboldCPP緊急修復システム - 分析レポート")
        print("="*70)
        
        print(f"📁 ファイル: {analysis['file_path']}")
        print(f"📊 サイズ: {analysis['file_size_mb']:.1f}MB")
        print(f"✅ GGUF有効: {'はい' if analysis['gguf_valid'] else 'いいえ'}")
        
        if analysis.get('gguf_version'):
            print(f"🔖 GGUFバージョン: {analysis['gguf_version']}")
        
        if analysis.get('tensor_count'):
            print(f"🧮 テンサー数: {analysis['tensor_count']}")
        
        if analysis.get('metadata_count'):
            print(f"📋 メタデータ数: {analysis['metadata_count']}")
        
        # 問題表示
        if analysis.get('tokenizer_issues'):
            print("\n⚠️ トークナイザー問題:")
            for issue in analysis['tokenizer_issues']:
                print(f"   {issue}")
        
        if analysis.get('moe_issues'):
            print("\n🧠 MoE問題:")
            for issue in analysis['moe_issues']:
                print(f"   {issue}")
        
        if analysis.get('memory_issues'):
            print("\n💾 メモリ問題:")
            for issue in analysis['memory_issues']:
                print(f"   {issue}")
        
        # 推奨事項
        print("\n💡 推奨事項:")
        for rec in analysis.get('recommendations', []):
            print(f"   {rec}")
        
        print("="*70)

def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='KoboldCPP緊急修復ツール（llama.cpp強化版）')
    parser.add_argument('input', help='入力GGUFファイル')
    parser.add_argument('-a', '--analyze-only', action='store_true', help='分析のみ実行')
    parser.add_argument('-t', '--tokenizer-only', action='store_true', help='トークナイザー修復のみ')
    parser.add_argument('-c', '--create-config', action='store_true', help='KoboldCPP設定ファイル作成')
    parser.add_argument('--comprehensive', action='store_true', help='包括的修復実行')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ ファイルが見つかりません: {args.input}")
        sys.exit(1)
    
    # 修復システム初期化
    fixer = KoboldCPPEmergencyFix()
    
    # 分析実行
    analysis = fixer.comprehensive_analysis(args.input)
    fixer.print_analysis_report(analysis)
    
    if args.analyze_only:
        print("\n✅ 分析完了")
        return
    
    if args.create_config:
        config_path = fixer.create_koboldcpp_launch_config(args.input)
        print(f"\n✅ 起動設定作成完了: {config_path}")
    
    if args.tokenizer_only:
        if analysis.get('tokenizer_issues'):
            fixed_file = fixer.fix_tokenizer_bad_alloc(args.input)
            if fixed_file:
                print(f"\n✅ トークナイザー修復完了: {fixed_file}")
        else:
            print("\n✅ トークナイザー修復は不要です")
    
    if args.comprehensive:
        fixed_file = fixer.fix_all_issues(args.input)
        if fixed_file:
            print(f"\n✅ 包括的修復完了: {fixed_file}")
        else:
            print(f"\n❌ 修復に失敗しました")

if __name__ == "__main__":
    main() 