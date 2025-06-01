#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合NKAT GGUF処理システム
Integrated NKAT GGUF Processing System with GUI and Advanced Features
"""

import os
import sys
import json
import time
import shutil
import struct
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import configparser

# tqdmインポート
try:
    from tqdm import tqdm
except ImportError:
    # tqdmが無い場合のダミー実装
    class tqdm:
        def __init__(self, iterable=None, desc=None, total=None):
            self.iterable = iterable
            self.desc = desc
            self.total = total
            self._current = 0
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
        
        def update(self, n=1):
            self._current += n
        
        def set_description(self, desc):
            self.desc = desc

# ドラッグアンドドロップサポート
try:
    import tkinterdnd2 as TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False
    TkinterDnD = None

# CUDA対応チェック
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
except ImportError:
    CUDA_AVAILABLE = False
    DEVICE = "cpu"

@dataclass
class IntegratedNKATConfig:
    """統合NKAT設定"""
    # 基本設定
    enable_ka_operators: bool = True
    ka_grid_size: int = 8
    lie_algebra_dim: int = 4
    noncommutative_strength: float = 0.1
    differential_geometric_scale: float = 0.01
    spectral_radius_bound: float = 1.0
    quantization_aware: bool = True
    
    # 64bit精度設定
    use_64bit_precision: bool = True
    data_alignment: int = 8
    
    # コルモゴロフ設定
    max_rank: int = 8
    tolerance: float = 1e-6
    kolmogorov_strength: float = 0.05
    
    # バックアップ設定
    auto_backup: bool = True
    backup_dir: str = "backups"
    max_backups: int = 10
    
    # プリセット設定
    auto_save_presets: bool = True
    preset_file: str = "nkat_presets.json"
    remember_file_locations: bool = True
    
    # パフォーマンス設定
    enable_cuda: bool = CUDA_AVAILABLE
    max_threads: int = 4
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)

class IntegratedKolmogorovOperator:
    """統合コルモゴロフ演算子"""
    
    def __init__(self, config: IntegratedNKATConfig):
        self.config = config
        self.generators = self._initialize_generators()
        print(f"🔬 統合コルモゴロフ演算子を初期化しました")
    
    def _initialize_generators(self) -> List[np.ndarray]:
        """非可換代数生成子の初期化"""
        gen1 = np.array([[0, 1], [1, 0]], dtype=np.float64)
        gen2 = np.array([[0, -1], [1, 0]], dtype=np.float64)
        gen3 = np.array([[1, 0], [0, -1]], dtype=np.float64)
        identity = np.eye(2, dtype=np.float64)
        return [gen1, gen2, gen3, identity]
    
    def enhance_tensor(self, tensor: np.ndarray) -> Dict[str, Any]:
        """テンソル拡張処理"""
        print(f"  🔧 テンソル拡張処理: {tensor.shape}")
        
        try:
            # 前処理
            preprocessed = self._preprocess_tensor(tensor)
            
            # 非可換変換
            noncommutative = self._apply_noncommutative_transform(preprocessed)
            
            # コルモゴロフ理論適用
            kolmogorov_enhanced = self._apply_kolmogorov_theory(noncommutative)
            
            # 品質評価
            quality = self._evaluate_quality(tensor, kolmogorov_enhanced)
            
            return {
                'enhanced_tensor': kolmogorov_enhanced,
                'quality_metrics': quality,
                'success': True
            }
            
        except Exception as e:
            print(f"  ❌ テンソル拡張エラー: {e}")
            return {
                'enhanced_tensor': tensor,
                'quality_metrics': {'enhancement_score': 0.0},
                'success': False
            }
    
    def _preprocess_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """テンソル前処理（数値安定性最強化版）"""
        # Step 1: データ型確認と変換
        if tensor.dtype != np.float32:
            tensor = tensor.astype(np.float32)
        
        # Step 2: NaN/Inf値の徹底検出と修正
        # math.isnan(), math.isinf() よりもnp.isnan(), np.isinf()の方が配列処理に適している
        nan_mask = np.isnan(tensor)
        posinf_mask = np.isposinf(tensor)
        neginf_mask = np.isneginf(tensor)
        
        if np.any(nan_mask) or np.any(posinf_mask) or np.any(neginf_mask):
            print(f"    ⚠️ 異常値検出: NaN={np.sum(nan_mask)}, +Inf={np.sum(posinf_mask)}, -Inf={np.sum(neginf_mask)}")
            
            # より保守的な値で置換
            tensor = np.where(nan_mask, 0.0, tensor)
            tensor = np.where(posinf_mask, 1.0, tensor)  # より小さな値
            tensor = np.where(neginf_mask, -1.0, tensor)
        
        # Step 3: float32の限界値チェック
        FLOAT32_MAX = 3.4028235e+38
        SAFE_MAX = 1e6  # より保守的な最大値
        
        extreme_mask = np.abs(tensor) > SAFE_MAX
        if np.any(extreme_mask):
            extreme_count = np.sum(extreme_mask)
            max_val = np.max(np.abs(tensor))
            print(f"    📊 極端値修正: {extreme_count}個の値 (最大: {max_val:.2e})")
            
            # 極端値をクリッピング
            tensor = np.clip(tensor, -SAFE_MAX, SAFE_MAX)
        
        # Step 4: 統計的異常値の検出と修正
        if tensor.size > 1:
            # 中央値とMADを使用（外れ値に頑健）
            median_val = np.median(tensor)
            mad = np.median(np.abs(tensor - median_val))
            
            if mad > 1e-10:  # MADが0でない場合
                # 修正Zスコアを使用
                modified_z_scores = 0.6745 * (tensor - median_val) / mad
                outlier_mask = np.abs(modified_z_scores) > 3.5
                
                if np.any(outlier_mask):
                    outlier_count = np.sum(outlier_mask)
                    print(f"    📊 外れ値修正: {outlier_count}個")
                    
                    # 外れ値を中央値に置換
                    tensor = np.where(outlier_mask, median_val, tensor)
        
        # Step 5: 正規化（数値安定版）
        tensor_std = np.std(tensor)
        tensor_mean = np.mean(tensor)
        
        # 標準偏差が小さすぎる場合の処理
        if tensor_std < 1e-10:
            print(f"    ⚠️ 標準偏差が小さすぎます: {tensor_std:.2e}")
            # 小さなノイズを追加して数値安定性を確保
            noise = np.random.normal(0, 1e-8, tensor.shape).astype(np.float32)
            tensor = tensor + noise
            tensor_std = np.std(tensor)
            tensor_mean = np.mean(tensor)
        
        # 正規化実行
        if tensor_std > 1e-10:
            normalized = (tensor - tensor_mean) / tensor_std
            # より厳しいクリッピング
            normalized = np.clip(normalized, -3.0, 3.0)
        else:
            normalized = tensor
        
        # Step 6: 最終チェック
        final_tensor = normalized.astype(np.float32)
        
        # 最終的なNaN/Inf確認
        if np.any(np.isnan(final_tensor)) or np.any(np.isinf(final_tensor)):
            print(f"    🚨 最終チェック失敗、ゼロで初期化")
            final_tensor = np.zeros_like(tensor, dtype=np.float32)
        
        return final_tensor
    
    def _apply_noncommutative_transform(self, tensor: np.ndarray) -> np.ndarray:
        """非可換変換（数値安定性強化版）"""
        try:
            original_shape = tensor.shape
            
            # 入力検証
            if tensor.size == 0:
                return tensor
            
            # 2x2ブロック処理用に整形
            if len(original_shape) >= 2 and original_shape[-1] >= 2:
                reshaped = tensor.reshape(-1, original_shape[-1])
                transformed = np.zeros_like(reshaped, dtype=np.float32)
                
                # 生成子選択（サイクリック）
                generator_idx = np.random.randint(0, len(self.generators))
                generator = self.generators[generator_idx].astype(np.float32)
                
                # 安全な強度設定
                safe_strength = min(self.config.noncommutative_strength, 0.1)
                
                for j in range(0, reshaped.shape[1], 2):
                    if j + 1 < reshaped.shape[1]:
                        # 2x2ブロック抽出
                        block = reshaped[:, j:j+2].astype(np.float64)  # 高精度計算
                        
                        # NaN/Inf チェック
                        if np.any(np.isnan(block)) or np.any(np.isinf(block)):
                            transformed[:, j:j+2] = reshaped[:, j:j+2]
                            continue
                        
                        # 数値安定性のための正規化
                        block_max = np.max(np.abs(block))
                        if block_max > 1e3:
                            block = block / block_max
                        
                        # 交換子計算（数値安定版）
                        try:
                            # 各行を独立処理
                            for i in range(block.shape[0]):
                                row_2x2 = block[i, :].reshape(1, 2)
                                
                                # より安全な交換子計算
                                left_mult = np.dot(generator[:1, :], row_2x2.T)
                                right_mult = np.dot(row_2x2, generator[:, :1])
                                
                                if left_mult.shape == right_mult.T.shape:
                                    commutator_row = left_mult.flatten() - right_mult.flatten()
                                    
                                    # オーバーフローチェック
                                    if np.any(np.abs(commutator_row) > 1e6):
                                        commutator_row = np.clip(commutator_row, -100, 100)
                                    
                                    # 最終変換
                                    corrected_row = row_2x2.flatten() + safe_strength * commutator_row
                                    
                                    # NaN/Infチェック
                                    if not (np.any(np.isnan(corrected_row)) or np.any(np.isinf(corrected_row))):
                                        transformed[i, j:j+2] = corrected_row[:2].astype(np.float32)
                                    else:
                                        transformed[i, j:j+2] = reshaped[i, j:j+2]
                                else:
                                    transformed[i, j:j+2] = reshaped[i, j:j+2]
                                    
                        except Exception as e:
                            print(f"      ⚠️ 交換子計算エラー: {e}")
                            transformed[:, j:j+2] = reshaped[:, j:j+2]
                    else:
                        # 余った1列
                        transformed[:, j] = reshaped[:, j]
                
                return transformed.reshape(original_shape)
            
            return tensor
            
        except Exception as e:
            print(f"    ⚠️ 非可換変換エラー: {e}")
            return tensor
    
    def _apply_kolmogorov_theory(self, tensor: np.ndarray) -> np.ndarray:
        """コルモゴロフ理論適用（数値安定版）"""
        try:
            if tensor.size <= 1:
                return tensor
            
            # ラプラシアン計算（安定版）
            laplacian = self._compute_stable_laplacian(tensor)
            
            # 勾配計算（安定版）
            gradient = self._compute_stable_gradient(tensor)
            
            # 安全な組み合わせ
            safe_scale = min(self.config.kolmogorov_strength, 0.01)
            
            # 段階的適用
            enhanced = tensor.copy()
            if not (np.any(np.isnan(laplacian)) or np.any(np.isinf(laplacian))):
                enhanced = enhanced + safe_scale * 0.1 * laplacian
            
            if not (np.any(np.isnan(gradient)) or np.any(np.isinf(gradient))):
                enhanced = enhanced + safe_scale * 0.01 * gradient
            
            # 最終安定性チェック
            if np.any(np.isnan(enhanced)) or np.any(np.isinf(enhanced)):
                print(f"    🔧 コルモゴロフ適用後にNaN/Inf検出、元テンソル使用")
                return tensor
            
            # 適度な範囲にクリッピング
            enhanced = np.clip(enhanced, -100.0, 100.0)
            
            return enhanced.astype(np.float32)
            
        except Exception as e:
            print(f"    ⚠️ コルモゴロフ理論適用エラー: {e}")
            return tensor
    
    def _compute_stable_laplacian(self, tensor: np.ndarray) -> np.ndarray:
        """安定なラプラシアン計算"""
        try:
            laplacian = np.zeros_like(tensor, dtype=np.float64)
            
            for axis in range(len(tensor.shape)):
                if tensor.shape[axis] >= 3:
                    # 2階差分（中央差分）
                    try:
                        # パディング
                        padded = np.pad(tensor, [(1, 1) if i == axis else (0, 0) 
                                               for i in range(len(tensor.shape))], 
                                      mode='edge')
                        
                        # インデックス作成
                        slices_left = [slice(None)] * len(tensor.shape)
                        slices_center = [slice(None)] * len(tensor.shape)
                        slices_right = [slice(None)] * len(tensor.shape)
                        
                        slices_left[axis] = slice(0, -2)
                        slices_center[axis] = slice(1, -1)
                        slices_right[axis] = slice(2, None)
                        
                        # 2階差分
                        second_diff = (padded[tuple(slices_left)] - 
                                     2 * padded[tuple(slices_center)] + 
                                     padded[tuple(slices_right)])
                        
                        # NaN/Infチェック
                        if not (np.any(np.isnan(second_diff)) or np.any(np.isinf(second_diff))):
                            laplacian += second_diff
                        
                    except Exception as e:
                        print(f"      ⚠️ 軸{axis}のラプラシアン計算エラー: {e}")
                        continue
            
            # スケーリング
            laplacian = laplacian / max(len(tensor.shape), 1)
            
            return laplacian.astype(np.float32)
            
        except Exception as e:
            print(f"    ⚠️ ラプラシアン計算エラー: {e}")
            return np.zeros_like(tensor, dtype=np.float32)
    
    def _compute_stable_gradient(self, tensor: np.ndarray) -> np.ndarray:
        """安定な勾配計算"""
        try:
            gradient = np.zeros_like(tensor, dtype=np.float64)
            
            for axis in range(len(tensor.shape)):
                if tensor.shape[axis] >= 2:
                    try:
                        # 前進差分
                        diff = np.diff(tensor, axis=axis)
                        
                        # パディングして元のサイズに戻す
                        pad_widths = [(0, 0)] * len(tensor.shape)
                        pad_widths[axis] = (0, 1)
                        padded_diff = np.pad(diff, pad_widths, mode='edge')
                        
                        # NaN/Infチェック
                        if not (np.any(np.isnan(padded_diff)) or np.any(np.isinf(padded_diff))):
                            gradient += padded_diff
                        
                    except Exception as e:
                        print(f"      ⚠️ 軸{axis}の勾配計算エラー: {e}")
                        continue
            
            return gradient.astype(np.float32)
            
        except Exception as e:
            print(f"    ⚠️ 勾配計算エラー: {e}")
            return np.zeros_like(tensor, dtype=np.float32)
    
    def _evaluate_quality(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, float]:
        """品質評価（数値安定性強化版）"""
        try:
            # 形状とサイズの事前チェック
            if original.size == 0 or enhanced.size == 0:
                return {'enhancement_score': 0.0, 'correlation': 0.0, 'mse': float('inf'), 'snr_db': -100.0}
            
            # サイズを合わせる
            min_size = min(original.size, enhanced.size)
            if min_size < 2:
                return {'enhancement_score': 0.0, 'correlation': 0.0, 'mse': 0.0, 'snr_db': 0.0}
            
            orig_flat = original.flatten()[:min_size].astype(np.float64)
            enh_flat = enhanced.flatten()[:min_size].astype(np.float64)
            
            # NaN/Inf値の事前除去
            valid_mask = np.isfinite(orig_flat) & np.isfinite(enh_flat)
            
            if np.sum(valid_mask) < 2:
                print(f"    ⚠️ 有効な値が不足: {np.sum(valid_mask)}")
                return {'enhancement_score': 0.0, 'correlation': 0.0, 'mse': float('inf'), 'snr_db': -100.0}
            
            orig_clean = orig_flat[valid_mask]
            enh_clean = enh_flat[valid_mask]
            
            # MSE計算（数値安定版）
            try:
                diff = orig_clean - enh_clean
                # 差分の範囲チェック
                if np.max(np.abs(diff)) > 1e10:
                    print(f"    ⚠️ 差分が大きすぎます: {np.max(np.abs(diff)):.2e}")
                    diff = np.clip(diff, -1e6, 1e6)  # クリッピング
                
                mse = np.mean(diff ** 2)
                
                # MSEのオーバーフローチェック
                if not np.isfinite(mse) or mse > 1e20:
                    print(f"    ⚠️ MSEオーバーフロー: {mse}")
                    mse = 1e6  # 適度な大きさに制限
                
            except (OverflowError, ValueError) as e:
                print(f"    ❌ MSE計算エラー: {e}")
                mse = 1e6
            
            # 相関係数計算（数値安定版）
            try:
                # 標準偏差の事前チェック
                orig_std = np.std(orig_clean)
                enh_std = np.std(enh_clean)
                
                if orig_std < 1e-10 or enh_std < 1e-10:
                    print(f"    ⚠️ 標準偏差が小さすぎます: orig={orig_std:.2e}, enh={enh_std:.2e}")
                    correlation = 0.0
                else:
                    correlation_matrix = np.corrcoef(orig_clean, enh_clean)
                    correlation = correlation_matrix[0, 1]
                    
                    # 相関係数の有効性チェック
                    if not np.isfinite(correlation):
                        print(f"    ⚠️ 相関係数が無効: {correlation}")
                        correlation = 0.0
                
            except (OverflowError, ValueError, np.linalg.LinAlgError) as e:
                print(f"    ❌ 相関計算エラー: {e}")
                correlation = 0.0
            
            # SNR計算（数値安定版）
            try:
                signal_power = np.mean(orig_clean ** 2)
                
                # 信号パワーのチェック
                if not np.isfinite(signal_power) or signal_power < 1e-20:
                    snr_db = -100.0
                else:
                    noise_power = max(mse, 1e-20)  # ゼロ除算防止
                    snr = signal_power / noise_power
                    
                    if np.isfinite(snr) and snr > 0:
                        snr_db = float(10 * np.log10(snr))
                        # SNRの範囲制限
                        snr_db = np.clip(snr_db, -100.0, 100.0)
                    else:
                        snr_db = -100.0
                
            except (OverflowError, ValueError) as e:
                print(f"    ❌ SNR計算エラー: {e}")
                snr_db = -100.0
            
            # 拡張スコア計算
            enhancement_score = max(0.0, float(correlation)) if np.isfinite(correlation) else 0.0
            
            # 結果の最終検証
            result = {
                'enhancement_score': float(enhancement_score),
                'correlation': float(correlation) if np.isfinite(correlation) else 0.0,
                'mse': float(mse) if np.isfinite(mse) else 1e6,
                'snr_db': float(snr_db) if np.isfinite(snr_db) else -100.0
            }
            
            return result
            
        except Exception as e:
            print(f"    ❌ 品質評価エラー: {e}")
            return {
                'enhancement_score': 0.0,
                'correlation': 0.0,
                'mse': 1e6,
                'snr_db': -100.0
            }


class BackupManager:
    """自動バックアップ管理"""
    
    def __init__(self, config: IntegratedNKATConfig):
        self.config = config
        self.backup_dir = Path(config.backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        print(f"📁 バックアップマネージャー初期化: {self.backup_dir}")
    
    def create_backup(self, file_path: str) -> Optional[str]:
        """ファイルバックアップ作成"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return None
            
            # バックアップファイル名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_path.stem}_{timestamp}{source_path.suffix}"
            backup_path = self.backup_dir / backup_name
            
            # バックアップ実行
            shutil.copy2(source_path, backup_path)
            
            # 古いバックアップ削除
            self._cleanup_old_backups(source_path.stem)
            
            print(f"✅ バックアップ作成: {backup_path.name}")
            return str(backup_path)
            
        except Exception as e:
            print(f"❌ バックアップエラー: {e}")
            return None
    
    def _cleanup_old_backups(self, file_stem: str):
        """古いバックアップの削除"""
        try:
            # 同じファイルのバックアップを取得
            backups = []
            for backup_file in self.backup_dir.glob(f"{file_stem}_*"):
                backups.append(backup_file)
            
            # 作成日時でソート
            backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 最大数を超えた分を削除
            for old_backup in backups[self.config.max_backups:]:
                old_backup.unlink()
                print(f"🗑️ 古いバックアップ削除: {old_backup.name}")
                
        except Exception as e:
            print(f"⚠️ バックアップクリーンアップエラー: {e}")

class PresetManager:
    """プリセット管理"""
    
    def __init__(self, config: IntegratedNKATConfig):
        self.config = config
        self.preset_file = Path(config.preset_file)
        self.settings_file = Path("settings.ini")
        self.presets = self._load_presets()
        self.file_locations = self._load_file_locations()
        print(f"⚙️ プリセットマネージャー初期化")
    
    def _load_presets(self) -> Dict:
        """プリセット読み込み"""
        try:
            if self.preset_file.exists():
                with open(self.preset_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ プリセット読み込みエラー: {e}")
        
        return {
            "default": self.config.to_dict(),
            "high_quality": {**self.config.to_dict(), "max_rank": 16, "tolerance": 1e-8},
            "fast_processing": {**self.config.to_dict(), "max_rank": 4, "tolerance": 1e-4},
            "experimental": {**self.config.to_dict(), "noncommutative_strength": 0.2, "kolmogorov_strength": 0.1}
        }
    
    def save_presets(self):
        """プリセット保存"""
        try:
            with open(self.preset_file, 'w', encoding='utf-8') as f:
                json.dump(self.presets, f, indent=2, ensure_ascii=False)
            print(f"💾 プリセット保存完了")
        except Exception as e:
            print(f"❌ プリセット保存エラー: {e}")
    
    def add_preset(self, name: str, config: IntegratedNKATConfig):
        """プリセット追加"""
        self.presets[name] = config.to_dict()
        if self.config.auto_save_presets:
            self.save_presets()
    
    def get_preset(self, name: str) -> Optional[IntegratedNKATConfig]:
        """プリセット取得"""
        if name in self.presets:
            return IntegratedNKATConfig.from_dict(self.presets[name])
        return None
    
    def _load_file_locations(self) -> Dict:
        """ファイル位置情報読み込み"""
        try:
            config = configparser.ConfigParser()
            if self.settings_file.exists():
                config.read(self.settings_file, encoding='utf-8')
                return dict(config['file_locations']) if 'file_locations' in config else {}
        except Exception as e:
            print(f"⚠️ ファイル位置読み込みエラー: {e}")
        return {}
    
    def save_file_location(self, key: str, path: str):
        """ファイル位置保存"""
        if not self.config.remember_file_locations:
            return
        
        try:
            self.file_locations[key] = path
            
            config = configparser.ConfigParser()
            config['file_locations'] = self.file_locations
            
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                config.write(f)
                
        except Exception as e:
            print(f"❌ ファイル位置保存エラー: {e}")
    
    def get_file_location(self, key: str) -> Optional[str]:
        """ファイル位置取得"""
        return self.file_locations.get(key)


class IntegratedGGUFProcessor:
    """統合GGUF処理システム"""
    
    GGUF_MAGIC = b'GGUF'
    
    def __init__(self, config: IntegratedNKATConfig):
        self.config = config
        self.kolmogorov_op = IntegratedKolmogorovOperator(config)
        self.backup_manager = BackupManager(config)
        self.preset_manager = PresetManager(config)
        
        # 統計
        self.stats = {
            'processed_files': 0,
            'enhanced_tensors': 0,
            'total_enhancement_score': 0.0,
            'processing_time': 0.0
        }
        
        print(f"🧠 統合GGUF処理システム初期化完了")
    
    def process_gguf_file(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """GGUFファイル処理"""
        print(f"\n🌀 GGUFファイル処理開始...")
        print(f"  入力: {Path(input_path).name}")
        
        start_time = time.time()
        
        try:
            # バックアップ作成
            backup_path = None
            if self.config.auto_backup:
                backup_path = self.backup_manager.create_backup(input_path)
            
            # 出力パス決定
            if output_path is None:
                input_path_obj = Path(input_path)
                output_path = str(input_path_obj.parent / f"{input_path_obj.stem}_nkat_enhanced{input_path_obj.suffix}")
            
            # ファイル処理
            success = self._process_gguf_internal(input_path, output_path)
            
            # 統計更新
            processing_time = time.time() - start_time
            self.stats['processing_time'] += processing_time
            
            if success:
                self.stats['processed_files'] += 1
                print(f"✅ 処理完了: {Path(output_path).name}")
                print(f"  処理時間: {processing_time:.2f}秒")
            
            return {
                'success': success,
                'output_path': output_path if success else None,
                'backup_path': backup_path,
                'processing_time': processing_time,
                'stats': self.stats.copy()
            }
            
        except Exception as e:
            print(f"❌ 処理エラー: {e}")
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats.copy()
            }
    
    def _process_gguf_internal(self, input_path: str, output_path: str) -> bool:
        """内部GGUF処理"""
        try:
            with open(input_path, 'rb') as f:
                # ヘッダー読み込み
                magic = f.read(4)
                if magic != self.GGUF_MAGIC:
                    print(f"  ❌ 無効なGGUFマジック")
                    return False
                
                version = struct.unpack('<I', f.read(4))[0]
                tensor_count = struct.unpack('<Q', f.read(8))[0]
                metadata_count = struct.unpack('<Q', f.read(8))[0]
                
                print(f"  📊 GGUF v{version}: {tensor_count}個のテンソル, {metadata_count}個のメタデータ")
                
                # 元のファイルサイズ取得
                original_size = os.path.getsize(input_path)
                print(f"  📏 元ファイルサイズ: {original_size / (1024*1024):.2f} MB")
                
                # 残りのデータ全体を読み込み（メタデータ+テンソル）
                remaining_data = f.read()
                
                # メタデータとテンソルデータを分離
                metadata_data, tensor_data = self._separate_metadata_and_tensors(
                    remaining_data, metadata_count, tensor_count
                )
                
                # テンソルデータ処理（制限を大幅緩和）
                enhanced_data = self._process_tensor_data_full(tensor_data, tensor_count, original_size)
                
                # 拡張ファイル書き込み（元サイズ保持）
                self._write_enhanced_gguf_full(output_path, version, metadata_data, enhanced_data, tensor_count)
                
                return True
                
        except Exception as e:
            print(f"  ❌ 処理エラー: {e}")
            return False
    
    def _separate_metadata_and_tensors(self, data: bytes, metadata_count: int, tensor_count: int) -> Tuple[bytes, bytes]:
        """メタデータとテンソルデータを分離"""
        # 簡単な推定（より正確な分離を実装）
        estimated_metadata_size = min(len(data) // 3, metadata_count * 128)  # メタデータは最大1/3
        
        metadata_data = data[:estimated_metadata_size]
        tensor_data = data[estimated_metadata_size:]
        
        print(f"  📊 分離: メタデータ {len(metadata_data)//1024}KB, テンソル {len(tensor_data)//1024}KB")
        
        return metadata_data, tensor_data
    
    def _process_tensor_data_full(self, data: bytes, tensor_count: int, original_size: int) -> List[bytes]:
        """完全なテンソルデータ処理（サイズ保持版）"""
        print(f"  🔧 完全テンソルデータ処理中... ({len(data) // 1024}KB)")
        
        enhanced_data = []
        
        if len(data) == 0:
            return enhanced_data
        
        # より多くのテンソルを処理（制限大幅緩和）
        max_tensors = min(tensor_count, max(100, tensor_count // 2))  # 最低100個または半分
        chunk_size = max(len(data) // max_tensors, 128)  # 最小128バイト
        
        print(f"  📊 処理予定: {max_tensors}個のテンソル (chunk: {chunk_size}bytes)")
        
        with tqdm(total=max_tensors, desc="フルテンソル処理", leave=False) as pbar:
            for i in range(max_tensors):
                try:
                    start_idx = i * chunk_size
                    end_idx = min(start_idx + chunk_size, len(data))
                    
                    # 最後のチャンクの場合、残り全部を含める
                    if i == max_tensors - 1:
                        end_idx = len(data)
                    
                    tensor_bytes = data[start_idx:end_idx]
                    
                    if len(tensor_bytes) >= 32:  # 最小サイズ緩和
                        # バイト→float32変換
                        float_count = len(tensor_bytes) // 4
                        if float_count >= 4:  # 最小要素数緩和
                            tensor_array = np.frombuffer(
                                tensor_bytes[:float_count * 4], 
                                dtype=np.float32
                            )
                            
                            # より保守的なテンソル整形
                            if len(tensor_array) >= 4:
                                # 元のサイズを可能な限り保持
                                if len(tensor_array) >= 64:
                                    # 8x8 or 4x4x4
                                    side = int(np.sqrt(len(tensor_array)))
                                    if side * side <= len(tensor_array):
                                        tensor_2d = tensor_array[:side*side].reshape(side, side)
                                    else:
                                        tensor_2d = tensor_array.reshape(-1, 4)
                                else:
                                    tensor_2d = tensor_array.reshape(-1, min(4, len(tensor_array)))
                            else:
                                tensor_2d = tensor_array.reshape(-1, 1)
                            
                            # コルモゴロフ拡張（より保守的）
                            result = self.kolmogorov_op.enhance_tensor(tensor_2d)
                            
                            if result['success'] and result['quality_metrics']['enhancement_score'] > 0.05:
                                enhanced_tensor = result['enhanced_tensor']
                                
                                # 元のサイズに近づける
                                enhanced_flat = enhanced_tensor.flatten()
                                if len(enhanced_flat) < len(tensor_array):
                                    # パディングして元サイズに
                                    padded = np.zeros(len(tensor_array), dtype=np.float32)
                                    padded[:len(enhanced_flat)] = enhanced_flat
                                    enhanced_bytes = padded.tobytes()
                                else:
                                    # 切り詰めて元サイズに
                                    enhanced_bytes = enhanced_flat[:len(tensor_array)].astype(np.float32).tobytes()
                                
                                enhanced_data.append(enhanced_bytes)
                                self.stats['enhanced_tensors'] += 1
                                self.stats['total_enhancement_score'] += result['quality_metrics']['enhancement_score']
                                
                                pbar.set_description(f"拡張: {result['quality_metrics']['enhancement_score']:.3f}")
                            else:
                                # 元データをそのまま保持
                                enhanced_data.append(tensor_bytes)
                                pbar.set_description("保持")
                        else:
                            enhanced_data.append(tensor_bytes)
                            pbar.set_description("小さすぎ")
                    else:
                        enhanced_data.append(tensor_bytes)
                        pbar.set_description("スキップ")
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"    ⚠️ テンソル{i+1}エラー: {e}")
                    enhanced_data.append(tensor_bytes if 'tensor_bytes' in locals() else b'')
                    pbar.update(1)
        
        total_enhanced_size = sum(len(data) for data in enhanced_data)
        print(f"  ✅ 処理完了: {len(enhanced_data)}個, 総サイズ {total_enhanced_size // 1024}KB")
        
        return enhanced_data

    def _write_enhanced_gguf_full(self, output_path: str, version: int, metadata_data: bytes, 
                                 enhanced_data: List[bytes], tensor_count: int):
        """完全な拡張GGUFファイル書き込み"""
        print(f"  💾 完全拡張ファイル書き込み中...")
        
        total_tensor_size = sum(len(data) for data in enhanced_data)
        print(f"  📊 書き込み予定: テンソル{len(enhanced_data)}個, {total_tensor_size // 1024}KB")
        
        with open(output_path, 'wb') as f:
            # ヘッダー
            f.write(self.GGUF_MAGIC)
            f.write(struct.pack('<I', version))
            f.write(struct.pack('<Q', len(enhanced_data)))  # 実際のテンソル数
            f.write(struct.pack('<Q', 8))  # 拡張メタデータ数
            
            # 拡張メタデータ
            metadata = [
                ("nkat.integrated.enabled", True),
                ("nkat.enhanced_tensors", self.stats['enhanced_tensors']),
                ("nkat.total_tensors", len(enhanced_data)),
                ("nkat.total_score", self.stats['total_enhancement_score']),
                ("nkat.config.kolmogorov_strength", self.config.kolmogorov_strength),
                ("nkat.config.noncommutative_strength", self.config.noncommutative_strength),
                ("nkat.original_tensor_count", tensor_count),
                ("nkat.system.version", "integrated_full_v1.0")
            ]
            
            for key, value in metadata:
                self._write_metadata_item(f, key, value)
            
            # 元のメタデータを部分的に保持
            if len(metadata_data) > 0:
                f.write(metadata_data[:min(len(metadata_data), 4096)])  # 最大4KB保持
                
                # 必要に応じてパディング
                padding_size = (8 - (f.tell() % 8)) % 8
                if padding_size > 0:
                    f.write(b'\x00' * padding_size)
            
            # 拡張テンソルデータ
            for i, data in enumerate(enhanced_data):
                f.write(data)
                
                # 64bit境界整列
                if self.config.use_64bit_precision:
                    padding = (8 - (len(data) % 8)) % 8
                    if padding > 0:
                        f.write(b'\x00' * padding)
        
        final_size = os.path.getsize(output_path)
        print(f"  ✅ 書き込み完了: {final_size // (1024*1024)}MB")

    def _write_metadata_item(self, f, key, value):
        """メタデータアイテム書き込み"""
        key_bytes = key.encode('utf-8')
        f.write(struct.pack('<Q', len(key_bytes)))
        f.write(key_bytes)
        
        if isinstance(value, bool):
            f.write(struct.pack('<I', 6))  # bool type
            f.write(struct.pack('<?', value))
        elif isinstance(value, int):
            f.write(struct.pack('<I', 4))  # int type
            f.write(struct.pack('<q', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 5))  # float type
            f.write(struct.pack('<d', value))
        else:
            value_bytes = str(value).encode('utf-8')
            f.write(struct.pack('<I', 3))  # string type
            f.write(struct.pack('<Q', len(value_bytes)))
            f.write(value_bytes) 