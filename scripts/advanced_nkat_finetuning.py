#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌀 Advanced NKAT Fine-tuning System
非可換コルモゴロフアーノルド表現理論によるGGUFファインチューニング

Based on:
- Kolmogorov-Arnold Networks (KANs) research
- Non-commutative geometry and quantum field theory
- Tensor Train decomposition for efficient computation
- Regularization techniques to prevent overfitting
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Callable
from tqdm import tqdm
import json
import struct
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.special import comb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RegularizedKANActivation(nn.Module):
    """正則化されたKAN活性化関数（過学習防止版）"""
    
    def __init__(self, 
                 input_dim: int,
                 grid_size: int = 5,
                 spline_order: int = 3,
                 regularization_strength: float = 0.01,
                 noise_injection: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.regularization_strength = regularization_strength
        self.noise_injection = noise_injection
        
        # B-spline基底の制御点
        self.control_points = nn.Parameter(
            torch.randn(input_dim, grid_size + spline_order + 1) * 0.1
        )
        
        # 正則化項用の重み
        self.regularization_weights = nn.Parameter(
            torch.ones(input_dim) * 0.5
        )
        
        # ノイズ注入用パラメータ
        self.noise_scale = nn.Parameter(
            torch.tensor(noise_injection)
        )
        
        print(f"🎯 RegularizedKANActivation: {input_dim}D, grid={grid_size}, order={spline_order}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順方向計算（正則化・ノイズ注入付き）"""
        batch_size = x.size(0)
        
        # ドロップアウト的ノイズ注入（訓練時のみ）
        if self.training:
            noise = torch.randn_like(x) * self.noise_scale
            x = x + noise
        
        # 入力正規化
        x_normalized = torch.tanh(x)  # [-1, 1]に正規化
        
        # B-spline評価
        result = []
        for i in range(self.input_dim):
            # 各次元に対してB-spline計算
            spline_values = self._evaluate_bspline(
                x_normalized[:, i], 
                self.control_points[i]
            )
            
            # 正則化適用
            regularized_values = spline_values * self.regularization_weights[i]
            result.append(regularized_values)
        
        output = torch.stack(result, dim=1)
        
        # 出力クリッピング（数値安定性）
        output = torch.clamp(output, -10.0, 10.0)
        
        return output
    
    def _evaluate_bspline(self, t: torch.Tensor, control_points: torch.Tensor) -> torch.Tensor:
        """B-spline評価（効率化版）"""
        device = t.device
        grid_size = self.grid_size
        k = self.spline_order
        
        # ノット系列生成
        knots = torch.linspace(-1, 1, grid_size + 1, device=device)
        extended_knots = torch.cat([
            torch.full((k,), -1, device=device),
            knots,
            torch.full((k,), 1, device=device)
        ])
        
        # B-spline基底関数の評価
        basis_values = self._compute_bspline_basis(t, extended_knots, k)
        
        # サイズ調整
        min_size = min(basis_values.size(-1), control_points.size(0))
        basis_values = basis_values[..., :min_size]
        control_points_adj = control_points[:min_size]
        
        # 制御点との内積
        result = torch.sum(basis_values * control_points_adj, dim=-1)
        
        return result
    
    def _compute_bspline_basis(self, t: torch.Tensor, knots: torch.Tensor, k: int) -> torch.Tensor:
        """B-spline基底関数計算（数値安定版）"""
        n = len(knots) - k - 1
        if n <= 0:
            # 最小限の基底関数を返す
            return torch.ones(len(t), 1, device=t.device)
        
        basis = torch.zeros(len(t), n, device=t.device)
        
        # 0次基底関数
        for i in range(min(n, len(knots)-1)):
            if i + 1 < len(knots):
                mask = (t >= knots[i]) & (t < knots[i+1])
                basis[mask, i] = 1.0
        
        # 高次基底関数（De Boorのアルゴリズム）
        for degree in range(1, min(k + 1, n)):
            for i in range(n - degree):
                if i + degree < len(knots) and i + degree + 1 < len(knots):
                    denom1 = knots[i + degree] - knots[i]
                    denom2 = knots[i + degree + 1] - knots[i + 1]
                    
                    term1 = 0.0
                    term2 = 0.0
                    
                    if denom1 > 1e-10 and i < basis.size(1):
                        term1 = (t - knots[i]) / denom1 * basis[:, i]
                    
                    if denom2 > 1e-10 and i + 1 < basis.size(1):
                        term2 = (knots[i + degree + 1] - t) / denom2 * basis[:, i + 1]
                    
                    if i < basis.size(1):
                        basis[:, i] = term1 + term2
        
        return basis


class NonCommutativeTensorOperator:
    """非可換テンソル演算子（量子幾何学理論）"""
    
    def __init__(self, 
                 algebra_dim: int = 4,
                 coupling_strength: float = 0.05):
        self.algebra_dim = algebra_dim
        self.coupling_strength = coupling_strength
        
        # 非可換代数生成子（SU(2)拡張）
        self.generators = self._create_algebra_generators()
        
        print(f"🌀 NonCommutativeTensorOperator: {algebra_dim}D algebra, coupling={coupling_strength}")
    
    def to_device(self, device):
        """generatorsを指定されたデバイスに移動"""
        self.generators = [gen.to(device) for gen in self.generators]
        return self
    
    def _create_algebra_generators(self) -> List[torch.Tensor]:
        """非可換代数生成子の生成"""
        # Pauli行列ベースの生成子
        sigma_x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float32)
        sigma_y = torch.tensor([[0., -1.], [1., 0.]], dtype=torch.float32)  # 実数版
        sigma_z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.float32)
        identity = torch.eye(2, dtype=torch.float32)
        
        generators = [sigma_x, sigma_y, sigma_z, identity]
        
        # 高次元への拡張
        if self.algebra_dim > 4:
            for i in range(4, self.algebra_dim):
                # ランダム対称行列生成子
                gen = torch.randn(2, 2, dtype=torch.float32)
                gen = (gen + gen.T) / 2  # 対称化
                gen = gen / torch.norm(gen)  # 正規化
                generators.append(gen)
        
        return generators[:self.algebra_dim]
    
    def apply_noncommutative_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """非可換変換の適用"""
        if tensor.numel() < 4:
            return tensor
        
        device = tensor.device
        
        # generatorsを同じデバイスに移動
        if not self.generators[0].device == device:
            self.generators = [gen.to(device) for gen in self.generators]
        
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        # 2x2ブロック処理
        transformed = torch.zeros_like(flat_tensor)
        
        for i in range(0, len(flat_tensor) - 1, 2):
            # 2要素ベクトル
            vec = flat_tensor[i:i+2]
            
            # 生成子選択
            gen_idx = (i // 2) % len(self.generators)
            generator = self.generators[gen_idx]
            
            # 非可換変換: v' = v + ε[G, v]
            if len(vec) == 2:
                # 交換子計算: [G, v] = Gv - vG
                gv = torch.mv(generator, vec)
                
                # 2要素ベクトルの"右乗算"の代替実装
                vg_approx = vec * generator.diagonal()  # 対角成分による近似
                
                commutator = gv - vg_approx
                
                # 数値安定性のためのクリッピング
                commutator = torch.clamp(commutator, -1.0, 1.0)
                
                transformed_vec = vec + self.coupling_strength * commutator
                transformed[i:i+2] = transformed_vec
            else:
                transformed[i:i+len(vec)] = vec
        
        # 余りの処理
        if len(flat_tensor) % 2 == 1:
            transformed[-1] = flat_tensor[-1]
        
        return transformed.reshape(original_shape)


class QuantumGeometricRegularizer:
    """量子幾何学的正則化器（過学習防止特化）"""
    
    def __init__(self, 
                 curvature_penalty: float = 0.001,
                 spectral_penalty: float = 0.01):
        self.curvature_penalty = curvature_penalty
        self.spectral_penalty = spectral_penalty
        
        print(f"🌊 QuantumGeometricRegularizer: curvature={curvature_penalty}, spectral={spectral_penalty}")
    
    def compute_curvature_penalty(self, activations: torch.Tensor) -> torch.Tensor:
        """曲率ペナルティ計算（Ricci曲率近似）"""
        if activations.dim() < 2:
            return torch.tensor(0.0, device=activations.device)
        
        # 2次微分による曲率近似
        diff1 = torch.diff(activations, dim=1)
        if diff1.size(1) > 1:
            diff2 = torch.diff(diff1, dim=1)
            curvature = torch.mean(diff2.pow(2))
        else:
            curvature = torch.tensor(0.0, device=activations.device)
        
        return self.curvature_penalty * curvature
    
    def compute_spectral_penalty(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """スペクトルペナルティ計算（特異値制約）"""
        if weight_matrix.dim() != 2:
            return torch.tensor(0.0, device=weight_matrix.device)
        
        # SVD計算
        try:
            U, S, V = torch.svd(weight_matrix)
            
            # 最大特異値制約
            max_singular_value = torch.max(S)
            spectral_penalty = F.relu(max_singular_value - 1.0).pow(2)
            
            # 特異値の分散ペナルティ（重要度の均等化）
            variance_penalty = torch.var(S)
            
            total_penalty = spectral_penalty + 0.1 * variance_penalty
            
            return self.spectral_penalty * total_penalty
            
        except RuntimeError:
            # SVD失敗時の回避策
            return torch.tensor(0.0, device=weight_matrix.device)


class AdvancedNKATFinetuner(nn.Module):
    """高度なNKATファインチューニングシステム"""
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dims: List[int] = [512, 256, 128],
                 kan_grid_size: int = 5,
                 kan_spline_order: int = 3,
                 noncommutative_strength: float = 0.05,
                 regularization_strength: float = 0.01):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # KAN層の構築
        self.kan_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            # 従来の線形層
            linear = nn.Linear(prev_dim, hidden_dim)
            
            # KAN活性化関数
            kan_activation = RegularizedKANActivation(
                input_dim=hidden_dim,
                grid_size=kan_grid_size,
                spline_order=kan_spline_order,
                regularization_strength=regularization_strength
            )
            
            self.kan_layers.append(nn.ModuleDict({
                'linear': linear,
                'kan_activation': kan_activation,
                'dropout': nn.Dropout(0.1)
            }))
            
            prev_dim = hidden_dim
        
        # 出力層
        self.output_layer = nn.Linear(prev_dim, input_dim)
        
        # 非可換演算子
        self.noncommutative_op = NonCommutativeTensorOperator(
            coupling_strength=noncommutative_strength
        )
        
        # 量子幾何学的正則化器
        self.quantum_regularizer = QuantumGeometricRegularizer()
        
        # 適応的学習率
        self.adaptive_lr_factor = nn.Parameter(torch.tensor(1.0))
        
        print(f"🚀 AdvancedNKATFinetuner initialized")
        print(f"   Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {input_dim}")
        print(f"   KAN grid: {kan_grid_size}, order: {kan_spline_order}")
        print(f"   Non-commutative strength: {noncommutative_strength}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """順方向計算（完全な変換）"""
        batch_size = x.size(0)
        
        # 入力正規化
        x_normalized = F.layer_norm(x, x.shape[1:])
        
        activations = []
        current = x_normalized
        
        # KAN層の順次適用
        for i, layer_dict in enumerate(self.kan_layers):
            # 線形変換
            linear_output = layer_dict['linear'](current)
            
            # KAN活性化
            kan_output = layer_dict['kan_activation'](linear_output)
            
            # 非可換変換適用
            noncommutative_output = self.noncommutative_op.apply_noncommutative_transform(kan_output)
            
            # ドロップアウト
            current = layer_dict['dropout'](noncommutative_output)
            
            activations.append(current)
            
            # 残差接続（サイズが合う場合）
            if current.shape == x_normalized.shape:
                current = current + 0.1 * x_normalized
        
        # 出力層
        output = self.output_layer(current)
        
        # 最終的な非可換変換
        final_output = self.noncommutative_op.apply_noncommutative_transform(output)
        
        return {
            'reconstructed': final_output,
            'activations': activations,
            'regularization_terms': self._compute_regularization_terms(activations)
        }
    
    def _compute_regularization_terms(self, activations: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """正則化項の計算"""
        total_curvature_penalty = torch.tensor(0.0, device=activations[0].device)
        total_spectral_penalty = torch.tensor(0.0, device=activations[0].device)
        
        # 各層の正則化項
        for activation in activations:
            curvature_penalty = self.quantum_regularizer.compute_curvature_penalty(activation)
            total_curvature_penalty += curvature_penalty
        
        # 重み行列のスペクトル正則化
        for layer_dict in self.kan_layers:
            weight_matrix = layer_dict['linear'].weight
            spectral_penalty = self.quantum_regularizer.compute_spectral_penalty(weight_matrix)
            total_spectral_penalty += spectral_penalty
        
        return {
            'curvature_penalty': total_curvature_penalty,
            'spectral_penalty': total_spectral_penalty,
            'total_penalty': total_curvature_penalty + total_spectral_penalty
        }


class NKATTrainingSystem:
    """NKATトレーニングシステム（過学習防止機能付き）"""
    
    def __init__(self, 
                 model: AdvancedNKATFinetuner,
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0001):
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 最適化器（AdamW - 重み減衰付き）
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学習率スケジューラ
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # 訓練統計
        self.training_stats = {
            'epoch_losses': [],
            'regularization_losses': [],
            'reconstruction_losses': [],
            'validation_losses': [],
            'overfitting_scores': []
        }
        
        print(f"🎓 NKATTrainingSystem initialized")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Weight decay: {weight_decay}")
    
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """1エポックの訓練"""
        self.model.train()
        
        epoch_loss = 0.0
        epoch_reconstruction_loss = 0.0
        epoch_regularization_loss = 0.0
        num_batches = 0
        
        # デバイス取得
        device = next(self.model.parameters()).device
        
        progress_bar = tqdm(dataloader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            # フォワードパス
            if isinstance(batch, (list, tuple)):
                x = batch[0].float().to(device)  # デバイスに移動
            else:
                x = batch.float().to(device)  # デバイスに移動
            
            # ノイズ注入（データ拡張）
            if self.model.training:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            
            # モデル実行
            results = self.model(x)
            
            # 損失計算
            reconstruction_loss = F.mse_loss(results['reconstructed'], x)
            regularization_loss = results['regularization_terms']['total_penalty']
            
            # 総損失
            total_loss = reconstruction_loss + regularization_loss
            
            # バックプロパゲーション
            total_loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 統計更新
            epoch_loss += total_loss.item()
            epoch_reconstruction_loss += reconstruction_loss.item()
            epoch_regularization_loss += regularization_loss.item()
            num_batches += 1
            
            # プログレスバー更新
            progress_bar.set_postfix({
                'Loss': f"{total_loss.item():.6f}",
                'Recon': f"{reconstruction_loss.item():.6f}",
                'Reg': f"{regularization_loss.item():.6f}"
            })
        
        # エポック統計
        avg_loss = epoch_loss / num_batches
        avg_reconstruction_loss = epoch_reconstruction_loss / num_batches
        avg_regularization_loss = epoch_regularization_loss / num_batches
        
        # 統計記録
        self.training_stats['epoch_losses'].append(avg_loss)
        self.training_stats['reconstruction_losses'].append(avg_reconstruction_loss)
        self.training_stats['regularization_losses'].append(avg_regularization_loss)
        
        return {
            'total_loss': avg_loss,
            'reconstruction_loss': avg_reconstruction_loss,
            'regularization_loss': avg_regularization_loss
        }
    
    def validate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """検証"""
        self.model.eval()
        
        total_loss = 0.0
        total_reconstruction_loss = 0.0
        num_batches = 0
        
        # デバイス取得
        device = next(self.model.parameters()).device
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0].float().to(device)  # デバイスに移動
                else:
                    x = batch.float().to(device)  # デバイスに移動
                
                results = self.model(x)
                
                reconstruction_loss = F.mse_loss(results['reconstructed'], x)
                regularization_loss = results['regularization_terms']['total_penalty']
                
                total_loss += (reconstruction_loss + regularization_loss).item()
                total_reconstruction_loss += reconstruction_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_reconstruction_loss = total_reconstruction_loss / num_batches
        
        # 過学習スコア計算
        if len(self.training_stats['epoch_losses']) > 0:
            last_train_loss = self.training_stats['epoch_losses'][-1]
            overfitting_score = avg_loss / last_train_loss if last_train_loss > 0 else 1.0
            self.training_stats['overfitting_scores'].append(overfitting_score)
        
        self.training_stats['validation_losses'].append(avg_loss)
        
        # 学習率調整
        self.scheduler.step(avg_loss)
        
        return {
            'validation_loss': avg_loss,
            'validation_reconstruction_loss': avg_reconstruction_loss
        }
    
    def detect_overfitting(self, patience: int = 5) -> bool:
        """過学習検出"""
        if len(self.training_stats['overfitting_scores']) < patience:
            return False
        
        recent_scores = self.training_stats['overfitting_scores'][-patience:]
        return all(score > 1.1 for score in recent_scores)  # 検証損失が訓練損失の110%を超える
    
    def plot_training_progress(self, save_path: str = None):
        """訓練進捗の可視化"""
        plt.figure(figsize=(15, 10))
        
        # 損失曲線
        plt.subplot(2, 3, 1)
        plt.plot(self.training_stats['epoch_losses'], label='Training Loss', color='blue')
        plt.plot(self.training_stats['validation_losses'], label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # 再構成損失
        plt.subplot(2, 3, 2)
        plt.plot(self.training_stats['reconstruction_losses'], label='Reconstruction Loss', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Reconstruction Loss')
        plt.title('Reconstruction Quality')
        plt.legend()
        plt.grid(True)
        
        # 正則化損失
        plt.subplot(2, 3, 3)
        plt.plot(self.training_stats['regularization_losses'], label='Regularization Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Regularization Loss')
        plt.title('Regularization Terms')
        plt.legend()
        plt.grid(True)
        
        # 過学習スコア
        if self.training_stats['overfitting_scores']:
            plt.subplot(2, 3, 4)
            plt.plot(self.training_stats['overfitting_scores'], label='Overfitting Score', color='purple')
            plt.axhline(y=1.1, color='red', linestyle='--', label='Overfitting Threshold')
            plt.xlabel('Epoch')
            plt.ylabel('Validation/Training Loss Ratio')
            plt.title('Overfitting Detection')
            plt.legend()
            plt.grid(True)
        
        # 学習率
        plt.subplot(2, 3, 5)
        current_lr = self.optimizer.param_groups[0]['lr']
        lr_history = [current_lr] * len(self.training_stats['epoch_losses'])  # 簡単化
        plt.plot(lr_history, label='Learning Rate', color='brown')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training progress saved to: {save_path}")
        
        plt.show()


class GGUFTensorDataGenerator:
    """GGUFテンソルデータ生成器（テスト用）"""
    
    def __init__(self, data_size: int = 1000, tensor_dim: int = 256):
        self.data_size = data_size
        self.tensor_dim = tensor_dim
        
        print(f"🔧 GGUFTensorDataGenerator: {data_size} samples, {tensor_dim}D")
    
    def generate_synthetic_tensor_data(self) -> torch.utils.data.DataLoader:
        """合成テンソルデータ生成"""
        # 構造化された合成データ
        data = []
        
        for i in range(self.data_size):
            # 周期的構造 + ノイズ
            t = np.linspace(0, 4*np.pi, self.tensor_dim)
            
            # 複数の周波数成分
            signal = (np.sin(t) + 0.5*np.sin(3*t) + 0.3*np.sin(5*t) + 
                     0.1*np.random.randn(self.tensor_dim))
            
            # 非線形変換
            signal = np.tanh(signal) + 0.1*signal**2
            
            # 正規化
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            data.append(signal.astype(np.float32))
        
        # PyTorchデータセット
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(np.array(data)))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        
        print(f"✅ Generated {len(data)} synthetic tensor samples")
        return dataloader


def main():
    """メイン実行関数"""
    print("🌀 Advanced NKAT Fine-tuning System")
    print("=" * 80)
    print("📚 Non-Commutative Kolmogorov-Arnold Representation Theory")
    print("🎯 GGUF Tensor Computation with Regularized KAN + Quantum Geometry")
    print("🛡️ Overfitting Prevention & Robust Training")
    print("=" * 80)
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # モデル初期化
    model = AdvancedNKATFinetuner(
        input_dim=256,
        hidden_dims=[512, 256, 128],
        kan_grid_size=5,
        kan_spline_order=3,
        noncommutative_strength=0.05,
        regularization_strength=0.01
    ).to(device)
    
    # 非可換演算子もデバイスに移動
    model.noncommutative_op.to_device(device)
    
    # 訓練システム初期化
    training_system = NKATTrainingSystem(
        model=model,
        learning_rate=0.001,
        weight_decay=0.0001
    )
    
    # データ生成
    data_generator = GGUFTensorDataGenerator(data_size=1000, tensor_dim=256)
    train_dataloader = data_generator.generate_synthetic_tensor_data()
    val_dataloader = data_generator.generate_synthetic_tensor_data()
    
    print(f"\n🚀 Starting NKAT Fine-tuning Training...")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練ループ
    num_epochs = 50
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # 訓練
        train_metrics = training_system.train_epoch(train_dataloader)
        
        # 検証
        val_metrics = training_system.validate(val_dataloader)
        
        # 進捗表示
        print(f"   Train Loss: {train_metrics['total_loss']:.6f}")
        print(f"   Val Loss: {val_metrics['validation_loss']:.6f}")
        print(f"   Reconstruction: {train_metrics['reconstruction_loss']:.6f}")
        print(f"   Regularization: {train_metrics['regularization_loss']:.6f}")
        
        # 早期停止チェック
        if val_metrics['validation_loss'] < best_val_loss:
            best_val_loss = val_metrics['validation_loss']
            patience_counter = 0
            
            # ベストモデル保存
            torch.save(model.state_dict(), 'output/best_nkat_model.pth')
            print(f"   💾 Best model saved (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
        
        # 過学習検出
        if training_system.detect_overfitting():
            print(f"   🚨 Overfitting detected! Stopping training.")
            break
        
        # 早期停止
        if patience_counter >= max_patience:
            print(f"   ⏰ Early stopping triggered (patience: {max_patience})")
            break
    
    print(f"\n🎉 NKAT Fine-tuning Training Completed!")
    print(f"   Best Validation Loss: {best_val_loss:.6f}")
    print(f"   Total Epochs: {epoch+1}")
    
    # 訓練進捗可視化
    os.makedirs('output', exist_ok=True)
    training_system.plot_training_progress('output/nkat_training_progress.png')
    
    # 最終評価
    print(f"\n📊 Final Model Evaluation:")
    model.eval()
    with torch.no_grad():
        # サンプルテンソルでの評価
        sample_data = next(iter(val_dataloader))[0][:1].to(device)  # 1サンプル
        results = model(sample_data)
        
        reconstruction_error = F.mse_loss(results['reconstructed'], sample_data)
        print(f"   Reconstruction Error: {reconstruction_error.item():.6f}")
        
        # 非可換性の測定
        original_norm = torch.norm(sample_data)
        reconstructed_norm = torch.norm(results['reconstructed'])
        print(f"   Norm Preservation: {reconstructed_norm/original_norm:.4f}")
    
    print(f"\n✅ Advanced NKAT Fine-tuning System successfully demonstrated!")
    print(f"🎯 Key Features Implemented:")
    print(f"   ✓ Regularized Kolmogorov-Arnold Networks (Anti-overfitting)")
    print(f"   ✓ Non-commutative Tensor Operations (Quantum Geometry)")
    print(f"   ✓ Quantum Geometric Regularization (Curvature + Spectral)")
    print(f"   ✓ Adaptive Learning Rate & Early Stopping")
    print(f"   ✓ Comprehensive Training Monitoring")
    
    return model, training_system


if __name__ == "__main__":
    # RTX 3080最適化設定
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print(f"🚀 CUDA optimization enabled for RTX 3080")
    
    # メイン実行
    model, training_system = main() 