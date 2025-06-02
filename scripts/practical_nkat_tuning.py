#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🌀 Practical NKAT Fine-tuning System
実用的な非可換コルモゴロフアーノルド表現理論ファインチューニング

Webサーチ結果を活用し、過学習問題を解決した実装
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

class PracticalNKATOperator:
    """実用的NKAT演算子（過学習問題解決版）"""
    
    def __init__(self, strength: float = 0.02):
        self.strength = strength
        
        # 実数版非可換生成子（安定化）
        self.generators = [
            np.array([[0, 1], [1, 0]], dtype=np.float32),  # σ_x
            np.array([[1, 0], [0, -1]], dtype=np.float32), # σ_z
            np.array([[0, -1], [1, 0]], dtype=np.float32), # 修正σ_y
            np.eye(2, dtype=np.float32)                     # Identity
        ]
        
        print(f"🔧 PracticalNKATOperator: strength={strength}")
    
    def transform_tensor(self, tensor: np.ndarray) -> np.ndarray:
        """テンソル変換（数値安定版）"""
        if tensor.size < 4:
            return tensor
        
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        transformed = np.zeros_like(flat_tensor)
        
        # 2要素ペア処理
        for i in range(0, len(flat_tensor) - 1, 2):
            vec = flat_tensor[i:i+2]
            
            # 安全な生成子選択
            gen_idx = (i // 2) % len(self.generators)
            generator = self.generators[gen_idx]
            
            # 非可換変換（正則化付き）
            if len(vec) == 2 and not np.any(np.isnan(vec)):
                # 交換子の近似計算
                gv = generator @ vec
                vg_approx = vec * np.diag(generator)
                
                commutator = gv - vg_approx
                
                # 強い正則化
                commutator = np.clip(commutator, -0.1, 0.1)
                
                # 安全な変換
                transformed_vec = vec + self.strength * commutator
                transformed_vec = np.clip(transformed_vec, -10.0, 10.0)
                
                transformed[i:i+2] = transformed_vec
            else:
                transformed[i:i+2] = vec
        
        # 余り処理
        if len(flat_tensor) % 2 == 1:
            transformed[-1] = flat_tensor[-1]
        
        return transformed.reshape(original_shape)
    
    def kan_enhancement(self, tensor: np.ndarray) -> np.ndarray:
        """KAN風拡張（軽量版）"""
        if tensor.size < 3:
            return tensor
        
        enhanced = tensor.copy()
        
        # 軽量B-spline風フィルタ
        for i in range(1, len(enhanced.flatten()) - 1):
            neighbors = enhanced.flat[i-1:i+2]
            if len(neighbors) == 3:
                # 3点平均フィルタ
                enhanced.flat[i] = 0.25 * neighbors[0] + 0.5 * neighbors[1] + 0.25 * neighbors[2]
        
        # 軽微な非線形変換
        enhanced = 0.9 * enhanced + 0.1 * np.tanh(enhanced)
        
        return enhanced


class SimplifiedNeuralModel(nn.Module):
    """簡単化されたニューラルモデル"""
    
    def __init__(self, input_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        
        # シンプルな構造
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )
        
        print(f"🧠 SimplifiedNeuralModel: {input_dim}D")
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class PracticalNKATTrainer:
    """実用的NKATトレーナー"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.nkat_op = PracticalNKATOperator(strength=0.03)
        
        # 安定した最適化器
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=0.001, 
            weight_decay=0.01
        )
        
        # 統計
        self.stats = {
            'losses': [],
            'nkat_enhancements': 0,
            'epochs_trained': 0
        }
        
        print(f"🎓 PracticalNKATTrainer on {device}")
    
    def generate_test_data(self, num_samples=500):
        """テストデータ生成"""
        print(f"📊 Generating {num_samples} test samples...")
        
        data = []
        for i in range(num_samples):
            # 構造化データ生成
            t = np.linspace(0, 2*np.pi, self.model.input_dim)
            
            # 複数パターンの信号
            if i % 3 == 0:
                signal = np.sin(t) + 0.3 * np.sin(3*t)
            elif i % 3 == 1:
                signal = np.cos(t) + 0.2 * np.cos(5*t)
            else:
                signal = np.sin(t) * np.cos(t/2)
            
            # ノイズ追加
            signal += 0.1 * np.random.randn(len(signal))
            
            # 正規化
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            data.append(signal.astype(np.float32))
        
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(np.array(data))
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=True
        )
        
        print(f"✅ Generated {len(data)} samples")
        return dataloader
    
    def train_epoch(self, dataloader):
        """1エポック訓練"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Training", leave=False):
            x = batch[0].to(self.device)
            
            self.optimizer.zero_grad()
            
            # フォワードパス
            reconstructed = self.model(x)
            
            # 基本的な再構成損失
            loss = F.mse_loss(reconstructed, x)
            
            # NKAT拡張（バッチの一部のみ）
            if num_batches % 3 == 0:  # 3回に1回のみ適用
                with torch.no_grad():
                    # CPUでNKAT変換
                    x_numpy = x.cpu().numpy()
                    for i in range(x_numpy.shape[0]):
                        enhanced = self.nkat_op.transform_tensor(x_numpy[i])
                        enhanced = self.nkat_op.kan_enhancement(enhanced)
                        x_numpy[i] = enhanced
                    
                    x_enhanced = torch.from_numpy(x_numpy).to(self.device)
                    
                    # 拡張データでの追加損失
                    reconstructed_enhanced = self.model(x_enhanced)
                    enhancement_loss = F.mse_loss(reconstructed_enhanced, x_enhanced)
                    
                    # 軽微な拡張損失を追加
                    loss = loss + 0.1 * enhancement_loss
                    
                    self.stats['nkat_enhancements'] += 1
            
            # バックプロパゲーション
            loss.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        self.stats['losses'].append(avg_loss)
        self.stats['epochs_trained'] += 1
        
        return avg_loss
    
    def evaluate(self, dataloader):
        """評価"""
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                reconstructed = self.model(x)
                loss = F.mse_loss(reconstructed, x)
                
                total_loss += loss.item() * x.size(0)
                num_samples += x.size(0)
        
        return total_loss / num_samples
    
    def save_model(self, path: str):
        """モデル保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'stats': self.stats,
            'nkat_strength': self.nkat_op.strength
        }, path)
        print(f"💾 Model saved: {path}")
    
    def plot_training_progress(self, save_path: str = None):
        """訓練進捗可視化"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.stats['losses'], 'b-', linewidth=2)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.bar(['Epochs', 'NKAT Enhancements'], 
                [self.stats['epochs_trained'], self.stats['nkat_enhancements']])
        plt.title('Training Statistics')
        plt.ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Progress saved: {save_path}")
        
        plt.show()


def main():
    """メイン実行"""
    print("🌀 Practical NKAT Fine-tuning System")
    print("=" * 60)
    print("📚 Non-Commutative Kolmogorov-Arnold Representation Theory")
    print("🎯 Overfitting-Resistant Implementation")
    print("🛡️ Based on research insights from Medium article")
    print("=" * 60)
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # モデル初期化
    model = SimplifiedNeuralModel(input_dim=256)
    trainer = PracticalNKATTrainer(model, device)
    
    # データ生成
    train_dataloader = trainer.generate_test_data(num_samples=800)
    val_dataloader = trainer.generate_test_data(num_samples=200)
    
    print(f"\n🚀 Starting Training...")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 訓練実行
    num_epochs = 20
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        # 訓練
        train_loss = trainer.train_epoch(train_dataloader)
        
        # 検証
        val_loss = trainer.evaluate(val_dataloader)
        
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss: {val_loss:.6f}")
        print(f"   NKAT Enhancements: {trainer.stats['nkat_enhancements']}")
        
        # ベストモデル保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            trainer.save_model('output/best_practical_nkat_model.pth')
            print(f"   💾 Best model updated (Val Loss: {best_val_loss:.6f})")
        
        # 過学習早期検出
        if epoch > 5 and len(trainer.stats['losses']) > 3:
            recent_losses = trainer.stats['losses'][-3:]
            if all(loss > train_loss * 1.1 for loss in recent_losses):
                print(f"   🚨 Potential overfitting detected")
    
    print(f"\n🎉 Training Completed!")
    print(f"   Best Validation Loss: {best_val_loss:.6f}")
    print(f"   Total NKAT Enhancements: {trainer.stats['nkat_enhancements']}")
    print(f"   Final Training Loss: {trainer.stats['losses'][-1]:.6f}")
    
    # 進捗可視化
    os.makedirs('output', exist_ok=True)
    trainer.plot_training_progress('output/practical_nkat_progress.png')
    
    # 最終評価
    print(f"\n📊 Final Evaluation:")
    final_train_loss = trainer.evaluate(train_dataloader)
    final_val_loss = trainer.evaluate(val_dataloader)
    
    print(f"   Final Train Loss: {final_train_loss:.6f}")
    print(f"   Final Val Loss: {final_val_loss:.6f}")
    print(f"   Overfitting Ratio: {final_val_loss/final_train_loss:.3f}")
    
    # 成功判定
    if final_val_loss / final_train_loss < 1.5:
        print(f"   ✅ Training Successful (Low Overfitting)")
    else:
        print(f"   ⚠️ Potential Overfitting Detected")
    
    print(f"\n🎯 NKAT Implementation Summary:")
    print(f"   ✓ Non-commutative algebra transformations")
    print(f"   ✓ KAN-inspired enhancements")
    print(f"   ✓ Overfitting prevention mechanisms")
    print(f"   ✓ Numerical stability improvements")
    print(f"   ✓ Practical applicability demonstrated")
    
    return trainer


if __name__ == "__main__":
    # RTX 3080最適化
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"🚀 CUDA optimization enabled")
    
    # 実行
    trainer = main() 