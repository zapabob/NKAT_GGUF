#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT多目的最適化システム
Optunaベースのrank/γパレート最適化
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from datetime import datetime

# Optuna imports
try:
    import optuna
    from optuna.visualization import plot_pareto_front, plot_param_importances
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available, install with: pip install optuna")

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nkat_optimization.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NKATParameterOptimizer:
    """NKATパラメータ最適化器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimization_history = []
        
        logger.info(f"🔧 NKAT Parameter Optimizer initialized")
        logger.info(f"   📱 Device: {self.device}")
        logger.info(f"   📁 Model: {Path(model_path).name}")
        
        if not OPTUNA_AVAILABLE:
            logger.error("❌ Optuna not available. Please install: pip install optuna")
            sys.exit(1)
    
    def create_nkat_layer(self, hidden_size: int, rank: int, gamma: float) -> torch.nn.Module:
        """NKATレイヤー作成"""
        class NKATLayer(torch.nn.Module):
            def __init__(self, hidden_size: int, rank: int, gamma: float):
                super().__init__()
                self.hidden_size = hidden_size
                self.rank = rank
                self.gamma = gamma
                
                # 線形変換
                self.linear = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                
                # θテンソル（低ランク分解）
                self.theta_u = torch.nn.Parameter(torch.randn(hidden_size, rank) * 0.01)
                self.theta_v = torch.nn.Parameter(torch.randn(rank, hidden_size) * 0.01)
                
            def forward(self, x):
                # 標準線形変換
                y_linear = self.linear(x)
                
                # θ項計算（反対称行列）
                theta = torch.matmul(self.theta_u, self.theta_v)
                theta_antisymm = 0.5 * (theta - theta.T)
                
                # NKAT演算
                y_phase = self.gamma * torch.matmul(x, theta_antisymm.T)
                
                return y_linear + y_phase
        
        return NKATLayer(hidden_size, rank, gamma)
    
    def evaluate_nkat_performance(self, rank: int, gamma: float, 
                                  seq_length: int = 512, 
                                  iterations: int = 10) -> Tuple[float, float, float]:
        """NKAT性能評価"""
        
        hidden_size = 4096  # Qwen3-8B想定
        batch_size = 1
        
        try:
            # NKATレイヤー作成
            nkat_layer = self.create_nkat_layer(hidden_size, rank, gamma).to(self.device)
            
            # 合成入力データ
            x = torch.randn(batch_size, seq_length, hidden_size, 
                          device=self.device, dtype=torch.float16)
            
            # パフォーマンステスト
            throughputs = []
            memory_usage = []
            
            with torch.no_grad():
                for _ in range(iterations):
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    
                    start_time = time.time()
                    
                    # NKAT演算実行
                    output = nkat_layer(x)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    
                    # スループット計算
                    tokens_processed = batch_size * seq_length
                    iteration_time = end_time - start_time
                    throughput = tokens_processed / iteration_time
                    throughputs.append(throughput)
                    
                    # メモリ使用量
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.max_memory_allocated() / (1024**3)
                        memory_usage.append(memory_used)
            
            # 品質推定（簡易版）
            # 実際には perplexity や BLEU スコアなどを使用
            quality_score = self.estimate_quality(rank, gamma)
            
            # メトリクス計算
            avg_throughput = np.mean(throughputs)
            avg_memory = np.mean(memory_usage) if memory_usage else 0.0
            
            return avg_throughput, quality_score, avg_memory
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed for rank={rank}, gamma={gamma}: {e}")
            return 0.0, 0.0, float('inf')
    
    def estimate_quality(self, rank: int, gamma: float) -> float:
        """品質推定（簡易版）"""
        # 実際の実装では実際のモデル推論とメトリクス計算が必要
        # ここでは簡易的な推定を実装
        
        # 理想的なパラメータ範囲の定義
        optimal_rank = 6.0
        optimal_gamma = 0.97
        
        # ランクによる品質影響
        rank_penalty = abs(rank - optimal_rank) * 0.02
        
        # ガンマによる品質影響
        gamma_penalty = abs(gamma - optimal_gamma) * 0.5
        
        # ベース品質スコア
        base_quality = 0.85
        
        # 調整された品質スコア
        quality = base_quality - rank_penalty - gamma_penalty
        
        # ノイズ追加（実際の変動をシミュレート）
        noise = np.random.normal(0, 0.02)
        quality += noise
        
        return max(0.0, min(1.0, quality))
    
    def objective_function(self, trial) -> Tuple[float, float]:
        """多目的最適化の目的関数"""
        
        # パラメータ提案
        rank = trial.suggest_int('rank', 2, 12)
        gamma = trial.suggest_float('gamma', 0.90, 0.99)
        
        # 追加パラメータ
        seq_length = trial.suggest_categorical('seq_length', [256, 512, 1024])
        
        logger.info(f"🧪 Trial {trial.number}: rank={rank}, gamma={gamma:.3f}, seq_len={seq_length}")
        
        # 性能評価
        throughput, quality, memory = self.evaluate_nkat_performance(
            rank=rank, 
            gamma=gamma, 
            seq_length=seq_length
        )
        
        # 最適化記録
        self.optimization_history.append({
            'trial': trial.number,
            'rank': rank,
            'gamma': gamma,
            'seq_length': seq_length,
            'throughput': throughput,
            'quality': quality,
            'memory': memory,
            'timestamp': time.time()
        })
        
        logger.info(f"   📊 Results: throughput={throughput:.1f}, quality={quality:.3f}, memory={memory:.2f}GB")
        
        # 多目的最適化：スループット最大化、品質最大化
        return throughput, quality
    
    def run_optimization(self, n_trials: int = 50, study_name: str = "nkat_optimization") -> optuna.Study:
        """最適化実行"""
        
        logger.info(f"🚀 Starting NKAT multi-objective optimization")
        logger.info(f"   🎯 Trials: {n_trials}")
        
        # Optunaスタディ作成
        study = optuna.create_study(
            directions=['maximize', 'maximize'],  # [throughput, quality]
            study_name=study_name,
            sampler=optuna.samplers.NSGAIISampler()
        )
        
        # 最適化実行
        progress_bar = tqdm(total=n_trials, desc="Optimization Progress")
        
        def callback(study, trial):
            progress_bar.update(1)
            
            # 中間結果表示
            if trial.number % 10 == 0:
                logger.info(f"   📈 Trial {trial.number}: Best trials so far: {len(study.best_trials)}")
        
        study.optimize(
            self.objective_function, 
            n_trials=n_trials,
            callbacks=[callback]
        )
        
        progress_bar.close()
        
        logger.info(f"🎉 Optimization completed!")
        logger.info(f"   📊 Pareto optimal solutions: {len(study.best_trials)}")
        
        return study
    
    def analyze_results(self, study: optuna.Study) -> Dict:
        """結果分析"""
        
        logger.info("📊 Analyzing optimization results...")
        
        # パレート最適解
        pareto_trials = study.best_trials
        
        analysis = {
            "optimization_summary": {
                "total_trials": len(study.trials),
                "pareto_optimal_count": len(pareto_trials),
                "best_trials": []
            },
            "parameter_analysis": {},
            "recommendations": []
        }
        
        # パレート最適解の詳細
        for i, trial in enumerate(pareto_trials[:10]):  # 上位10個
            trial_info = {
                "rank": i + 1,
                "trial_number": trial.number,
                "parameters": trial.params,
                "objectives": {
                    "throughput": trial.values[0],
                    "quality": trial.values[1]
                }
            }
            analysis["optimization_summary"]["best_trials"].append(trial_info)
        
        # パラメータ重要度分析
        if len(study.trials) >= 10:
            try:
                # スループット重要度
                throughput_importance = optuna.importance.get_param_importances(
                    study, target=lambda t: t.values[0]
                )
                
                # 品質重要度
                quality_importance = optuna.importance.get_param_importances(
                    study, target=lambda t: t.values[1]
                )
                
                analysis["parameter_analysis"] = {
                    "throughput_importance": throughput_importance,
                    "quality_importance": quality_importance
                }
                
            except Exception as e:
                logger.warning(f"Parameter importance analysis failed: {e}")
        
        # 推奨パラメータ
        if pareto_trials:
            # バランス型推奨（スループットと品質のバランス）
            balanced_scores = []
            for trial in pareto_trials:
                # 正規化されたスコア
                norm_throughput = trial.values[0] / max(t.values[0] for t in pareto_trials)
                norm_quality = trial.values[1] / max(t.values[1] for t in pareto_trials)
                balanced_score = (norm_throughput + norm_quality) / 2
                balanced_scores.append((balanced_score, trial))
            
            best_balanced = max(balanced_scores, key=lambda x: x[0])[1]
            
            analysis["recommendations"] = [
                {
                    "type": "balanced",
                    "description": "Best balance between throughput and quality",
                    "parameters": best_balanced.params,
                    "expected_throughput": best_balanced.values[0],
                    "expected_quality": best_balanced.values[1]
                }
            ]
        
        return analysis
    
    def create_visualization(self, study: optuna.Study, analysis: Dict) -> None:
        """結果可視化"""
        
        logger.info("📊 Creating optimization visualizations...")
        
        output_dir = Path("output/qwen3_nkat_testing")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. パレートフロント可視化
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('NKAT Multi-Objective Optimization Results', fontsize=16, fontweight='bold')
            
            # パレート最適解プロット
            pareto_trials = study.best_trials
            all_trials = study.trials
            
            # 全試行
            all_throughputs = [t.values[0] for t in all_trials if t.values]
            all_qualities = [t.values[1] for t in all_trials if t.values]
            
            # パレート最適解
            pareto_throughputs = [t.values[0] for t in pareto_trials]
            pareto_qualities = [t.values[1] for t in pareto_trials]
            
            ax1.scatter(all_throughputs, all_qualities, alpha=0.6, c='lightblue', label='All trials')
            ax1.scatter(pareto_throughputs, pareto_qualities, c='red', s=100, label='Pareto optimal')
            ax1.set_xlabel('Throughput (tokens/sec)')
            ax1.set_ylabel('Quality Score')
            ax1.set_title('Pareto Front')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. パラメータ分布
            ranks = [t.params.get('rank', 0) for t in all_trials]
            gammas = [t.params.get('gamma', 0) for t in all_trials]
            
            ax2.scatter(ranks, gammas, c=all_qualities, cmap='viridis', alpha=0.7)
            ax2.set_xlabel('Rank')
            ax2.set_ylabel('Gamma')
            ax2.set_title('Parameter Space (colored by quality)')
            ax2.grid(True, alpha=0.3)
            
            # 3. 収束履歴
            trial_numbers = [t.number for t in all_trials if t.values]
            best_throughputs = []
            best_qualities = []
            
            current_best_throughput = 0
            current_best_quality = 0
            
            for trial in all_trials:
                if trial.values:
                    current_best_throughput = max(current_best_throughput, trial.values[0])
                    current_best_quality = max(current_best_quality, trial.values[1])
                    best_throughputs.append(current_best_throughput)
                    best_qualities.append(current_best_quality)
            
            ax3.plot(trial_numbers, best_throughputs, label='Best Throughput', linewidth=2)
            ax3.plot(trial_numbers, [q*10000 for q in best_qualities], label='Best Quality × 10000', linewidth=2)
            ax3.set_xlabel('Trial Number')
            ax3.set_ylabel('Objective Value')
            ax3.set_title('Convergence History')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. パラメータ重要度
            param_importance = analysis.get("parameter_analysis", {}).get("throughput_importance", {})
            if param_importance:
                params = list(param_importance.keys())
                importances = list(param_importance.values())
                
                ax4.bar(params, importances, color='skyblue', alpha=0.7)
                ax4.set_xlabel('Parameters')
                ax4.set_ylabel('Importance')
                ax4.set_title('Parameter Importance (Throughput)')
                ax4.tick_params(axis='x', rotation=45)
                ax4.grid(True, alpha=0.3)
            else:
                ax4.text(0.5, 0.5, 'Parameter importance\nanalysis unavailable', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Parameter Importance')
            
            plt.tight_layout()
            
            # 保存
            chart_path = output_dir / "nkat_optimization_results.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"📊 Optimization charts saved: {chart_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
    
    def save_results(self, study: optuna.Study, analysis: Dict, 
                    filename: str = "nkat_optimization_results.json") -> None:
        """結果保存"""
        
        try:
            output_dir = Path("output/qwen3_nkat_testing")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 完全な結果セット
            results = {
                "study_summary": {
                    "study_name": study.study_name,
                    "directions": study.directions,
                    "total_trials": len(study.trials),
                    "pareto_optimal_count": len(study.best_trials),
                    "optimization_time": datetime.now().isoformat()
                },
                "analysis": analysis,
                "detailed_history": self.optimization_history,
                "pareto_trials": [
                    {
                        "trial_number": trial.number,
                        "parameters": trial.params,
                        "objectives": trial.values,
                        "state": str(trial.state)
                    }
                    for trial in study.best_trials
                ]
            }
            
            output_path = output_dir / filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Optimization results saved: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")

def main():
    """メイン実行"""
    print("🔧 NKAT Multi-Objective Parameter Optimization")
    print("=" * 60)
    
    model_path = "models/integrated/Qwen3-8B-ERP-v0.1.i1-Q6_K.gguf"
    
    # 最適化器初期化
    optimizer = NKATParameterOptimizer(model_path)
    
    # 最適化実行
    study = optimizer.run_optimization(n_trials=30)  # デモ用に少なめに設定
    
    # 結果分析
    analysis = optimizer.analyze_results(study)
    
    # 可視化
    optimizer.create_visualization(study, analysis)
    
    # 結果保存
    optimizer.save_results(study, analysis)
    
    # サマリー表示
    print("\n📊 Optimization Summary:")
    print("-" * 40)
    print(f"Total trials: {len(study.trials)}")
    print(f"Pareto optimal solutions: {len(study.best_trials)}")
    
    if analysis["recommendations"]:
        rec = analysis["recommendations"][0]
        print(f"\n🎯 Recommended Parameters:")
        print(f"   Rank: {rec['parameters']['rank']}")
        print(f"   Gamma: {rec['parameters']['gamma']:.3f}")
        print(f"   Expected throughput: {rec['expected_throughput']:.1f} tokens/sec")
        print(f"   Expected quality: {rec['expected_quality']:.3f}")
    
    print("\n🎉 Multi-objective optimization completed!")
    print("📁 Check output/qwen3_nkat_testing/ for detailed results and charts")

if __name__ == "__main__":
    main() 