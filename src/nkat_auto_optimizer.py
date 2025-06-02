#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Auto Optimizer
自動パラメータ最適化とTPEスコア最大化
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
import logging
from tqdm import tqdm
import optuna
import subprocess
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # GUI無効化
plt.rcParams['font.size'] = 10
import warnings
warnings.filterwarnings('ignore')

# ローカルモジュール
from nkat_gguf_converter import NKATGGUFConverter, calculate_tpe_score
from nkat_inference_engine import NKATInferenceEngine

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

class NKATAutoOptimizer:
    """NKAT自動最適化器"""
    
    def __init__(self, base_model_path: str, output_dir: str = "output/optimized"):
        self.base_model_path = base_model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimization_history = []
        self.best_params = None
        self.best_score = -np.inf
        
        logger.info(f"🚀 NKAT Auto Optimizer 初期化")
        logger.info(f"   📁 ベースモデル: {base_model_path}")
        logger.info(f"   📁 出力ディレクトリ: {output_dir}")
    
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna最適化関数"""
        # パラメータ探索範囲
        theta_rank = trial.suggest_int('theta_rank', 2, 8, step=2)
        theta_gamma = trial.suggest_float('theta_gamma', 0.90, 0.99, step=0.01)
        
        logger.info(f"🧪 Trial {trial.number}: rank={theta_rank}, gamma={theta_gamma}")
        
        try:
            # NKAT-GGUF変換
            output_path = self.output_dir / f"trial_{trial.number}_rank{theta_rank}_gamma{theta_gamma:.2f}.nkat"
            converter = NKATGGUFConverter(theta_rank, theta_gamma)
            
            success = converter.convert_to_nkat_gguf(
                self.base_model_path, 
                str(output_path)
            )
            
            if not success:
                logger.warning(f"   ❌ 変換失敗: trial {trial.number}")
                return -1000.0
            
            # 推論性能評価
            engine = NKATInferenceEngine(str(output_path))
            if not engine.load_model():
                logger.warning(f"   ❌ モデル読み込み失敗: trial {trial.number}")
                return -1000.0
            
            # ベンチマーク実行
            benchmark_results = engine.benchmark_inference(
                sequence_length=256,  # 短縮して高速化
                num_iterations=20
            )
            
            # ベースライン比較
            comparison = engine.compare_with_baseline(256)
            
            # 疑似perplexity計算（実際のモデルでは推論で計算）
            mock_perplexity = self._estimate_perplexity(theta_rank, theta_gamma)
            
            # TPEスコア計算
            lambda_theta = theta_rank * 0.05  # rankに比例するλ
            tpe_score = calculate_tpe_score(mock_perplexity, lambda_theta)
            
            # 結果記録
            result = {
                "trial": trial.number,
                "theta_rank": theta_rank,
                "theta_gamma": theta_gamma,
                "tokens_per_second": benchmark_results["tokens_per_second"],
                "overhead_percentage": comparison["overhead_percentage"],
                "estimated_perplexity": mock_perplexity,
                "tpe_score": tpe_score,
                "lambda_theta": lambda_theta
            }
            
            self.optimization_history.append(result)
            
            logger.info(f"   📊 trial {trial.number}: TPE={tpe_score:.4f}")
            logger.info(f"      tok/s={benchmark_results['tokens_per_second']:.1f}")
            logger.info(f"      overhead={comparison['overhead_percentage']:+.1f}%")
            logger.info(f"      est_ppl={mock_perplexity:.3f}")
            
            # 最良結果更新
            if tpe_score > self.best_score:
                self.best_score = tpe_score
                self.best_params = result.copy()
                logger.info(f"   🏆 新ベスト! TPE={tpe_score:.4f}")
            
            # 一時ファイル削除（ディスク容量節約）
            if output_path.exists():
                output_path.unlink()
            
            return tpe_score
            
        except Exception as e:
            logger.error(f"   ❌ trial {trial.number} 実行エラー: {e}")
            return -1000.0
    
    def _estimate_perplexity(self, theta_rank: int, theta_gamma: float) -> float:
        """疑似perplexity推定（実際の実装では実推論が必要）"""
        # 簡易モデル：rank↑でperplexity↓、gamma最適値周辺で最小
        base_ppl = 6.5
        rank_effect = -0.03 * theta_rank  # rank高いほど改善
        gamma_effect = -0.2 * np.exp(-((theta_gamma - 0.97) / 0.02) ** 2)  # 0.97付近で最良
        noise = np.random.normal(0, 0.05)  # ランダムノイズ
        
        estimated_ppl = base_ppl + rank_effect + gamma_effect + noise
        return max(estimated_ppl, 3.0)  # 最小値制限
    
    def run_optimization(self, n_trials: int = 50) -> Dict:
        """最適化実行"""
        logger.info(f"🔍 NKAT最適化開始: {n_trials} trials")
        
        # Optuna study作成
        study = optuna.create_study(
            direction='maximize',
            study_name='nkat_optimization',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        # 最適化実行
        study.optimize(self.objective, n_trials=n_trials)
        
        # 結果解析
        best_trial = study.best_trial
        logger.info(f"🏆 最適化完了!")
        logger.info(f"   🎯 ベストスコア: {best_trial.value:.4f}")
        logger.info(f"   ⚙️  ベストパラメータ:")
        for key, value in best_trial.params.items():
            logger.info(f"      {key}: {value}")
        
        # 結果保存
        self._save_optimization_results(study)
        
        # 最適モデル生成
        best_model_path = self._generate_optimal_model(best_trial.params)
        
        return {
            "best_params": best_trial.params,
            "best_score": best_trial.value,
            "best_model_path": best_model_path,
            "optimization_history": self.optimization_history,
            "study": study
        }
    
    def _save_optimization_results(self, study: optuna.Study):
        """最適化結果保存"""
        # 履歴をJSON保存
        history_file = self.output_dir / "optimization_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.optimization_history, f, indent=2, ensure_ascii=False)
        
        # Optuna結果保存
        study_file = self.output_dir / "optuna_study.json"
        with open(study_file, 'w', encoding='utf-8') as f:
            study_summary = {
                "best_trial": {
                    "number": study.best_trial.number,
                    "value": study.best_trial.value,
                    "params": study.best_trial.params
                },
                "n_trials": len(study.trials),
                "direction": study.direction.name
            }
            json.dump(study_summary, f, indent=2, ensure_ascii=False)
        
        # 可視化グラフ生成
        self._plot_optimization_results(study)
        
        logger.info(f"📄 結果保存完了:")
        logger.info(f"   📊 履歴: {history_file}")
        logger.info(f"   🔬 Study: {study_file}")
    
    def _plot_optimization_results(self, study: optuna.Study):
        """最適化結果可視化"""
        try:
            # 1. TPEスコア推移
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            values = [trial.value for trial in study.trials if trial.value is not None]
            plt.plot(values, 'b-', alpha=0.7)
            plt.title('TPE Score Progress', fontsize=10)
            plt.xlabel('Trial', fontsize=8)
            plt.ylabel('TPE Score', fontsize=8)
            plt.grid(True, alpha=0.3)
            
            # 2. パラメータ分布
            plt.subplot(2, 2, 2)
            ranks = [h["theta_rank"] for h in self.optimization_history]
            gammas = [h["theta_gamma"] for h in self.optimization_history]
            scores = [h["tpe_score"] for h in self.optimization_history]
            
            scatter = plt.scatter(ranks, gammas, c=scores, cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label='TPE Score')
            plt.title('Parameter Space Exploration', fontsize=10)
            plt.xlabel('Theta Rank', fontsize=8)
            plt.ylabel('Theta Gamma', fontsize=8)
            
            # 3. Perplexity vs Speed
            plt.subplot(2, 2, 3)
            ppls = [h["estimated_perplexity"] for h in self.optimization_history]
            speeds = [h["tokens_per_second"] for h in self.optimization_history]
            
            plt.scatter(ppls, speeds, c=scores, cmap='plasma', alpha=0.7)
            plt.colorbar(label='TPE Score')
            plt.title('Perplexity vs Speed', fontsize=10)
            plt.xlabel('Estimated Perplexity', fontsize=8)
            plt.ylabel('Tokens/sec', fontsize=8)
            
            # 4. オーバーヘッド分布
            plt.subplot(2, 2, 4)
            overheads = [h["overhead_percentage"] for h in self.optimization_history]
            plt.hist(overheads, bins=20, alpha=0.7, color='orange')
            plt.title('Speed Overhead Distribution', fontsize=10)
            plt.xlabel('Overhead (%)', fontsize=8)
            plt.ylabel('Frequency', fontsize=8)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = self.output_dir / "optimization_results.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"   📈 可視化: {plot_file}")
            
        except Exception as e:
            logger.warning(f"⚠️  可視化失敗: {e}")
    
    def _generate_optimal_model(self, best_params: Dict) -> str:
        """最適パラメータでモデル生成"""
        logger.info(f"🏭 最適モデル生成中...")
        
        output_path = self.output_dir / f"optimal_rank{best_params['theta_rank']}_gamma{best_params['theta_gamma']:.2f}.nkat"
        
        converter = NKATGGUFConverter(
            best_params['theta_rank'], 
            best_params['theta_gamma']
        )
        
        success = converter.convert_to_nkat_gguf(
            self.base_model_path,
            str(output_path)
        )
        
        if success:
            logger.info(f"✅ 最適モデル生成完了: {output_path}")
            
            # 設定ファイル保存
            config_file = output_path.with_suffix('.json')
            config = {
                "optimal_parameters": best_params,
                "base_model": self.base_model_path,
                "optimization_timestamp": str(torch.tensor(0).item()),  # ダミー
                "usage_command": f"py -3 nkat_inference_engine.py -m {output_path} --theta-gamma {best_params['theta_gamma']} --benchmark"
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            return str(output_path)
        else:
            logger.error(f"❌ 最適モデル生成失敗")
            return ""

def run_quick_optimization(model_path: str, output_dir: str = "output/quick") -> Dict:
    """クイック最適化（少数trial）"""
    logger.info(f"⚡ クイック最適化モード")
    
    optimizer = NKATAutoOptimizer(model_path, output_dir)
    results = optimizer.run_optimization(n_trials=12)  # 少数trial
    
    return results

def run_full_optimization(model_path: str, output_dir: str = "output/full") -> Dict:
    """完全最適化（多数trial）"""
    logger.info(f"🔬 完全最適化モード")
    
    optimizer = NKATAutoOptimizer(model_path, output_dir)
    results = optimizer.run_optimization(n_trials=100)  # 多数trial
    
    return results

def main():
    parser = argparse.ArgumentParser(description="NKAT Auto Optimizer")
    parser.add_argument("--model", "-m", required=True, help="ベースGGUFモデル")
    parser.add_argument("--output-dir", "-o", default="output/optimized", help="出力ディレクトリ")
    parser.add_argument("--mode", choices=["quick", "full", "custom"], default="quick", help="最適化モード")
    parser.add_argument("--trials", type=int, default=50, help="trial数（customモード用）")
    parser.add_argument("--target-rank", type=int, help="特定rank指定")
    parser.add_argument("--target-gamma", type=float, help="特定gamma指定")
    
    args = parser.parse_args()
    
    # モデル存在確認
    if not os.path.exists(args.model):
        logger.error(f"❌ モデルファイルが見つかりません: {args.model}")
        sys.exit(1)
    
    # 特定パラメータ指定の場合
    if args.target_rank and args.target_gamma:
        logger.info(f"🎯 特定パラメータで実行: rank={args.target_rank}, gamma={args.target_gamma}")
        
        converter = NKATGGUFConverter(args.target_rank, args.target_gamma)
        output_path = Path(args.output_dir) / f"target_rank{args.target_rank}_gamma{args.target_gamma:.2f}.nkat"
        
        success = converter.convert_to_nkat_gguf(args.model, str(output_path))
        if success:
            print(f"✅ 完了: {output_path}")
        sys.exit(0)
    
    # 最適化実行
    if args.mode == "quick":
        results = run_quick_optimization(args.model, args.output_dir)
    elif args.mode == "full":
        results = run_full_optimization(args.model, args.output_dir)
    else:  # custom
        optimizer = NKATAutoOptimizer(args.model, args.output_dir)
        results = optimizer.run_optimization(args.trials)
    
    # 結果表示
    print(f"\n🎉 NKAT最適化完了!")
    print(f"🏆 ベストパラメータ:")
    for key, value in results["best_params"].items():
        print(f"   {key}: {value}")
    print(f"📊 ベストTPEスコア: {results['best_score']:.4f}")
    print(f"📁 最適モデル: {results['best_model_path']}")
    
    print(f"\n🚀 使用例:")
    print(f"py -3 nkat_inference_engine.py -m {results['best_model_path']} --benchmark")

if __name__ == "__main__":
    main() 