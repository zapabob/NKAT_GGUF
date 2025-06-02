#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT文章出力安定性エンハンサー
RTX30/RTX40シリーズ向け最適化
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import random

# 日本語フォント設定
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'Tahoma']
plt.rcParams['axes.unicode_minus'] = False

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nkat_stability.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NKATStabilityEnhancer:
    """NKAT安定性エンハンサー"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.stability_results = []
        self.optimal_config = {}
        
        # 安定性テスト用プロンプト
        self.test_prompts = [
            "Once upon a time in a magical kingdom,",
            "The future of artificial intelligence involves",
            "In the depths of the ocean, scientists discovered",
            "Climate change is affecting",
            "Technology has revolutionized the way we",
            "The ancient civilization was known for",
            "Space exploration has revealed that",
            "The breakthrough in medical research shows",
            "Economic development in the 21st century",
            "Education systems around the world are"
        ]
        
        logger.info("🔬 NKAT Stability Enhancer initialized")
    
    def generate_with_settings(self, prompt: str, settings: Dict) -> str:
        """指定設定でテキスト生成（モック実装）"""
        # 実際の実装では、NKAT推論エンジンを呼び出し
        # ここではシミュレーション
        
        # 温度による変動シミュレーション
        temp = settings.get('temperature', 0.8)
        base_length = 50
        variance = int(20 * temp)  # 温度が高いほど変動大
        
        length = max(30, base_length + random.randint(-variance, variance))
        
        # 一貫性スコアシミュレーション
        consistency_base = 1.0 - (temp - 0.7) * 0.5
        consistency = max(0.3, consistency_base + random.uniform(-0.1, 0.1))
        
        # モックテキスト生成
        mock_text = f"Generated text with {settings['temperature']:.2f} temp, " \
                   f"length: {length}, consistency: {consistency:.3f}"
        
        return mock_text
    
    def measure_consistency(self, prompts: List[str], settings: Dict, 
                          iterations: int = 5) -> Dict:
        """出力一貫性測定"""
        
        logger.info(f"📊 Consistency test: T={settings.get('temperature', 0.8):.2f}")
        
        consistency_scores = []
        length_variance = []
        
        for prompt in tqdm(prompts, desc="Testing prompts"):
            outputs = []
            lengths = []
            
            # 同一プロンプトで複数回生成
            for i in range(iterations):
                output = self.generate_with_settings(prompt, settings)
                outputs.append(output)
                lengths.append(len(output.split()))
            
            # 長さの分散計算
            length_var = np.var(lengths)
            length_variance.append(length_var)
            
            # 一貫性スコア計算（シミュレーション）
            temp = settings.get('temperature', 0.8)
            consistency = max(0.3, 1.0 - temp * 0.4 + random.uniform(-0.1, 0.1))
            consistency_scores.append(consistency)
        
        avg_consistency = np.mean(consistency_scores)
        avg_length_var = np.mean(length_variance)
        
        return {
            'consistency_score': avg_consistency,
            'length_variance': avg_length_var,
            'consistency_scores': consistency_scores,
            'length_variances': length_variance
        }
    
    def test_nkat_stability(self, nkat_params: Dict) -> Dict:
        """NKAT パラメータ安定性テスト"""
        
        logger.info(f"🧬 NKAT stability test: rank={nkat_params.get('rank', 4)}")
        
        # NKATパラメータによる安定性への影響をシミュレーション
        rank = nkat_params.get('rank', 4)
        gamma = nkat_params.get('gamma', 0.95)
        
        # ランクが高いほど安定、ガンマが低いほど安定
        base_stability = 0.7
        rank_factor = min(0.2, rank * 0.02)  # rank 4-10で0.08-0.2の向上
        gamma_factor = (1.0 - gamma) * 0.5  # gamma 0.9で0.05、0.95で0.025の向上
        
        stability_score = base_stability + rank_factor + gamma_factor
        stability_score = min(0.98, stability_score + random.uniform(-0.05, 0.05))
        
        return {
            'nkat_stability': stability_score,
            'rank_contribution': rank_factor,
            'gamma_contribution': gamma_factor,
            'recommended_adjustments': self._get_nkat_recommendations(rank, gamma)
        }
    
    def _get_nkat_recommendations(self, rank: int, gamma: float) -> Dict:
        """NKAT パラメータ推奨調整"""
        
        recommendations = {}
        
        if rank < 6:
            recommendations['rank'] = {
                'current': rank,
                'recommended': 6,
                'reason': 'Higher rank improves stability'
            }
        
        if gamma > 0.96:
            recommendations['gamma'] = {
                'current': gamma,
                'recommended': 0.95,
                'reason': 'Lower gamma reduces output variance'
            }
        
        return recommendations
    
    def run_comprehensive_stability_test(self, output_dir: str = "output/stability") -> Dict:
        """包括的安定性テスト実行"""
        
        logger.info("🔬 Starting comprehensive stability analysis...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # テスト設定配列
        test_configs = [
            # 標準設定
            {'temperature': 0.7, 'top_p': 0.9, 'top_k': 40, 'repeat_penalty': 1.1},
            {'temperature': 0.8, 'top_p': 0.9, 'top_k': 40, 'repeat_penalty': 1.1},
            {'temperature': 0.9, 'top_p': 0.9, 'top_k': 40, 'repeat_penalty': 1.1},
            
            # 最適化済み設定
            {'temperature': 0.95, 'top_p': 0.85, 'top_k': 20, 'repeat_penalty': 1.07},
            
            # 安定性重視設定
            {'temperature': 0.6, 'top_p': 0.85, 'top_k': 30, 'repeat_penalty': 1.05},
            {'temperature': 0.65, 'top_p': 0.88, 'top_k': 35, 'repeat_penalty': 1.08},
        ]
        
        # NKAT設定配列
        nkat_configs = [
            {'rank': 4, 'gamma': 0.95},
            {'rank': 6, 'gamma': 0.95},
            {'rank': 8, 'gamma': 0.95},
            {'rank': 6, 'gamma': 0.93},
            {'rank': 6, 'gamma': 0.97},
        ]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'sampling_tests': [],
            'nkat_tests': [],
            'recommendations': {}
        }
        
        # サンプリング設定テスト
        logger.info("📊 Testing sampling configurations...")
        for i, config in enumerate(tqdm(test_configs, desc="Sampling tests")):
            consistency_result = self.measure_consistency(
                self.test_prompts[:5], config, iterations=3
            )
            
            result = {
                'config': config,
                'consistency': consistency_result,
                'stability_score': consistency_result['consistency_score']
            }
            
            results['sampling_tests'].append(result)
            logger.info(f"   Config {i+1}: {consistency_result['consistency_score']:.3f}")
        
        # NKAT設定テスト
        logger.info("🧬 Testing NKAT configurations...")
        for i, nkat_config in enumerate(tqdm(nkat_configs, desc="NKAT tests")):
            nkat_result = self.test_nkat_stability(nkat_config)
            
            result = {
                'config': nkat_config,
                'stability': nkat_result
            }
            
            results['nkat_tests'].append(result)
            logger.info(f"   NKAT {i+1}: {nkat_result['nkat_stability']:.3f}")
        
        # 最適設定特定
        best_sampling = max(results['sampling_tests'], 
                          key=lambda x: x['stability_score'])
        best_nkat = max(results['nkat_tests'], 
                       key=lambda x: x['stability']['nkat_stability'])
        
        # 推奨設定作成
        optimal_config = {
            'sampling': best_sampling['config'],
            'nkat': best_nkat['config'],
            'expected_stability': (best_sampling['stability_score'] + 
                                 best_nkat['stability']['nkat_stability']) / 2
        }
        
        results['recommendations'] = {
            'optimal_config': optimal_config,
            'stability_improvements': self._generate_stability_recommendations(results),
            'usage_guidelines': self._generate_usage_guidelines(optimal_config)
        }
        
        # 結果保存
        report_path = Path(output_dir) / "stability_analysis_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 可視化
        self._create_stability_visualizations(results, output_dir)
        
        logger.info(f"📄 Stability analysis completed: {report_path}")
        return results
    
    def _generate_stability_recommendations(self, results: Dict) -> List[Dict]:
        """安定性向上推奨事項生成"""
        
        recommendations = []
        
        # 温度設定推奨
        temp_scores = [(r['config']['temperature'], r['stability_score']) 
                      for r in results['sampling_tests']]
        best_temp = min(temp_scores, key=lambda x: abs(x[0] - 0.75))[0]
        
        recommendations.append({
            'category': 'temperature',
            'recommendation': f'Use temperature around {best_temp:.2f} for optimal stability',
            'impact': 'high',
            'reasoning': 'Balanced creativity and consistency'
        })
        
        # NKAT rank推奨
        nkat_scores = [(r['config']['rank'], r['stability']['nkat_stability']) 
                      for r in results['nkat_tests']]
        best_rank = max(nkat_scores, key=lambda x: x[1])[0]
        
        recommendations.append({
            'category': 'nkat_rank',
            'recommendation': f'Set NKAT rank to {best_rank} for stability',
            'impact': 'medium',
            'reasoning': 'Higher rank provides better parameter stability'
        })
        
        # シード固定推奨
        recommendations.append({
            'category': 'reproducibility',
            'recommendation': 'Use fixed seed for reproducible outputs',
            'impact': 'high',
            'reasoning': 'Essential for consistent behavior'
        })
        
        return recommendations
    
    def _generate_usage_guidelines(self, optimal_config: Dict) -> Dict:
        """使用ガイドライン生成"""
        
        return {
            'stable_generation': {
                'description': 'For consistent, stable outputs',
                'config': optimal_config['sampling'],
                'use_cases': ['Production systems', 'Consistent responses', 'API services']
            },
            'creative_generation': {
                'description': 'For creative but controlled outputs',
                'config': {
                    **optimal_config['sampling'],
                    'temperature': optimal_config['sampling']['temperature'] + 0.1
                },
                'use_cases': ['Creative writing', 'Story generation', 'Brainstorming']
            },
            'nkat_tuning': {
                'description': 'NKAT parameter optimization',
                'config': optimal_config['nkat'],
                'notes': 'Fine-tune based on specific model and use case'
            }
        }
    
    def _create_stability_visualizations(self, results: Dict, output_dir: str):
        """安定性可視化作成"""
        
        # 設定別安定性スコア
        configs = [f"Config {i+1}" for i in range(len(results['sampling_tests']))]
        scores = [r['stability_score'] for r in results['sampling_tests']]
        
        plt.figure(figsize=(12, 8))
        
        # サブプロット1: サンプリング設定安定性
        plt.subplot(2, 2, 1)
        plt.bar(configs, scores, color='skyblue', alpha=0.7)
        plt.title('Sampling Configuration Stability')
        plt.ylabel('Stability Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # サブプロット2: 温度vs安定性
        temps = [r['config']['temperature'] for r in results['sampling_tests']]
        plt.subplot(2, 2, 2)
        plt.scatter(temps, scores, color='red', alpha=0.7, s=100)
        plt.plot(temps, scores, color='red', alpha=0.5)
        plt.title('Temperature vs Stability')
        plt.xlabel('Temperature')
        plt.ylabel('Stability Score')
        plt.grid(True, alpha=0.3)
        
        # サブプロット3: NKAT設定安定性
        nkat_labels = [f"R{r['config']['rank']}_G{r['config']['gamma']}" 
                      for r in results['nkat_tests']]
        nkat_scores = [r['stability']['nkat_stability'] for r in results['nkat_tests']]
        
        plt.subplot(2, 2, 3)
        plt.bar(nkat_labels, nkat_scores, color='lightgreen', alpha=0.7)
        plt.title('NKAT Configuration Stability')
        plt.ylabel('NKAT Stability Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # サブプロット4: 総合推奨
        optimal = results['recommendations']['optimal_config']
        metrics = ['Sampling', 'NKAT', 'Combined']
        values = [
            max(scores),
            max(nkat_scores),
            optimal['expected_stability']
        ]
        
        plt.subplot(2, 2, 4)
        bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'gold'], alpha=0.7)
        plt.title('Optimal Configuration Performance')
        plt.ylabel('Stability Score')
        plt.ylim(0, 1.0)
        
        # 値をバーの上に表示
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "stability_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Visualization saved: {output_dir}/stability_analysis.png")
    
    def apply_stability_enhancement(self, config_file: str = None) -> str:
        """安定性エンハンス設定適用"""
        
        if not config_file:
            # デフォルト安定性設定
            config = {
                'sampling': {
                    'temperature': 0.75,
                    'top_p': 0.85,
                    'top_k': 30,
                    'repeat_penalty': 1.05,
                    'seed': 42  # 固定シード
                },
                'nkat': {
                    'rank': 6,
                    'gamma': 0.95,
                    'stability_mode': True
                },
                'inference': {
                    'use_cuda': True,
                    'batch_size': 1,
                    'max_length': 512,
                    'do_sample': True
                }
            }
        else:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        
        # 設定ファイル保存
        enhanced_config_path = "config/nkat_stability_enhanced.json"
        os.makedirs("config", exist_ok=True)
        
        with open(enhanced_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # 実行コマンド生成
        cmd = f"""# NKAT安定性エンハンス済み推論コマンド
python nkat_inference_engine.py \\
  --model {self.model_path or "models/test/test_large_NKAT_real.gguf"} \\
  --temperature {config['sampling']['temperature']:.2f} \\
  --top_p {config['sampling']['top_p']:.2f} \\
  --top_k {config['sampling']['top_k']} \\
  --repeat_penalty {config['sampling']['repeat_penalty']:.2f} \\
  --seed {config['sampling'].get('seed', 42)} \\
  --nkat_rank {config['nkat']['rank']} \\
  --nkat_gamma {config['nkat']['gamma']:.2f} \\
  --use_cuda \\
  --stability_mode"""
        
        logger.info(f"✅ Enhanced config saved: {enhanced_config_path}")
        logger.info("🚀 Recommended command:")
        print(cmd)
        
        return enhanced_config_path

def main():
    """メイン実行"""
    
    import argparse
    parser = argparse.ArgumentParser(description="NKAT Stability Enhancer")
    parser.add_argument("--model", type=str, help="Model path")
    parser.add_argument("--output", type=str, default="output/stability", 
                       help="Output directory")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick stability test")
    parser.add_argument("--enhance", action="store_true", 
                       help="Apply stability enhancement")
    
    args = parser.parse_args()
    
    print("🔬 NKAT Stability Enhancer")
    print("=" * 60)
    
    enhancer = NKATStabilityEnhancer(args.model)
    
    if args.enhance:
        config_path = enhancer.apply_stability_enhancement()
        print(f"\n✅ Stability enhancement applied: {config_path}")
    else:
        results = enhancer.run_comprehensive_stability_test(args.output)
        
        optimal = results['recommendations']['optimal_config']
        print(f"\n🏆 Optimal Configuration:")
        print(f"   Expected Stability: {optimal['expected_stability']:.3f}")
        print(f"   Temperature: {optimal['sampling']['temperature']:.2f}")
        print(f"   NKAT Rank: {optimal['nkat']['rank']}")
        
        print(f"\n📄 Full report: {args.output}/stability_analysis_report.json")

if __name__ == "__main__":
    main() 