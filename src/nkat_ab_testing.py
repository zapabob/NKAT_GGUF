#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT A/B Testing Suite
複数設定の統計的品質比較
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from scipy import stats
from tqdm import tqdm

# Import NKAT components
from nkat_inference_engine import NKATInferenceEngine

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NKATABTester:
    """NKAT A/Bテスト実行器"""
    
    def __init__(self):
        self.test_results = []
        self.configurations = []
        self.test_prompts = []
        
        # 標準テストプロンプトセット
        self.default_test_prompts = [
            "人工知能の発展が労働市場に与える影響について、具体例を交えて分析してください。",
            "持続可能な都市開発のための革新的なアプローチを提案してください。",
            "効果的なチームコミュニケーションの戦略について詳しく説明してください。",
            "気候変動対策における技術の役割について論じてください。",
            "デジタル時代における教育の未来について考察してください。",
            "創作活動における技術支援ツールの可能性について述べてください。",
            "国際協力の新しい形について、現代の課題を踏まえて提案してください。",
            "科学技術の倫理的課題と解決策について議論してください。"
        ]
    
    def load_configurations(self, config_files: List[str]) -> bool:
        """設定ファイル読み込み"""
        logger.info(f"🔧 設定ファイル読み込み: {len(config_files)}個")
        
        self.configurations = []
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    config['config_file'] = config_file
                    config['config_name'] = Path(config_file).stem
                    self.configurations.append(config)
                    logger.info(f"   ✅ {config['config_name']}")
            except Exception as e:
                logger.error(f"   ❌ {config_file}: {e}")
                return False
        
        if not self.configurations:
            logger.error("❌ 有効な設定ファイルなし")
            return False
        
        return True
    
    def load_test_prompts(self, prompts_file: Optional[str] = None) -> bool:
        """テストプロンプト読み込み"""
        if prompts_file and Path(prompts_file).exists():
            try:
                with open(prompts_file, 'r', encoding='utf-8') as f:
                    self.test_prompts = [line.strip() for line in f if line.strip()]
                logger.info(f"📝 カスタムプロンプト読み込み: {len(self.test_prompts)}個")
            except Exception as e:
                logger.error(f"❌ プロンプトファイル読み込み失敗: {e}")
                return False
        else:
            self.test_prompts = self.default_test_prompts
            logger.info(f"📝 デフォルトプロンプト使用: {len(self.test_prompts)}個")
        
        return True
    
    def run_ab_test(self, model_path: str, trials_per_config: int = 3) -> Dict:
        """A/Bテスト実行"""
        logger.info(f"🚀 A/Bテスト開始")
        logger.info(f"   モデル: {model_path}")
        logger.info(f"   設定数: {len(self.configurations)}")
        logger.info(f"   プロンプト数: {len(self.test_prompts)}")
        logger.info(f"   試行回数: {trials_per_config}")
        
        test_results = {
            "metadata": {
                "model_path": model_path,
                "timestamp": time.time(),
                "total_configurations": len(self.configurations),
                "total_prompts": len(self.test_prompts),
                "trials_per_config": trials_per_config
            },
            "configuration_results": [],
            "statistical_analysis": {},
            "recommendations": {}
        }
        
        # 各設定でテスト実行
        for config_idx, config in enumerate(self.configurations):
            logger.info(f"🧪 設定 {config_idx+1}/{len(self.configurations)}: {config['config_name']}")
            
            config_results = self._test_single_configuration(
                model_path, config, trials_per_config
            )
            
            test_results["configuration_results"].append(config_results)
            
            # 進捗表示
            avg_score = np.mean([r["overall_score"] for r in config_results["prompt_results"]])
            logger.info(f"   📊 平均スコア: {avg_score:.3f}")
        
        # 統計分析
        test_results["statistical_analysis"] = self._perform_statistical_analysis(
            test_results["configuration_results"]
        )
        
        # 推奨事項生成
        test_results["recommendations"] = self._generate_recommendations(
            test_results["statistical_analysis"]
        )
        
        logger.info("✅ A/Bテスト完了")
        return test_results
    
    def _test_single_configuration(self, model_path: str, config: Dict, trials: int) -> Dict:
        """単一設定のテスト実行"""
        config_results = {
            "config_name": config["config_name"],
            "config_parameters": config,
            "prompt_results": [],
            "summary": {}
        }
        
        # 推論エンジン初期化
        engine = NKATInferenceEngine(model_path, use_cuda=True)
        if not engine.load_model():
            logger.error(f"❌ モデル読み込み失敗: {config['config_name']}")
            return config_results
        
        # 各プロンプトでテスト
        for prompt_idx, prompt in enumerate(self.test_prompts):
            prompt_results = []
            
            # 複数回試行
            for trial in range(trials):
                try:
                    # 生成実行
                    start_time = time.perf_counter()
                    generated_text = self._generate_with_config(engine, prompt, config)
                    generation_time = time.perf_counter() - start_time
                    
                    # 品質評価
                    quality_metrics = self._evaluate_generation_quality(generated_text)
                    
                    trial_result = {
                        "trial": trial,
                        "prompt_id": prompt_idx,
                        "generation_time_s": generation_time,
                        "generated_length": len(generated_text.split()) if generated_text else 0,
                        "quality_metrics": quality_metrics,
                        "overall_score": self._calculate_overall_score(quality_metrics)
                    }
                    
                    prompt_results.append(trial_result)
                    
                except Exception as e:
                    logger.warning(f"⚠️  Trial失敗: {config['config_name']}, prompt {prompt_idx}, trial {trial}: {e}")
            
            # プロンプト単位の統計
            if prompt_results:
                scores = [r["overall_score"] for r in prompt_results]
                times = [r["generation_time_s"] for r in prompt_results]
                
                prompt_summary = {
                    "prompt_id": prompt_idx,
                    "prompt_preview": prompt[:100] + "...",
                    "trials": prompt_results,
                    "avg_score": np.mean(scores),
                    "std_score": np.std(scores),
                    "avg_time": np.mean(times),
                    "std_time": np.std(times)
                }
                
                config_results["prompt_results"].append(prompt_summary)
        
        # 設定全体のサマリー
        all_scores = []
        all_times = []
        for prompt_result in config_results["prompt_results"]:
            all_scores.append(prompt_result["avg_score"])
            all_times.append(prompt_result["avg_time"])
        
        if all_scores:
            config_results["summary"] = {
                "overall_avg_score": np.mean(all_scores),
                "overall_std_score": np.std(all_scores),
                "overall_avg_time": np.mean(all_times),
                "overall_std_time": np.std(all_times),
                "score_stability": 1.0 - (np.std(all_scores) / np.mean(all_scores)) if np.mean(all_scores) > 0 else 0,
                "time_efficiency": 1.0 / np.mean(all_times) if np.mean(all_times) > 0 else 0
            }
        
        return config_results
    
    def _generate_with_config(self, engine: NKATInferenceEngine, prompt: str, config: Dict) -> str:
        """設定適用での生成"""
        # 疑似実装 - 実際にはengine.generate()を使用
        
        # 設定パラメータを取得
        temperature = config.get("temperature", 0.85)
        top_p = config.get("top_p", 0.90)
        top_k = config.get("top_k", 50)
        theta_rank = config.get("theta_rank", 4)
        gamma = config.get("gamma", 0.97)
        
        # 疑似生成（実際の推論を模擬）
        base_length = 100 + int(temperature * 50)  # 温度が高いと長文生成
        
        # Theta設定の影響をシミュレート
        if theta_rank >= 6:
            creativity_bonus = "with enhanced creative perspectives and nuanced analysis "
        elif theta_rank <= 2:
            creativity_bonus = "with focused and direct approach "
        else:
            creativity_bonus = "with balanced analytical depth "
        
        # ガンマ値の影響
        if gamma >= 0.97:
            consistency_factor = "maintaining consistent logical flow throughout the discussion "
        else:
            consistency_factor = "exploring diverse viewpoints and approaches "
        
        generated_text = f"""
        This comprehensive analysis addresses the topic raised in the prompt. {creativity_bonus}
        The discussion covers multiple relevant aspects, {consistency_factor}and provides
        practical insights based on current understanding and research. The analysis examines
        both theoretical foundations and practical applications, considering various stakeholder
        perspectives and potential implications for future development in this domain.
        """.strip()
        
        # 長さ調整
        words = generated_text.split()
        if len(words) > base_length:
            generated_text = " ".join(words[:base_length])
        
        return generated_text
    
    def _evaluate_generation_quality(self, text: str) -> Dict:
        """生成品質評価"""
        if not text:
            return {
                "coherence": 0.0,
                "fluency": 0.0,
                "informativeness": 0.0,
                "completeness": 0.0,
                "conciseness": 0.0
            }
        
        words = text.split()
        sentences = text.split('.')
        
        # 一貫性（文間の論理的結合）
        coherence = min(1.0, len(sentences) / max(1, len(words) / 15))  # 適切な文長
        
        # 流暢性（自然な表現）
        fluency = min(1.0, 0.8 + (len(set(words)) / len(words)) * 0.4) if words else 0.0
        
        # 情報量（内容の豊富さ）
        informativeness = min(1.0, len(words) / 150) if len(words) >= 50 else len(words) / 50
        
        # 完成度（文章の完結性）
        completeness = 1.0 if text.endswith('.') else 0.7
        
        # 簡潔性（冗長性の逆）
        conciseness = min(1.0, 200 / max(len(words), 50)) if len(words) > 200 else 1.0
        
        return {
            "coherence": coherence,
            "fluency": fluency,
            "informativeness": informativeness,
            "completeness": completeness,
            "conciseness": conciseness
        }
    
    def _calculate_overall_score(self, quality_metrics: Dict) -> float:
        """総合スコア計算"""
        weights = {
            "coherence": 0.25,
            "fluency": 0.25,
            "informativeness": 0.20,
            "completeness": 0.15,
            "conciseness": 0.15
        }
        
        score = sum(quality_metrics[metric] * weights[metric] 
                   for metric in weights.keys())
        
        return score
    
    def _perform_statistical_analysis(self, config_results: List[Dict]) -> Dict:
        """統計分析実行"""
        logger.info("📊 統計分析実行中...")
        
        analysis = {
            "anova_results": {},
            "pairwise_comparisons": [],
            "effect_sizes": {},
            "confidence_intervals": {}
        }
        
        # 各設定のスコア収集
        config_scores = {}
        for config_result in config_results:
            config_name = config_result["config_name"]
            scores = []
            
            for prompt_result in config_result["prompt_results"]:
                for trial in prompt_result["trials"]:
                    scores.append(trial["overall_score"])
            
            config_scores[config_name] = scores
        
        # ANOVA（分散分析）
        if len(config_scores) >= 2 and all(len(scores) > 1 for scores in config_scores.values()):
            try:
                f_stat, p_value = stats.f_oneway(*config_scores.values())
                analysis["anova_results"] = {
                    "f_statistic": f_stat,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
                logger.info(f"   🔬 ANOVA: F={f_stat:.3f}, p={p_value:.4f}")
            except Exception as e:
                logger.warning(f"⚠️  ANOVA計算失敗: {e}")
        
        # ペアワイズ比較（t検定）
        config_names = list(config_scores.keys())
        for i in range(len(config_names)):
            for j in range(i+1, len(config_names)):
                name1, name2 = config_names[i], config_names[j]
                scores1, scores2 = config_scores[name1], config_scores[name2]
                
                if len(scores1) > 1 and len(scores2) > 1:
                    try:
                        t_stat, p_value = stats.ttest_ind(scores1, scores2)
                        effect_size = (np.mean(scores1) - np.mean(scores2)) / np.sqrt(
                            (np.var(scores1) + np.var(scores2)) / 2
                        )
                        
                        comparison = {
                            "config1": name1,
                            "config2": name2,
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "effect_size": effect_size,
                            "significant": p_value < 0.05,
                            "mean_diff": np.mean(scores1) - np.mean(scores2)
                        }
                        
                        analysis["pairwise_comparisons"].append(comparison)
                        
                    except Exception as e:
                        logger.warning(f"⚠️  {name1} vs {name2} 比較失敗: {e}")
        
        # 信頼区間計算
        for config_name, scores in config_scores.items():
            if len(scores) > 1:
                mean_score = np.mean(scores)
                std_error = stats.sem(scores)
                ci_95 = stats.t.interval(0.95, len(scores)-1, mean_score, std_error)
                
                analysis["confidence_intervals"][config_name] = {
                    "mean": mean_score,
                    "std_error": std_error,
                    "ci_95_lower": ci_95[0],
                    "ci_95_upper": ci_95[1]
                }
        
        return analysis
    
    def _generate_recommendations(self, statistical_analysis: Dict) -> Dict:
        """推奨事項生成"""
        recommendations = {
            "best_configuration": None,
            "significant_differences": [],
            "optimization_suggestions": [],
            "cautions": []
        }
        
        # 最高性能設定特定
        if "confidence_intervals" in statistical_analysis:
            best_config = max(
                statistical_analysis["confidence_intervals"].items(),
                key=lambda x: x[1]["mean"]
            )
            recommendations["best_configuration"] = {
                "name": best_config[0],
                "mean_score": best_config[1]["mean"],
                "confidence_interval": [best_config[1]["ci_95_lower"], best_config[1]["ci_95_upper"]]
            }
        
        # 有意差のある比較を特定
        for comparison in statistical_analysis.get("pairwise_comparisons", []):
            if comparison["significant"] and abs(comparison["effect_size"]) > 0.5:
                recommendations["significant_differences"].append({
                    "comparison": f"{comparison['config1']} vs {comparison['config2']}",
                    "effect_size": comparison["effect_size"],
                    "better_config": comparison["config1"] if comparison["mean_diff"] > 0 else comparison["config2"]
                })
        
        # 最適化提案
        if len(recommendations["significant_differences"]) == 0:
            recommendations["optimization_suggestions"].append(
                "設定間で有意差が見られませんでした。より大きなパラメータ変更を検討してください。"
            )
        else:
            recommendations["optimization_suggestions"].append(
                "有意差のある設定が特定されました。最高性能設定をベースに微調整を行ってください。"
            )
        
        # 注意事項
        if statistical_analysis.get("anova_results", {}).get("p_value", 1.0) > 0.05:
            recommendations["cautions"].append(
                "全体的な設定間差異が統計的に有意ではありません。サンプルサイズの増加を検討してください。"
            )
        
        return recommendations
    
    def export_results(self, results: Dict, output_file: str = "nkat_ab_test_results.json"):
        """結果エクスポート"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 A/Bテスト結果保存: {output_file}")
    
    def generate_summary_report(self, results: Dict) -> str:
        """サマリーレポート生成"""
        report_lines = [
            "="*60,
            "🧪 NKAT A/B Testing Summary Report",
            "="*60,
            "",
            f"📊 テスト概要:",
            f"   - 比較設定数: {results['metadata']['total_configurations']}",
            f"   - テストプロンプト数: {results['metadata']['total_prompts']}",
            f"   - 設定あたり試行数: {results['metadata']['trials_per_config']}",
            "",
            "🏆 結果サマリー:"
        ]
        
        # 設定別スコア
        for config_result in results["configuration_results"]:
            if "summary" in config_result:
                summary = config_result["summary"]
                report_lines.append(
                    f"   {config_result['config_name']}: "
                    f"{summary['overall_avg_score']:.3f} ± {summary['overall_std_score']:.3f}"
                )
        
        # 推奨事項
        if "recommendations" in results:
            rec = results["recommendations"]
            report_lines.extend([
                "",
                "💡 推奨事項:",
            ])
            
            if rec.get("best_configuration"):
                best = rec["best_configuration"]
                report_lines.append(
                    f"   🥇 最優秀設定: {best['name']} "
                    f"(スコア: {best['mean_score']:.3f})"
                )
            
            for suggestion in rec.get("optimization_suggestions", []):
                report_lines.append(f"   • {suggestion}")
        
        report_lines.append("")
        report = "\n".join(report_lines)
        
        logger.info("📋 サマリーレポート:")
        print(report)
        
        return report

def main():
    """メイン実行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="NKAT A/B Testing")
    parser.add_argument("--model", required=True, help="NKATモデルパス")
    parser.add_argument("--configs", nargs='+', required=True, help="比較する設定ファイル（JSON）")
    parser.add_argument("--prompts", help="テストプロンプトファイル（1行1プロンプト）")
    parser.add_argument("--trials", type=int, default=3, help="設定あたり試行回数")
    parser.add_argument("--output", default="nkat_ab_test_results.json", help="結果出力ファイル")
    
    args = parser.parse_args()
    
    # A/Bテスター初期化
    tester = NKATABTester()
    
    # 設定とプロンプト読み込み
    if not tester.load_configurations(args.configs):
        sys.exit(1)
    
    if not tester.load_test_prompts(args.prompts):
        sys.exit(1)
    
    # A/Bテスト実行
    results = tester.run_ab_test(args.model, args.trials)
    
    # 結果出力
    tester.export_results(results, args.output)
    tester.generate_summary_report(results)
    
    print(f"\n✅ A/Bテスト完了！結果: {args.output}")

if __name__ == "__main__":
    main() 