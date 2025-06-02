#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT統合評価システム
前回の結果をもとにした総合的な評価とレポート生成
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from datetime import datetime
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nkat_integration_evaluation.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NKATIntegrationEvaluator:
    """NKAT統合評価器"""
    
    def __init__(self):
        self.output_dir = Path("output/qwen3_nkat_testing")
        self.evaluation_timestamp = datetime.now()
        
        logger.info("🔍 NKAT Integration Evaluator initialized")
    
    def load_previous_results(self) -> Dict:
        """前回の結果を読み込み"""
        results = {
            "inference_test": None,
            "quality_benchmark": None,
            "optimization": None
        }
        
        try:
            # 推論テスト結果
            inference_path = self.output_dir / "qwen3_nkat_test_results.json"
            if inference_path.exists():
                with open(inference_path, 'r', encoding='utf-8') as f:
                    results["inference_test"] = json.load(f)
                logger.info("✅ Inference test results loaded")
            
            # 品質ベンチマーク結果
            quality_path = self.output_dir / "nkat_quality_benchmark_results.json"
            if quality_path.exists():
                with open(quality_path, 'r', encoding='utf-8') as f:
                    results["quality_benchmark"] = json.load(f)
                logger.info("✅ Quality benchmark results loaded")
            
            # 最適化結果（あれば）
            optimization_path = self.output_dir / "nkat_optimization_results.json"
            if optimization_path.exists():
                with open(optimization_path, 'r', encoding='utf-8') as f:
                    results["optimization"] = json.load(f)
                logger.info("✅ Optimization results loaded")
            
        except Exception as e:
            logger.warning(f"⚠️ Error loading previous results: {e}")
        
        return results
    
    def analyze_performance_accuracy(self, results: Dict) -> Dict:
        """性能データの確度分析"""
        analysis = {
            "accuracy_assessment": "unknown",
            "potential_issues": [],
            "confidence_level": 0.0,
            "recommendations": []
        }
        
        inference_results = results.get("inference_test")
        if not inference_results:
            analysis["accuracy_assessment"] = "no_data"
            return analysis
        
        # NKATパフォーマンス指標の検証
        try:
            nkat_metrics = inference_results.get("nkat_performance", {})
            
            # 445K tokens/secの妥当性チェック
            throughput = nkat_metrics.get("throughput_tokens_per_sec", 0)
            
            if throughput > 300000:  # 300K以上は疑わしい
                analysis["potential_issues"].append({
                    "issue": "unusually_high_throughput",
                    "description": f"Throughput of {throughput:,.0f} tokens/sec seems too high for 8B Q6_K model",
                    "severity": "high"
                })
                analysis["confidence_level"] = 0.2
            elif throughput > 50000:
                analysis["potential_issues"].append({
                    "issue": "high_throughput",
                    "description": "Throughput is higher than typical for this model size",
                    "severity": "medium"
                })
                analysis["confidence_level"] = 0.6
            else:
                analysis["confidence_level"] = 0.8
            
            # VRAMの使用量チェック
            vram_usage = nkat_metrics.get("vram_usage_gb", 0)
            if vram_usage < 6.0:  # 8BモデルQ6_Kで6GB以下は低すぎる
                analysis["potential_issues"].append({
                    "issue": "low_vram_usage",
                    "description": f"VRAM usage of {vram_usage:.1f}GB seems low for 8B Q6_K model",
                    "severity": "medium"
                })
            
            # 推奨事項
            if throughput > 200000:
                analysis["recommendations"].extend([
                    "実際のllama.cpp実行ファイルでの検証推奨",
                    "長文ストリーム出力での実測必要",
                    "GPU使用率の詳細確認必要"
                ])
            
            # 総合評価
            if analysis["confidence_level"] >= 0.7:
                analysis["accuracy_assessment"] = "likely_accurate"
            elif analysis["confidence_level"] >= 0.4:
                analysis["accuracy_assessment"] = "needs_verification"
            else:
                analysis["accuracy_assessment"] = "likely_synthetic"
                
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            analysis["accuracy_assessment"] = "analysis_failed"
        
        return analysis
    
    def analyze_quality_metrics(self, results: Dict) -> Dict:
        """品質メトリクスの分析"""
        analysis = {
            "overall_score": 0.0,
            "category_performance": {},
            "strengths": [],
            "weaknesses": [],
            "improvement_areas": []
        }
        
        quality_results = results.get("quality_benchmark")
        if not quality_results:
            return analysis
        
        try:
            scenario_results = quality_results.get("scenario_results", [])
            
            category_scores = []
            for scenario in scenario_results:
                agg_metrics = scenario.get("aggregate_metrics", {})
                if agg_metrics:
                    category_name = scenario["scenario"]["category"]
                    avg_score = agg_metrics.get("avg_overall_score", 0.0)
                    
                    analysis["category_performance"][category_name] = {
                        "score": avg_score,
                        "scenario_name": scenario["scenario"]["name"]
                    }
                    category_scores.append(avg_score)
            
            if category_scores:
                analysis["overall_score"] = np.mean(category_scores)
                
                # ベストカテゴリ
                best_categories = sorted(
                    analysis["category_performance"].items(),
                    key=lambda x: x[1]["score"],
                    reverse=True
                )[:2]
                
                analysis["strengths"] = [
                    f"{cat[1]['scenario_name']} ({cat[1]['score']:.3f})"
                    for cat in best_categories
                ]
                
                # 改善が必要なカテゴリ
                weak_categories = [
                    cat for cat in analysis["category_performance"].items()
                    if cat[1]["score"] < 0.65
                ]
                
                analysis["weaknesses"] = [
                    f"{cat[1]['scenario_name']} ({cat[1]['score']:.3f})"
                    for cat in weak_categories
                ]
                
                # 改善提案
                if analysis["overall_score"] < 0.7:
                    analysis["improvement_areas"].extend([
                        "NKATパラメータ調整 (rank/gamma)",
                        "プロンプトエンジニアリング最適化",
                        "モデル微調整検討"
                    ])
                elif analysis["overall_score"] < 0.8:
                    analysis["improvement_areas"].extend([
                        "特定カテゴリの専門化",
                        "コンテキスト長最適化"
                    ])
                
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
        
        return analysis
    
    def generate_recommendations(self, performance_analysis: Dict, quality_analysis: Dict) -> List[Dict]:
        """推奨事項の生成"""
        recommendations = []
        
        # 1. 確度チェック推奨
        if performance_analysis["accuracy_assessment"] in ["needs_verification", "likely_synthetic"]:
            recommendations.append({
                "priority": "high",
                "category": "verification",
                "title": "実測推論性能の確認",
                "description": "実際のllama.cpp実行ファイルを使用した詳細な性能測定が必要",
                "actions": [
                    "llama.cpp/build_*/main.exeでの直接テスト",
                    "nvidia-smi dmonでのGPU使用率確認",
                    "長文ストリーム出力での実測"
                ]
            })
        
        # 2. 品質向上推奨
        overall_quality = quality_analysis.get("overall_score", 0.0)
        if overall_quality < 0.75:
            recommendations.append({
                "priority": "medium",
                "category": "quality_improvement",
                "title": "品質スコア向上",
                "description": f"現在の品質スコア{overall_quality:.3f}をさらに向上",
                "actions": [
                    "NKATパラメータの微調整",
                    "弱いカテゴリの重点的改善",
                    "より多様なテストシナリオでの評価"
                ]
            })
        
        # 3. 最適化推奨
        recommendations.append({
            "priority": "medium",
            "category": "optimization",
            "title": "パラメータ最適化",
            "description": "rank/gammaの体系的最適化",
            "actions": [
                "Optuna多目的最適化の実行",
                "パレートフロント分析",
                "実用シナリオでの検証"
            ]
        })
        
        # 4. CI/CD統合推奨
        recommendations.append({
            "priority": "low",
            "category": "automation",
            "title": "自動化システム構築",
            "description": "継続的品質監視の実装",
            "actions": [
                "GitHub Actions統合",
                "定期的ベンチマーク実行",
                "アラートシステム構築"
            ]
        })
        
        return recommendations
    
    def create_comprehensive_visualization(self, performance_analysis: Dict, quality_analysis: Dict) -> None:
        """総合的な可視化"""
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('NKAT Integration Evaluation Summary', fontsize=16, fontweight='bold')
            
            # 1. 品質スコア分布
            category_performance = quality_analysis.get("category_performance", {})
            if category_performance:
                categories = list(category_performance.keys())
                scores = [category_performance[cat]["score"] for cat in categories]
                
                bars = ax1.bar(range(len(categories)), scores, color='skyblue', alpha=0.7)
                ax1.set_xlabel('Test Categories')
                ax1.set_ylabel('Quality Score')
                ax1.set_title('Quality Scores by Category')
                ax1.set_xticks(range(len(categories)))
                ax1.set_xticklabels(categories, rotation=45, ha='right')
                ax1.set_ylim(0, 1.0)
                ax1.grid(True, alpha=0.3)
                
                # スコア値をバーの上に表示
                for i, (bar, score) in enumerate(zip(bars, scores)):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')
            
            # 2. 信頼性評価
            confidence_level = performance_analysis.get("confidence_level", 0.0)
            accuracy_assessment = performance_analysis.get("accuracy_assessment", "unknown")
            
            # 信頼性ゲージ
            angles = np.linspace(0, np.pi, 100)
            confidence_angle = np.pi * confidence_level
            
            ax2.plot(angles, np.ones_like(angles), 'lightgray', linewidth=10)
            ax2.plot(angles[angles <= confidence_angle], 
                    np.ones_like(angles[angles <= confidence_angle]), 
                    'green' if confidence_level > 0.7 else 'orange' if confidence_level > 0.4 else 'red',
                    linewidth=10)
            
            ax2.set_xlim(0, np.pi)
            ax2.set_ylim(0.5, 1.5)
            ax2.set_title(f'Performance Data Confidence\n{confidence_level:.1%} - {accuracy_assessment}')
            ax2.set_xticks([0, np.pi/2, np.pi])
            ax2.set_xticklabels(['Low', 'Medium', 'High'])
            ax2.set_yticks([])
            
            # 3. 問題の重要度分布
            issues = performance_analysis.get("potential_issues", [])
            if issues:
                severity_counts = {}
                for issue in issues:
                    severity = issue.get("severity", "unknown")
                    severity_counts[severity] = severity_counts.get(severity, 0) + 1
                
                severities = list(severity_counts.keys())
                counts = list(severity_counts.values())
                colors = {'high': 'red', 'medium': 'orange', 'low': 'yellow', 'unknown': 'gray'}
                bar_colors = [colors.get(sev, 'gray') for sev in severities]
                
                ax3.bar(severities, counts, color=bar_colors, alpha=0.7)
                ax3.set_xlabel('Issue Severity')
                ax3.set_ylabel('Count')
                ax3.set_title('Potential Issues by Severity')
                ax3.grid(True, alpha=0.3)
            else:
                ax3.text(0.5, 0.5, 'No issues detected', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=14)
                ax3.set_title('Issue Analysis')
            
            # 4. 総合評価サマリー
            overall_score = quality_analysis.get("overall_score", 0.0)
            
            # 円グラフで総合評価
            labels = ['Quality Score', 'Remaining']
            sizes = [overall_score, 1.0 - overall_score]
            colors = ['green' if overall_score > 0.8 else 'orange' if overall_score > 0.6 else 'red', 'lightgray']
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                              startangle=90)
            ax4.set_title(f'Overall Quality Score\n{overall_score:.3f}')
            
            plt.tight_layout()
            
            # 保存
            chart_path = self.output_dir / "nkat_integration_evaluation.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"📊 Evaluation charts saved: {chart_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"❌ Visualization failed: {e}")
    
    def generate_comprehensive_report(self, results: Dict, performance_analysis: Dict, 
                                    quality_analysis: Dict, recommendations: List[Dict]) -> Dict:
        """総合レポート生成"""
        
        report = {
            "evaluation_metadata": {
                "timestamp": self.evaluation_timestamp.isoformat(),
                "evaluator_version": "1.0",
                "model_tested": "Qwen3-8B-ERP-v0.1.i1-Q6_K.gguf"
            },
            "executive_summary": self.create_executive_summary(performance_analysis, quality_analysis),
            "detailed_analysis": {
                "performance_accuracy": performance_analysis,
                "quality_assessment": quality_analysis
            },
            "recommendations": recommendations,
            "next_steps": self.create_next_steps(recommendations),
            "appendix": {
                "raw_results": results
            }
        }
        
        return report
    
    def create_executive_summary(self, performance_analysis: Dict, quality_analysis: Dict) -> Dict:
        """エグゼクティブサマリー作成"""
        
        overall_quality = quality_analysis.get("overall_score", 0.0)
        confidence = performance_analysis.get("confidence_level", 0.0)
        
        # 総合評価
        if overall_quality >= 0.8 and confidence >= 0.7:
            overall_status = "excellent"
        elif overall_quality >= 0.7 and confidence >= 0.5:
            overall_status = "good"
        elif overall_quality >= 0.6 or confidence >= 0.4:
            overall_status = "fair"
        else:
            overall_status = "needs_improvement"
        
        return {
            "overall_status": overall_status,
            "key_findings": [
                f"品質スコア: {overall_quality:.3f}/1.0",
                f"データ信頼性: {confidence:.1%}",
                f"主要課題数: {len(performance_analysis.get('potential_issues', []))}"
            ],
            "critical_issues": [
                issue["description"] for issue in performance_analysis.get("potential_issues", [])
                if issue.get("severity") == "high"
            ],
            "strengths": quality_analysis.get("strengths", []),
            "immediate_actions_required": len([
                rec for rec in [] if rec.get("priority") == "high"
            ]) > 0
        }
    
    def create_next_steps(self, recommendations: List[Dict]) -> List[str]:
        """次のステップ作成"""
        
        next_steps = []
        
        # 優先度順で整理
        high_priority = [rec for rec in recommendations if rec.get("priority") == "high"]
        medium_priority = [rec for rec in recommendations if rec.get("priority") == "medium"]
        
        if high_priority:
            next_steps.append("🔴 即座に実行すべき項目:")
            for rec in high_priority:
                next_steps.extend([f"  • {action}" for action in rec.get("actions", [])])
        
        if medium_priority:
            next_steps.append("🟡 中期的に実行すべき項目:")
            for rec in medium_priority[:2]:  # 上位2つ
                next_steps.extend([f"  • {action}" for action in rec.get("actions", [])])
        
        next_steps.append("📋 継続的モニタリング:")
        next_steps.extend([
            "  • 週次品質ベンチマーク実行",
            "  • 月次性能評価",
            "  • 四半期最適化レビュー"
        ])
        
        return next_steps
    
    def save_comprehensive_report(self, report: Dict) -> None:
        """総合レポート保存"""
        
        try:
            # JSONレポート保存
            report_path = self.output_dir / "nkat_comprehensive_evaluation_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Comprehensive report saved: {report_path}")
            
            # Markdownレポート作成
            self.create_markdown_report(report)
            
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
    
    def create_markdown_report(self, report: Dict) -> None:
        """Markdownレポート作成"""
        
        try:
            md_content = f"""# NKAT Integration Evaluation Report

**Generated:** {report['evaluation_metadata']['timestamp']}  
**Model:** {report['evaluation_metadata']['model_tested']}

## 🎯 Executive Summary

**Overall Status:** {report['executive_summary']['overall_status'].upper()}

### Key Findings
{chr(10).join(f"- {finding}" for finding in report['executive_summary']['key_findings'])}

### Strengths
{chr(10).join(f"- {strength}" for strength in report['executive_summary']['strengths'])}

### Critical Issues
{chr(10).join(f"- ⚠️ {issue}" for issue in report['executive_summary']['critical_issues'])}

## 📊 Detailed Analysis

### Performance Data Accuracy
- **Confidence Level:** {report['detailed_analysis']['performance_accuracy']['confidence_level']:.1%}
- **Assessment:** {report['detailed_analysis']['performance_accuracy']['accuracy_assessment']}

### Quality Assessment
- **Overall Score:** {report['detailed_analysis']['quality_assessment']['overall_score']:.3f}/1.0

#### Category Performance
"""

            # カテゴリ別性能
            category_perf = report['detailed_analysis']['quality_assessment'].get('category_performance', {})
            for category, data in category_perf.items():
                md_content += f"- **{data['scenario_name']}:** {data['score']:.3f}\n"

            md_content += f"""

## 🚀 Recommendations

"""
            
            # 推奨事項
            for i, rec in enumerate(report['recommendations'], 1):
                md_content += f"""### {i}. {rec['title']} (Priority: {rec['priority'].upper()})

{rec['description']}

**Actions:**
{chr(10).join(f"- {action}" for action in rec['actions'])}

"""

            md_content += f"""## 📋 Next Steps

{chr(10).join(report['next_steps'])}

---
*Report generated by NKAT Integration Evaluator v{report['evaluation_metadata']['evaluator_version']}*
"""
            
            md_path = self.output_dir / "nkat_evaluation_report.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            
            logger.info(f"📄 Markdown report saved: {md_path}")
            
        except Exception as e:
            logger.error(f"❌ Markdown report creation failed: {e}")
    
    def run_comprehensive_evaluation(self) -> None:
        """総合評価実行"""
        
        logger.info("🔍 Starting comprehensive NKAT integration evaluation")
        
        # 1. 前回結果の読み込み
        results = self.load_previous_results()
        
        # 2. 性能確度分析
        performance_analysis = self.analyze_performance_accuracy(results)
        
        # 3. 品質分析
        quality_analysis = self.analyze_quality_metrics(results)
        
        # 4. 推奨事項生成
        recommendations = self.generate_recommendations(performance_analysis, quality_analysis)
        
        # 5. 可視化
        self.create_comprehensive_visualization(performance_analysis, quality_analysis)
        
        # 6. 総合レポート生成・保存
        comprehensive_report = self.generate_comprehensive_report(
            results, performance_analysis, quality_analysis, recommendations
        )
        self.save_comprehensive_report(comprehensive_report)
        
        # 7. サマリー表示
        self.print_evaluation_summary(comprehensive_report)
    
    def print_evaluation_summary(self, report: Dict) -> None:
        """評価サマリー表示"""
        
        print("\n" + "="*60)
        print("🔥 NKAT Integration Evaluation Summary")
        print("="*60)
        
        exec_summary = report['executive_summary']
        
        print(f"📊 Overall Status: {exec_summary['overall_status'].upper()}")
        print(f"📈 Quality Score: {report['detailed_analysis']['quality_assessment']['overall_score']:.3f}/1.0")
        print(f"🔍 Data Confidence: {report['detailed_analysis']['performance_accuracy']['confidence_level']:.1%}")
        
        print(f"\n✅ Strengths:")
        for strength in exec_summary['strengths'][:3]:
            print(f"   • {strength}")
        
        if exec_summary['critical_issues']:
            print(f"\n⚠️ Critical Issues:")
            for issue in exec_summary['critical_issues']:
                print(f"   • {issue}")
        
        print(f"\n🎯 High Priority Recommendations:")
        high_priority_recs = [rec for rec in report['recommendations'] if rec['priority'] == 'high']
        for rec in high_priority_recs[:2]:
            print(f"   • {rec['title']}")
        
        print(f"\n📁 Reports saved in: {self.output_dir}")
        print("   • nkat_comprehensive_evaluation_report.json")
        print("   • nkat_evaluation_report.md")
        print("   • nkat_integration_evaluation.png")

def main():
    """メイン実行"""
    print("🔍 NKAT Integration Comprehensive Evaluation")
    print("=" * 60)
    
    evaluator = NKATIntegrationEvaluator()
    evaluator.run_comprehensive_evaluation()
    
    print("\n🎉 Comprehensive evaluation completed!")

if __name__ == "__main__":
    main() 