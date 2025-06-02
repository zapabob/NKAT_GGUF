#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-GGUF リポジトリ整理ツール
散在しているファイルを適切なディレクトリに整理
"""

import os
import shutil
from pathlib import Path
import json
import logging
from datetime import datetime

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class RepositoryOrganizer:
    """リポジトリ整理ツール"""
    
    def __init__(self):
        self.root_path = Path(".")
        self.organization_plan = {
            # メインスクリプト群
            'src/': [
                'nkat_*.py',
                'fix_output_stability.py',
                'analyse_optuna_results.py',
                'backend_selector.py',
                'test_nkat_implementation.py'
            ],
            # ビルド関連
            'build_tools/': [
                'cuda_*.py',
                'vs_monitor_and_build.py',
                'integrate_nkat_llama_cpp.py',
                '*build*.bat',
                '*cuda*.bat',
                '*vs*.bat'
            ],
            # 設定ファイル
            'configs/': [
                '*.json',
                'theta_*.bin',
                'theta_*.npz'
            ],
            # ドキュメント
            'docs/': [
                '*.md',
                '*.ipynb'
            ],
            # テスト・ベンチマーク結果
            'results/': [
                'nkat_benchmark_*.json',
                'nkat_comparison_*.json',
                'qwen3_vram_validation.json',
                '*_build_state.json'
            ],
            # 実行スクリプト
            'scripts/run/': [
                'run_*.bat'
            ],
            # セットアップ・インストール
            'scripts/setup/': [
                'auto_setup_*.ps1',
                'install_*.bat',
                'check_*.bat',
                'fix_*.bat'
            ],
            # ログファイル
            'logs/': [
                '*.log'
            ],
            # バックアップ
            'archives/': [
                # 既存のbackupディレクトリ内容は保持
            ]
        }
        
    def create_directory_structure(self):
        """ディレクトリ構造作成"""
        
        logger.info("📁 Creating organized directory structure...")
        
        directories = [
            'src',
            'build_tools', 
            'configs/stability',
            'configs/optimization',
            'docs/guides',
            'docs/api',
            'results/benchmarks',
            'results/validation',
            'scripts/run',
            'scripts/setup',
            'scripts/utils',
            'logs',
            'archives',
            'examples',
            'tests/integration',
            'tests/unit'
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"   ✅ Created: {dir_path}")
    
    def organize_files(self):
        """ファイル整理実行"""
        
        logger.info("🗂️ Organizing files...")
        
        # 移動プラン
        file_moves = {
            # ソースコード
            'src/': [
                'nkat_stability_enhancer.py',
                'nkat_text_stability_validator.py', 
                'fix_output_stability.py',
                'nkat_inference_engine.py',
                'nkat_gguf_converter.py',
                'nkat_auto_optimizer.py',
                'nkat_multi_objective_optimizer.py',
                'nkat_ab_testing.py',
                'nkat_text_generation_optimizer.py',
                'nkat_interactive_demo.py',
                'nkat_validation_suite.py',
                'nkat_performance_monitor.py',
                'analyse_optuna_results.py',
                'backend_selector.py',
                'test_nkat_implementation.py',
                'qwen3_model_analyzer.py',
                'qwen3_nkat_inference_test.py',
                'repository_cleanup_organizer.py'
            ],
            
            # ビルドツール
            'build_tools/': [
                'cuda_direct_build.py',
                'cuda_direct_build_fixed.py', 
                'vs_monitor_and_build.py',
                'integrate_nkat_llama_cpp.py'
            ],
            
            # 設定ファイル
            'configs/': [
                'nkat_kobold_config.json',
                'nkat_kobold_config_final.json',
                'build_state.json',
                'cuda_build_state.json',
                'file_history.json',
                'cleanup_summary.json'
            ],
            
            'configs/stability/': [
                'theta_rank4.bin',
                'theta_rank4.npz'
            ],
            
            # 結果ファイル
            'results/benchmarks/': [
                'nkat_benchmark_512_3.json',
                'nkat_benchmark_512_5.json', 
                'nkat_comparison_256.json',
                'nkat_comparison_512.json',
                'nkat_comparison_1024.json'
            ],
            
            'results/validation/': [
                'qwen3_vram_validation.json'
            ],
            
            # ドキュメント
            'docs/guides/': [
                'NKAT_IMPLEMENTATION_GUIDE.md',
                'NKAT_INTEGRATION_SUMMARY.md',
                'NKAT_KOBOLD_TUNING_GUIDE.md',
                'NKAT_TEXT_QUALITY_OPTIMIZATION_GUIDE.md',
                'llama_cpp_integration_guide.md',
                'cmake_update_guide.md',
                'REPOSITORY_CLEANUP_SUMMARY.md',
                'ORGANIZATION_REPORT.md',
                'PROJECT_SUMMARY.md'
            ],
            
            'docs/': [
                'NKAT_GGUF_Colab_QuickStart.ipynb'
            ],
            
            # 実行スクリプト
            'scripts/run/': [
                'run_stable_inference_high_stability.bat',
                'run_stable_inference_balanced_stability.bat',
                'run_Vecteus-v1-IQ4_XS_optimized.bat',
                'run_nkat_llama_cpp_with_vs2019.bat',
                'run_llama_cpp_integration.bat',
                'run_nkat_integration_test.bat'
            ],
            
            # セットアップスクリプト  
            'scripts/setup/': [
                'auto_setup_rtx_environment.ps1',
                'install_vs_buildtools.bat',
                'check_python_installation.bat',
                'fix_llama_cpp_errors.bat',
                'run_cuda_only_build.bat',
                'run_cuda_with_vs_env.bat'
            ],
            
            # ログファイル
            'logs/': [
                'nkat_stability.log',
                'nkat_text_stability.log',
                'nkat_inference.log',
                'stability_fix.log',
                'comprehensive_rtx_benchmark.log',
                'repository_cleanup.log'
            ]
        }
        
        # ファイル移動実行
        for target_dir, files in file_moves.items():
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            for file_pattern in files:
                if '*' in file_pattern:
                    # ワイルドカード処理
                    for file_path in self.root_path.glob(file_pattern):
                        if file_path.is_file():
                            self.safe_move_file(file_path, target_path / file_path.name)
                else:
                    # 直接ファイル指定
                    file_path = self.root_path / file_pattern
                    if file_path.exists():
                        self.safe_move_file(file_path, target_path / file_path.name)
    
    def safe_move_file(self, source: Path, destination: Path):
        """安全なファイル移動"""
        try:
            if destination.exists():
                logger.warning(f"   ⚠️ Destination exists, skipping: {source} -> {destination}")
                return
                
            shutil.move(str(source), str(destination))
            logger.info(f"   ✅ Moved: {source} -> {destination}")
            
        except Exception as e:
            logger.error(f"   ❌ Failed to move {source}: {e}")
    
    def create_organization_report(self):
        """整理レポート作成"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'organization_summary': {
                'src/': 'Main NKAT implementation scripts',
                'build_tools/': 'Build and compilation utilities',
                'configs/': 'Configuration files and parameters',
                'docs/': 'Documentation and guides',
                'results/': 'Benchmark and validation results',
                'scripts/': 'Execution and setup scripts',
                'logs/': 'Log files',
                'models/': 'Model files and checkpoints',
                'output/': 'Generated outputs and temporary files',
                'tests/': 'Test suites and validation',
                'tools/': 'Additional utilities'
            },
            'key_files': {
                'src/nkat_inference_engine.py': 'Main NKAT inference engine',
                'src/fix_output_stability.py': 'Output stability fixing tool',
                'configs/stability/': 'Stability configuration files',
                'docs/guides/': 'Implementation and usage guides',
                'scripts/run/': 'Ready-to-use execution scripts'
            }
        }
        
        with open('docs/REPOSITORY_ORGANIZATION.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info("📄 Organization report created: docs/REPOSITORY_ORGANIZATION.json")
    
    def run_organization(self):
        """整理実行"""
        
        logger.info("🗂️ Starting repository organization...")
        
        # ディレクトリ構造作成
        self.create_directory_structure()
        
        # ファイル整理
        self.organize_files()
        
        # レポート作成
        self.create_organization_report()
        
        logger.info("✅ Repository organization completed!")

def main():
    organizer = RepositoryOrganizer()
    organizer.run_organization()

if __name__ == "__main__":
    main() 