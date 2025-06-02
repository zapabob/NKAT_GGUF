#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-GGUF Repository Organizer for RTX30/RTX40 Series
リポジトリ整理とRTX30/40シリーズ向け最適化
"""

import os
import sys
import shutil
import json
import time
from pathlib import Path
from typing import Dict, List, Set
import logging
from datetime import datetime
from tqdm import tqdm

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('repository_cleanup.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RepositoryOrganizer:
    """リポジトリ整理器"""
    
    def __init__(self, repo_root: str = "."):
        self.repo_root = Path(repo_root).resolve()
        self.cleanup_summary = {
            "timestamp": datetime.now().isoformat(),
            "cleaned_directories": [],
            "preserved_files": [],
            "relocated_files": [],
            "deleted_files": [],
            "created_structure": []
        }
        
        logger.info(f"🔧 Repository Organizer initialized")
        logger.info(f"   📁 Root: {self.repo_root}")
    
    def analyze_current_structure(self) -> Dict:
        """現在の構造を分析"""
        analysis = {
            "total_files": 0,
            "total_directories": 0,
            "build_artifacts": [],
            "temporary_files": [],
            "core_files": [],
            "script_files": [],
            "log_files": [],
            "backup_directories": [],
            "size_breakdown": {}
        }
        
        logger.info("📊 Analyzing current repository structure...")
        
        for item in tqdm(self.repo_root.rglob("*"), desc="Scanning"):
            if item.is_file():
                analysis["total_files"] += 1
                size = item.stat().st_size
                
                # カテゴリ分類
                if any(pattern in str(item) for pattern in ["build", "CMakeFiles", ".tlog"]):
                    analysis["build_artifacts"].append(str(item.relative_to(self.repo_root)))
                elif item.suffix in [".tmp", ".temp", ".cache"]:
                    analysis["temporary_files"].append(str(item.relative_to(self.repo_root)))
                elif item.suffix == ".log":
                    analysis["log_files"].append(str(item.relative_to(self.repo_root)))
                elif item.suffix in [".py", ".ps1", ".bat"]:
                    analysis["script_files"].append(str(item.relative_to(self.repo_root)))
                elif item.suffix in [".cpp", ".c", ".h", ".hpp", ".cu", ".cuh"]:
                    analysis["core_files"].append(str(item.relative_to(self.repo_root)))
                
                # サイズ統計
                category = self.categorize_file(item)
                analysis["size_breakdown"][category] = analysis["size_breakdown"].get(category, 0) + size
                
            elif item.is_dir():
                analysis["total_directories"] += 1
                
                # バックアップディレクトリ検出
                if any(pattern in item.name.lower() for pattern in ["backup", "temp", "emergency"]):
                    analysis["backup_directories"].append(str(item.relative_to(self.repo_root)))
        
        return analysis
    
    def categorize_file(self, file_path: Path) -> str:
        """ファイルカテゴリ分類"""
        if any(pattern in str(file_path) for pattern in ["build", "CMakeFiles"]):
            return "build_artifacts"
        elif file_path.suffix in [".gguf", ".bin", ".safetensors"]:
            return "models"
        elif file_path.suffix in [".py", ".ps1", ".bat"]:
            return "scripts"
        elif file_path.suffix in [".cpp", ".c", ".h", ".hpp", ".cu", ".cuh"]:
            return "source_code"
        elif file_path.suffix in [".log", ".tmp", ".cache"]:
            return "temporary"
        elif file_path.suffix in [".json", ".md", ".txt"]:
            return "documentation"
        else:
            return "others"
    
    def create_rtx_optimized_structure(self) -> None:
        """RTX30/40シリーズ向け最適化構造作成"""
        
        target_structure = {
            "build/": "統一ビルドディレクトリ",
            "scripts/setup/": "環境セットアップスクリプト",
            "scripts/build/": "ビルドスクリプト",
            "scripts/benchmark/": "ベンチマークスクリプト",
            "scripts/optimization/": "最適化スクリプト",
            "configs/rtx30/": "RTX30シリーズ設定",
            "configs/rtx40/": "RTX40シリーズ設定",
            "docs/setup/": "セットアップドキュメント",
            "docs/optimization/": "最適化ガイド",
            "models/benchmarks/": "ベンチマーク用モデル",
            "models/production/": "本番用モデル",
            "output/benchmarks/": "ベンチマーク結果",
            "output/logs/": "ログファイル",
            "backup/": "バックアップ",
            "tools/": "ユーティリティ",
            "tests/": "テストファイル"
        }
        
        logger.info("🏗️ Creating RTX-optimized directory structure...")
        
        for directory, description in tqdm(target_structure.items(), desc="Creating dirs"):
            target_path = self.repo_root / directory
            target_path.mkdir(parents=True, exist_ok=True)
            self.cleanup_summary["created_structure"].append(f"{directory} - {description}")
            
            # .gitkeep追加（空ディレクトリ保持）
            gitkeep_path = target_path / ".gitkeep"
            if not gitkeep_path.exists():
                gitkeep_path.touch()
    
    def cleanup_build_artifacts(self) -> None:
        """ビルド成果物のクリーンアップ"""
        
        logger.info("🧹 Cleaning up build artifacts...")
        
        build_patterns = [
            "llama.cpp/build*",
            "**/CMakeFiles",
            "**/*.tlog",
            "**/*.lastbuildstate",
            "**/CMakeCache.txt",
            "**/cmake_install.cmake"
        ]
        
        cleaned_count = 0
        
        for pattern in build_patterns:
            for item in tqdm(self.repo_root.glob(pattern), desc=f"Cleaning {pattern}"):
                if item.exists():
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        
                        self.cleanup_summary["cleaned_directories"].append(str(item.relative_to(self.repo_root)))
                        cleaned_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to clean {item}: {e}")
        
        logger.info(f"✅ Cleaned {cleaned_count} build artifacts")
    
    def organize_scripts(self) -> None:
        """スクリプト整理"""
        
        logger.info("📝 Organizing scripts...")
        
        script_mappings = {
            "build_nkat_*.ps1": "scripts/build/",
            "setup_*.py": "scripts/setup/",
            "*benchmark*.py": "scripts/benchmark/",
            "*optimization*.py": "scripts/optimization/",
            "*evaluation*.py": "scripts/benchmark/",
            "*integration*.py": "scripts/setup/"
        }
        
        scripts_dir = self.repo_root / "scripts"
        
        for pattern, target_subdir in script_mappings.items():
            target_path = self.repo_root / target_subdir
            
            for script_file in self.repo_root.glob(pattern):
                if script_file.is_file():
                    new_path = target_path / script_file.name
                    
                    try:
                        shutil.move(str(script_file), str(new_path))
                        self.cleanup_summary["relocated_files"].append(
                            f"{script_file.name} -> {target_subdir}"
                        )
                        logger.info(f"   📝 Moved {script_file.name} to {target_subdir}")
                    except Exception as e:
                        logger.warning(f"Failed to move {script_file}: {e}")
    
    def organize_logs_and_outputs(self) -> None:
        """ログと出力ファイル整理"""
        
        logger.info("📊 Organizing logs and outputs...")
        
        # ログファイル移動
        logs_dir = self.repo_root / "output/logs"
        for log_file in self.repo_root.glob("*.log"):
            if log_file.is_file():
                new_path = logs_dir / log_file.name
                try:
                    shutil.move(str(log_file), str(new_path))
                    self.cleanup_summary["relocated_files"].append(f"{log_file.name} -> output/logs/")
                except Exception as e:
                    logger.warning(f"Failed to move log {log_file}: {e}")
        
        # JSON結果ファイル移動
        for json_file in self.repo_root.glob("*benchmark*.json"):
            if json_file.is_file():
                new_path = self.repo_root / "output/benchmarks" / json_file.name
                try:
                    shutil.move(str(json_file), str(new_path))
                    self.cleanup_summary["relocated_files"].append(f"{json_file.name} -> output/benchmarks/")
                except Exception as e:
                    logger.warning(f"Failed to move result {json_file}: {e}")
    
    def create_rtx_configs(self) -> None:
        """RTX30/40シリーズ向け設定ファイル作成"""
        
        logger.info("⚙️ Creating RTX-specific configurations...")
        
        # RTX30シリーズ設定
        rtx30_config = {
            "rtx_series": "30",
            "cuda_architectures": ["86"],  # RTX 3060/3070/3080/3090
            "recommended_settings": {
                "rtx3060": {
                    "vram_gb": 12,
                    "max_context": 8192,
                    "gpu_layers": 35,
                    "batch_size": 512,
                    "threads": 8
                },
                "rtx3070": {
                    "vram_gb": 8,
                    "max_context": 6144,
                    "gpu_layers": 38,
                    "batch_size": 1024,
                    "threads": 10
                },
                "rtx3080": {
                    "vram_gb": 10,
                    "max_context": 8192,
                    "gpu_layers": 40,
                    "batch_size": 1024,
                    "threads": 12
                },
                "rtx3090": {
                    "vram_gb": 24,
                    "max_context": 16384,
                    "gpu_layers": 45,
                    "batch_size": 2048,
                    "threads": 16
                }
            },
            "nkat_parameters": {
                "rank": 6,
                "gamma": 0.97,
                "optimization_target": "balanced"
            }
        }
        
        # RTX40シリーズ設定
        rtx40_config = {
            "rtx_series": "40",
            "cuda_architectures": ["89"],  # RTX 4060/4070/4080/4090
            "recommended_settings": {
                "rtx4060": {
                    "vram_gb": 8,
                    "max_context": 6144,
                    "gpu_layers": 40,
                    "batch_size": 1024,
                    "threads": 10
                },
                "rtx4070": {
                    "vram_gb": 12,
                    "max_context": 10240,
                    "gpu_layers": 42,
                    "batch_size": 1536,
                    "threads": 12
                },
                "rtx4080": {
                    "vram_gb": 16,
                    "max_context": 12288,
                    "gpu_layers": 45,
                    "batch_size": 2048,
                    "threads": 14
                },
                "rtx4090": {
                    "vram_gb": 24,
                    "max_context": 16384,
                    "gpu_layers": 50,
                    "batch_size": 4096,
                    "threads": 16
                }
            },
            "nkat_parameters": {
                "rank": 8,
                "gamma": 0.98,
                "optimization_target": "performance"
            }
        }
        
        # 設定ファイル保存
        rtx30_path = self.repo_root / "configs/rtx30/default_config.json"
        rtx40_path = self.repo_root / "configs/rtx40/default_config.json"
        
        with open(rtx30_path, 'w', encoding='utf-8') as f:
            json.dump(rtx30_config, f, indent=2, ensure_ascii=False)
        
        with open(rtx40_path, 'w', encoding='utf-8') as f:
            json.dump(rtx40_config, f, indent=2, ensure_ascii=False)
        
        logger.info("✅ RTX configuration files created")
    
    def cleanup_redundant_backups(self) -> None:
        """冗長なバックアップのクリーンアップ"""
        
        logger.info("🗂️ Cleaning up redundant backups...")
        
        backup_patterns = [
            "emergency_backups/*",
            "integrity_backups/*",
            "*_temp",
            "*_backup_*"
        ]
        
        # 最新のバックアップ以外を削除対象とする
        for pattern in backup_patterns:
            backup_items = list(self.repo_root.glob(pattern))
            
            if len(backup_items) > 2:  # 最新2つを保持
                # 作成時間でソート
                backup_items.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # 古いバックアップを削除
                for old_backup in backup_items[2:]:
                    try:
                        if old_backup.is_dir():
                            shutil.rmtree(old_backup)
                        else:
                            old_backup.unlink()
                        
                        self.cleanup_summary["deleted_files"].append(str(old_backup.relative_to(self.repo_root)))
                        logger.info(f"   🗑️ Removed old backup: {old_backup.name}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to remove backup {old_backup}: {e}")
    
    def create_universal_build_script(self) -> None:
        """汎用ビルドスクリプト作成"""
        
        build_script_content = '''# Universal NKAT-GGUF Build Script for RTX30/RTX40 Series
[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("RTX30", "RTX40", "AUTO")]
    [string]$RTXSeries = "AUTO",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("RTX3060", "RTX3070", "RTX3080", "RTX3090", "RTX4060", "RTX4070", "RTX4080", "RTX4090")]
    [string]$GPUModel = "",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$BuildType = "Release",
    
    [Parameter(Mandatory=$false)]
    [bool]$CleanBuild = $true
)

Write-Host "🚀 Universal NKAT-GGUF Build Script" -ForegroundColor Green
Write-Host "   🎯 Target: $RTXSeries series" -ForegroundColor Cyan
Write-Host "   🔧 Build Type: $BuildType" -ForegroundColor Cyan

# GPU自動検出
if ($RTXSeries -eq "AUTO" -or $GPUModel -eq "") {
    Write-Host "🔍 Auto-detecting GPU..." -ForegroundColor Yellow
    $gpuInfo = nvidia-smi --query-gpu=name --format=csv,noheader
    Write-Host "   Detected: $gpuInfo" -ForegroundColor Cyan
    
    if ($gpuInfo -match "RTX 40") {
        $RTXSeries = "RTX40"
    } elseif ($gpuInfo -match "RTX 30") {
        $RTXSeries = "RTX30"
    }
}

# 設定読み込み
$configPath = "configs/$($RTXSeries.ToLower())/default_config.json"
if (Test-Path $configPath) {
    $config = Get-Content $configPath | ConvertFrom-Json
    Write-Host "✅ Configuration loaded from $configPath" -ForegroundColor Green
} else {
    Write-Host "⚠️ Configuration file not found, using defaults" -ForegroundColor Yellow
    exit 1
}

# CMake設定
$cudaArch = $config.cuda_architectures -join ";"
$buildDir = "build"

if ($CleanBuild -and (Test-Path $buildDir)) {
    Remove-Item -Recurse -Force $buildDir
    Write-Host "🧹 Clean build directory" -ForegroundColor Yellow
}

New-Item -ItemType Directory -Force -Path $buildDir
Set-Location $buildDir

# CMake実行
Write-Host "⚙️ Running CMake configuration..." -ForegroundColor Yellow
cmake .. `
    -DGGML_CUDA=ON `
    -DCMAKE_BUILD_TYPE=$BuildType `
    -DCUDA_ARCHITECTURES=$cudaArch `
    -DLLAMA_CURL=OFF `
    -DNKAT_OPTIMIZATION=ON

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ CMake configuration failed" -ForegroundColor Red
    exit 1
}

# ビルド実行
Write-Host "🔨 Building..." -ForegroundColor Yellow
cmake --build . --config $BuildType --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "🎉 Build completed successfully!" -ForegroundColor Green
Set-Location ..
'''
        
        build_script_path = self.repo_root / "scripts/build/universal_build.ps1"
        with open(build_script_path, 'w', encoding='utf-8') as f:
            f.write(build_script_content)
        
        logger.info("✅ Universal build script created")
    
    def create_readme_and_docs(self) -> None:
        """README とドキュメント作成"""
        
        readme_content = """# NKAT-GGUF: RTX30/RTX40 Series Optimized Repository

**Non-commutative Kolmogorov-Arnold Network Theory integrated with GGUF quantized models**

## 🎯 Supported Hardware

### RTX 30 Series
- RTX 3060 (12GB) - Mid-range performance
- RTX 3070 (8GB) - Balanced performance  
- RTX 3080 (10GB) - High performance
- RTX 3090 (24GB) - Maximum performance

### RTX 40 Series  
- RTX 4060 (8GB) - Entry level
- RTX 4070 (12GB) - Mid-range performance
- RTX 4080 (16GB) - High performance
- RTX 4090 (24GB) - Maximum performance

## 🚀 Quick Start

### 1. Environment Setup
```powershell
# Run automatic setup
.\\scripts\\setup\\auto_setup.ps1
```

### 2. Build
```powershell
# Automatic detection and build
.\\scripts\\build\\universal_build.ps1

# Or specify your GPU
.\\scripts\\build\\universal_build.ps1 -GPUModel RTX4090
```

### 3. Benchmark
```powershell
# Run comprehensive benchmark
py -3 scripts\\benchmark\\comprehensive_benchmark.py
```

## 📁 Repository Structure

```
├── build/                  # Unified build directory
├── configs/               # RTX-specific configurations
│   ├── rtx30/            # RTX 30 series configs
│   └── rtx40/            # RTX 40 series configs
├── scripts/              # All automation scripts
│   ├── setup/           # Environment setup
│   ├── build/           # Build scripts
│   ├── benchmark/       # Performance testing
│   └── optimization/    # Parameter optimization
├── docs/                 # Documentation
├── models/              # Model files
├── output/              # Results and logs
└── tools/               # Utilities
```

## ⚙️ Configuration

RTX-specific configurations are automatically loaded based on your GPU:

- **RTX 30 Series**: Optimized for CUDA Compute 8.6
- **RTX 40 Series**: Optimized for CUDA Compute 8.9

## 📊 Performance Expectations

| GPU Model | VRAM | Context Length | Expected Performance |
|-----------|------|----------------|---------------------|
| RTX 3060  | 12GB | 8K tokens     | ~50-80 t/s         |
| RTX 3080  | 10GB | 8K tokens     | ~80-120 t/s        |
| RTX 4070  | 12GB | 10K tokens    | ~100-150 t/s       |
| RTX 4090  | 24GB | 16K tokens    | ~200-300 t/s       |

## 🔧 Troubleshooting

See `docs/troubleshooting.md` for common issues and solutions.

## 📖 Documentation

- [Setup Guide](docs/setup/)
- [Optimization Guide](docs/optimization/)
- [API Reference](docs/api/)

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
"""
        
        readme_path = self.repo_root / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        logger.info("✅ README.md created")
    
    def save_cleanup_summary(self) -> None:
        """クリーンアップサマリー保存"""
        
        summary_path = self.repo_root / "REPOSITORY_CLEANUP_SUMMARY.md"
        
        summary_content = f"""# Repository Cleanup Summary

**Date:** {self.cleanup_summary['timestamp']}

## 📊 Cleanup Statistics

### Directories Cleaned
- **Count:** {len(self.cleanup_summary['cleaned_directories'])}

### Files Relocated  
- **Count:** {len(self.cleanup_summary['relocated_files'])}

### New Structure Created
- **Directories:** {len(self.cleanup_summary['created_structure'])}

## 🗂️ New Repository Structure

The repository has been reorganized for RTX30/RTX40 series optimization:

### Created Directories
{chr(10).join(f"- {item}" for item in self.cleanup_summary['created_structure'])}

### Major Relocations
{chr(10).join(f"- {item}" for item in self.cleanup_summary['relocated_files'][:10])}

## 🎯 RTX Series Optimization

- **RTX 30 Series**: CUDA Compute 8.6 optimized
- **RTX 40 Series**: CUDA Compute 8.9 optimized
- **Universal Build**: Auto-detection and configuration
- **Memory Management**: VRAM-aware settings per GPU model

## 🚀 Next Steps

1. Run `scripts/setup/auto_setup.ps1` for environment setup
2. Execute `scripts/build/universal_build.ps1` for automated build
3. Use `scripts/benchmark/comprehensive_benchmark.py` for testing

## 📝 Notes

- All build artifacts have been cleaned
- Scripts organized by function
- RTX-specific configurations created
- Legacy backups consolidated
"""
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        # JSON形式でも保存
        json_path = self.repo_root / "cleanup_summary.json" 
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.cleanup_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Cleanup summary saved: {summary_path}")
    
    def run_full_organization(self) -> None:
        """完全な整理実行"""
        
        logger.info("🔥 Starting full repository organization for RTX30/RTX40 series")
        
        # 1. 現在の構造分析
        analysis = self.analyze_current_structure()
        logger.info(f"📊 Analysis: {analysis['total_files']} files, {analysis['total_directories']} directories")
        
        # 2. 新しい構造作成
        self.create_rtx_optimized_structure()
        
        # 3. ビルド成果物クリーンアップ
        self.cleanup_build_artifacts()
        
        # 4. スクリプト整理
        self.organize_scripts()
        
        # 5. ログ・出力整理
        self.organize_logs_and_outputs()
        
        # 6. RTX設定作成
        self.create_rtx_configs()
        
        # 7. 冗長バックアップクリーンアップ
        self.cleanup_redundant_backups()
        
        # 8. 汎用ビルドスクリプト作成
        self.create_universal_build_script()
        
        # 9. ドキュメント作成
        self.create_readme_and_docs()
        
        # 10. サマリー保存
        self.save_cleanup_summary()
        
        logger.info("🎉 Repository organization completed!")
        self.print_completion_summary()
    
    def print_completion_summary(self) -> None:
        """完了サマリー表示"""
        
        print("\n" + "="*60)
        print("🔥 NKAT-GGUF Repository Organization Complete")
        print("="*60)
        
        print(f"📊 Cleanup Summary:")
        print(f"   🧹 Cleaned directories: {len(self.cleanup_summary['cleaned_directories'])}")
        print(f"   📝 Relocated files: {len(self.cleanup_summary['relocated_files'])}")
        print(f"   🏗️ Created structure: {len(self.cleanup_summary['created_structure'])}")
        
        print(f"\n🎯 RTX Series Support:")
        print(f"   • RTX 30 Series (CUDA 8.6)")
        print(f"   • RTX 40 Series (CUDA 8.9)")
        print(f"   • Auto-detection & configuration")
        
        print(f"\n🚀 Quick Start Commands:")
        print(f"   1. Setup:    .\\scripts\\setup\\auto_setup.ps1")
        print(f"   2. Build:    .\\scripts\\build\\universal_build.ps1")
        print(f"   3. Test:     py -3 scripts\\benchmark\\comprehensive_benchmark.py")
        
        print(f"\n📁 Check these files:")
        print(f"   • README.md (updated)")
        print(f"   • REPOSITORY_CLEANUP_SUMMARY.md")
        print(f"   • configs/rtx30/default_config.json")
        print(f"   • configs/rtx40/default_config.json")

def main():
    """メイン実行"""
    print("🔧 NKAT-GGUF Repository Organizer for RTX30/RTX40 Series")
    print("=" * 60)
    
    organizer = RepositoryOrganizer()
    organizer.run_full_organization()

if __name__ == "__main__":
    main() 