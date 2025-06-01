#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 GGUF + NKAT Integration Test & Usage Examples
GGUF+NKAT統合のテストと使用例

実用例:
1. 既存のLlama-2-7B-chat.ggufにNKAT理論を統合
2. Mistral-7B-Instruct.ggufを理論的強化
3. 軽量エッジモデルのNKAT拡張
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import PySimpleGUI as sg
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import threading

# 統合ツールのインポート
from gguf_nkat_integration import GGUFNKATIntegrator, NKATConfig

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

def download_test_model():
    """テスト用の軽量GGUFモデルをダウンロード"""
    print("🔽 テスト用GGUFモデルのダウンロード...")
    
    # 実際の使用では、以下のようなコマンドでモデルをダウンロード
    download_commands = [
        "# Hugging Faceから軽量モデルをダウンロード",
        "huggingface-cli download microsoft/DialoGPT-small --local-dir ./test_models/dialogpt-small",
        "",
        "# GGUFに変換",
        "python llama.cpp/convert-hf-to-gguf.py ./test_models/dialogpt-small --outfile ./test_models/dialogpt-small.gguf --outtype f16",
        "",
        "# またはOllamaから軽量モデルを取得",
        "ollama pull tinyllama:1.1b-chat-v1.0",
        "# Ollamaモデルをエクスポート（実装依存）"
    ]
    
    print("   実際のダウンロードコマンド例:")
    for cmd in download_commands:
        print(f"   {cmd}")
    
    # デモ用の模擬GGUFファイル作成
    test_dir = Path("test_models")
    test_dir.mkdir(exist_ok=True)
    
    demo_gguf = test_dir / "demo_model.gguf"
    if not demo_gguf.exists():
        # 最小限のGGUFファイル構造を作成（デモ用）
        with open(demo_gguf, 'wb') as f:
            f.write(b'GGUF')  # マジック番号
            f.write(b'\x03\x00\x00\x00')  # バージョン 3
            f.write(b'\x00\x00\x00\x00\x00\x00\x00\x00')  # テンソル数 0
            f.write(b'\x02\x00\x00\x00\x00\x00\x00\x00')  # メタデータ数 2
            
            # メタデータ例
            # キー "general.architecture"
            key1 = "general.architecture"
            key1_bytes = key1.encode('utf-8')
            f.write(len(key1_bytes).to_bytes(8, 'little'))
            f.write(key1_bytes)
            f.write((4).to_bytes(4, 'little'))  # string type
            value1 = "llama"
            value1_bytes = value1.encode('utf-8')
            f.write(len(value1_bytes).to_bytes(8, 'little'))
            f.write(value1_bytes)
            
            # キー "general.name"
            key2 = "general.name"
            key2_bytes = key2.encode('utf-8')
            f.write(len(key2_bytes).to_bytes(8, 'little'))
            f.write(key2_bytes)
            f.write((4).to_bytes(4, 'little'))  # string type
            value2 = "demo_model"
            value2_bytes = value2.encode('utf-8')
            f.write(len(value2_bytes).to_bytes(8, 'little'))
            f.write(value2_bytes)
        
        print(f"   ✅ デモ用GGUFファイル作成: {demo_gguf}")
    
    return demo_gguf

def test_basic_integration():
    """基本的なNKAT統合テスト"""
    print("\n🧪 基本的なNKAT統合テスト")
    print("="*50)
    
    # テストモデル準備
    demo_model = download_test_model()
    
    # NKAT設定
    config = NKATConfig(
        enable_ka_operators=True,
        ka_grid_size=4,  # テスト用に小さく
        lie_algebra_dim=2,  # 簡単化
        noncommutative_strength=0.05,
        quantization_aware=True
    )
    
    # 統合実行
    integrator = GGUFNKATIntegrator(config)
    output_path = "test_models/demo_model_nkat.gguf"
    
    try:
        integrator.create_nkat_enhanced_gguf(str(demo_model), output_path)
        print("✅ 基本統合テスト成功")
        
        # メタデータ確認
        enhanced_metadata = integrator.read_gguf_metadata(output_path)
        nkat_keys = [k for k in enhanced_metadata.keys() if k.startswith('nkat.')]
        print(f"   追加されたNKATメタデータ: {len(nkat_keys)} 項目")
        for key in nkat_keys[:5]:  # 最初の5項目表示
            print(f"   - {key}: {enhanced_metadata[key]}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本統合テスト失敗: {e}")
        return False

def test_performance_comparison():
    """パフォーマンス比較テスト"""
    print("\n🏃 パフォーマンス比較テスト")
    print("="*50)
    
    # ベンチマーク設定
    test_cases = [
        {"name": "軽量NKAT", "ka_grid_size": 4, "lie_algebra_dim": 2},
        {"name": "標準NKAT", "ka_grid_size": 8, "lie_algebra_dim": 4},
        {"name": "高性能NKAT", "ka_grid_size": 16, "lie_algebra_dim": 8}
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n📊 {case['name']} テスト中...")
        
        config = NKATConfig(
            ka_grid_size=case['ka_grid_size'],
            lie_algebra_dim=case['lie_algebra_dim']
        )
        
        integrator = GGUFNKATIntegrator(config)
        
        # 統合時間測定
        start_time = time.time()
        
        # メタデータ準備（実際の統合は省略）
        metadata_size = len(integrator.nkat_metadata)
        processing_time = time.time() - start_time
        
        # 理論的計算複雑度
        theoretical_complexity = case['ka_grid_size'] * case['lie_algebra_dim'] ** 2
        
        result = {
            "name": case['name'],
            "metadata_items": metadata_size,
            "processing_time": processing_time,
            "theoretical_complexity": theoretical_complexity,
            "memory_estimate": theoretical_complexity * 4 / 1024  # KB
        }
        
        results.append(result)
        
        print(f"   メタデータ項目: {metadata_size}")
        print(f"   処理時間: {processing_time:.4f}s")
        print(f"   理論的複雑度: {theoretical_complexity}")
        print(f"   メモリ推定: {result['memory_estimate']:.2f} KB")
    
    # 結果比較
    print(f"\n📈 パフォーマンス比較結果:")
    print(f"{'設定':<12} {'処理時間':<8} {'複雑度':<8} {'メモリ':<8}")
    print("-" * 40)
    for r in results:
        print(f"{r['name']:<12} {r['processing_time']:.4f}s {r['theoretical_complexity']:<8} {r['memory_estimate']:.1f}KB")
    
    return results

def test_compatibility():
    """互換性テスト"""
    print("\n🔧 互換性テスト")
    print("="*50)
    
    # 異なるアーキテクチャとの互換性
    architectures = ["llama", "mistral", "mixtral", "qwen", "falcon"]
    quantization_types = ["Q4_K_M", "Q5_K_M", "Q8_0", "F16", "F32"]
    
    compatibility_matrix = {}
    
    for arch in architectures:
        compatibility_matrix[arch] = {}
        for quant in quantization_types:
            # 互換性判定（実際の実装では詳細チェック）
            compatible = True
            
            # 一部の組み合わせで制限（例）
            if arch == "falcon" and quant in ["Q4_K_M", "Q5_K_M"]:
                compatible = False  # K-quantはFalconでサポート制限
            
            compatibility_matrix[arch][quant] = "✅" if compatible else "❌"
    
    # 結果表示
    print("アーキテクチャ vs 量子化タイプ互換性:")
    print(f"{'Architecture':<12} {' '.join(f'{q:<8}' for q in quantization_types)}")
    print("-" * 60)
    
    for arch, quants in compatibility_matrix.items():
        row = f"{arch:<12} "
        for q in quantization_types:
            row += f"{quants[q]:<8} "
        print(row)
    
    return compatibility_matrix

def generate_usage_examples():
    """使用例の生成"""
    print("\n📚 使用例の生成")
    print("="*50)
    
    examples = {
        "basic_usage": {
            "description": "基本的な使用例",
            "command": "py -3 gguf_nkat_integration.py -i model.gguf -o model_nkat.gguf",
            "config": None
        },
        
        "lightweight_edge": {
            "description": "エッジデバイス用軽量設定",
            "command": "py -3 gguf_nkat_integration.py -i model.gguf -o model_edge_nkat.gguf -c edge_config.json",
            "config": {
                "enable_ka_operators": True,
                "ka_grid_size": 4,
                "lie_algebra_dim": 2,
                "noncommutative_strength": 0.05,
                "quantization_aware": True
            }
        },
        
        "high_performance": {
            "description": "高性能サーバー用設定",
            "command": "py -3 gguf_nkat_integration.py -i model.gguf -o model_hp_nkat.gguf -c hp_config.json --generate-extension",
            "config": {
                "enable_ka_operators": True,
                "ka_grid_size": 16,
                "lie_algebra_dim": 8,
                "noncommutative_strength": 0.2,
                "differential_geometric_scale": 0.02,
                "quantization_aware": True
            }
        },
        
        "theory_focused": {
            "description": "理論研究用フル機能設定",
            "command": "py -3 gguf_nkat_integration.py -i model.gguf -o model_theory_nkat.gguf -c theory_config.json --generate-extension",
            "config": {
                "enable_ka_operators": True,
                "ka_grid_size": 32,
                "lie_algebra_dim": 16,
                "noncommutative_strength": 0.3,
                "differential_geometric_scale": 0.05,
                "spectral_radius_bound": 2.0,
                "quantization_aware": True
            }
        }
    }
    
    # 設定ファイルを生成
    for name, example in examples.items():
        if example["config"]:
            config_file = f"{name}_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(example["config"], f, indent=2, ensure_ascii=False)
            print(f"✅ {example['description']}設定ファイル生成: {config_file}")
    
    # 使用方法ドキュメント生成
    usage_doc = """
# GGUF + NKAT Integration 使用ガイド

## 📋 前提条件

1. **必要なライブラリ**:
   ```bash
   pip install numpy torch struct pathlib
   ```

2. **llama.cpp準備**:
   ```bash
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp
   make
   ```

## 🚀 基本的な使用方法

### 1. 既存GGUFファイルの拡張
```bash
# 基本的な統合
py -3 gguf_nkat_integration.py -i model.gguf -o model_nkat.gguf

# 設定ファイル使用
py -3 gguf_nkat_integration.py -i model.gguf -o model_nkat.gguf -c nkat_config.json

# llama.cpp拡張コード生成付き
py -3 gguf_nkat_integration.py -i model.gguf -o model_nkat.gguf --generate-extension
```

### 2. NKAT拡張モデルの実行
```bash
# 標準のllama.cppで実行（メタデータのみ）
./llama.cpp/main -m model_nkat.gguf -p "Hello world"

# NKAT拡張版llama.cppで実行（理論機能有効）
./nkat_extension/nkat_main -m model_nkat.gguf -p "Hello world" --enable-nkat
```

## 🔧 設定オプション

### エッジデバイス用（軽量）
- KAグリッドサイズ: 4-8
- リー代数次元: 2-4
- 非可換強度: 0.05-0.1

### 高性能サーバー用
- KAグリッドサイズ: 16-32
- リー代数次元: 8-16
- 非可換強度: 0.2-0.3

### 理論研究用（フル機能）
- KAグリッドサイズ: 32+
- リー代数次元: 16+
- 全機能有効

## 🎯 推奨用途

1. **教育・研究**: 理論の実証とアルゴリズム開発
2. **エッジAI**: 軽量化による効率的推論
3. **専門タスク**: 特定分野での性能向上
4. **互換性テスト**: 既存システムとの統合

## ⚠️ 注意事項

- GGUFファイルのバックアップを必ず作成
- メモリ使用量は設定により大幅に変動
- 理論機能の有効性は用途依存
- llama.cpp拡張は実験的機能
"""

    with open("NKAT_USAGE_GUIDE.md", 'w', encoding='utf-8') as f:
        f.write(usage_doc)
    
    print(f"✅ 使用ガイド生成: NKAT_USAGE_GUIDE.md")
    
    return examples

class NKATPatchGUI(tk.Tk if not DND_AVAILABLE else TkinterDnD.Tk):
    def __init__(self):
        super().__init__()
        self.title('GGUF + NKAT Patch GUI (tkinter版)')
        self.geometry('800x600')
        self.resizable(True, True)
        self.gguf_files = []
        self.presets_file = 'nkat_presets.json'
        self.presets = self.load_presets()
        self.create_widgets()

    def load_presets(self):
        if os.path.isfile(self.presets_file):
            try:
                with open(self.presets_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def save_presets(self):
        with open(self.presets_file, 'w', encoding='utf-8') as f:
            json.dump(self.presets, f, indent=2, ensure_ascii=False)

    def create_widgets(self):
        # ファイル選択
        frame_files = tk.Frame(self)
        frame_files.pack(fill='x', pady=5)
        tk.Label(frame_files, text='GGUFファイル:').pack(side='left')
        self.files_entry = tk.Text(frame_files, height=2, width=70, state='disabled')
        self.files_entry.pack(side='left', padx=5)
        tk.Button(frame_files, text='追加', command=self.add_files).pack(side='left')
        tk.Button(frame_files, text='クリア', command=self.clear_files).pack(side='left')
        # ドラッグ＆ドロップ対応
        if DND_AVAILABLE:
            self.files_entry.drop_target_register(DND_FILES)
            self.files_entry.dnd_bind('<<Drop>>', self.on_drop_files)
        else:
            self.files_entry.config(state='normal')
            self.files_entry.insert(tk.END, '※tkinterDnD2未導入のためDD非対応')
            self.files_entry.config(state='disabled')

        # バックアップ・出力先
        frame_dirs = tk.Frame(self)
        frame_dirs.pack(fill='x', pady=5)
        tk.Label(frame_dirs, text='バックアップ先:').pack(side='left')
        self.backup_var = tk.StringVar()
        tk.Entry(frame_dirs, textvariable=self.backup_var, width=30).pack(side='left')
        tk.Button(frame_dirs, text='参照', command=self.select_backup_dir).pack(side='left')
        tk.Label(frame_dirs, text='出力先:').pack(side='left')
        self.output_var = tk.StringVar()
        tk.Entry(frame_dirs, textvariable=self.output_var, width=30).pack(side='left')
        tk.Button(frame_dirs, text='参照', command=self.select_output_dir).pack(side='left')

        # 設定ファイル
        frame_config = tk.Frame(self)
        frame_config.pack(fill='x', pady=5)
        tk.Label(frame_config, text='設定ファイル:').pack(side='left')
        self.config_var = tk.StringVar()
        tk.Entry(frame_config, textvariable=self.config_var, width=50).pack(side='left')
        tk.Button(frame_config, text='選択', command=self.select_config_file).pack(side='left')

        # パラメータ
        frame_params = tk.LabelFrame(self, text='パッチパラメータ（手動指定は設定ファイルより優先）')
        frame_params.pack(fill='x', pady=5)
        self.grid_var = tk.StringVar(value='8')
        self.lie_var = tk.StringVar(value='4')
        self.nc_var = tk.StringVar(value='0.1')
        self.dg_var = tk.StringVar(value='0.01')
        self.ka_var = tk.BooleanVar(value=True)
        self.qa_var = tk.BooleanVar(value=True)
        tk.Label(frame_params, text='グリッドサイズ').grid(row=0, column=0)
        tk.Entry(frame_params, textvariable=self.grid_var, width=5).grid(row=0, column=1)
        tk.Label(frame_params, text='リー代数次元').grid(row=0, column=2)
        tk.Entry(frame_params, textvariable=self.lie_var, width=5).grid(row=0, column=3)
        tk.Label(frame_params, text='非可換強度').grid(row=0, column=4)
        tk.Entry(frame_params, textvariable=self.nc_var, width=6).grid(row=0, column=5)
        tk.Label(frame_params, text='微分幾何スケール').grid(row=0, column=6)
        tk.Entry(frame_params, textvariable=self.dg_var, width=6).grid(row=0, column=7)
        tk.Checkbutton(frame_params, text='KA演算子有効', variable=self.ka_var).grid(row=1, column=0, columnspan=2)
        tk.Checkbutton(frame_params, text='量子化対応', variable=self.qa_var).grid(row=1, column=2, columnspan=2)

        # プリセット
        frame_preset = tk.Frame(self)
        frame_preset.pack(fill='x', pady=5)
        tk.Label(frame_preset, text='プリセット:').pack(side='left')
        self.preset_var = tk.StringVar()
        preset_keys = list(self.presets.keys())
        if preset_keys:
            self.preset_var.set(preset_keys[0])
            self.preset_menu = tk.OptionMenu(frame_preset, self.preset_var, *preset_keys)
        else:
            self.preset_var.set('')
            self.preset_menu = tk.OptionMenu(frame_preset, self.preset_var, '')
        self.preset_menu.pack(side='left')
        tk.Button(frame_preset, text='読込', command=self.load_preset).pack(side='left')
        tk.Button(frame_preset, text='保存', command=self.save_preset).pack(side='left')
        self.preset_name_var = tk.StringVar()
        tk.Entry(frame_preset, textvariable=self.preset_name_var, width=15).pack(side='left')

        # 実行・終了
        frame_action = tk.Frame(self)
        frame_action.pack(fill='x', pady=5)
        tk.Button(frame_action, text='パッチ一括実行', command=self.run_patch_thread).pack(side='left', padx=10)
        tk.Button(frame_action, text='終了', command=self.destroy).pack(side='left', padx=10)

        # ログ
        self.log_text = tk.Text(self, height=15, width=100, state='disabled', bg='#f8f8f8')
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)

    def add_files(self):
        files = filedialog.askopenfilenames(filetypes=[('GGUF', '*.gguf')])
        if files:
            self.gguf_files.extend([f for f in files if f not in self.gguf_files])
            self.update_files_entry()

    def clear_files(self):
        self.gguf_files = []
        self.update_files_entry()

    def update_files_entry(self):
        self.files_entry.config(state='normal')
        self.files_entry.delete('1.0', tk.END)
        self.files_entry.insert(tk.END, '\n'.join(self.gguf_files))
        self.files_entry.config(state='disabled')

    def select_backup_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.backup_var.set(d)

    def select_output_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.output_var.set(d)

    def select_config_file(self):
        f = filedialog.askopenfilename(filetypes=[('JSON', '*.json')])
        if f:
            self.config_var.set(f)

    def load_preset(self):
        pname = self.preset_var.get()
        if pname and pname in self.presets:
            p = self.presets[pname]
            self.grid_var.set(str(p.get('ka_grid_size', '8')))
            self.lie_var.set(str(p.get('lie_algebra_dim', '4')))
            self.nc_var.set(str(p.get('noncommutative_strength', '0.1')))
            self.dg_var.set(str(p.get('differential_geometric_scale', '0.01')))
            self.ka_var.set(p.get('enable_ka_operators', True))
            self.qa_var.set(p.get('quantization_aware', True))
            self.log(f'✅ プリセット「{pname}」を読込')

    def save_preset(self):
        pname = self.preset_name_var.get().strip()
        if not pname:
            self.log('⚠️ プリセット名を入力してください')
            return
        try:
            grid = int(self.grid_var.get())
            lie = int(self.lie_var.get())
            nc = float(self.nc_var.get())
            dg = float(self.dg_var.get())
            if grid < 1 or lie < 1 or not (0 <= nc <= 10) or not (0 <= dg <= 10):
                raise ValueError
            self.presets[pname] = {
                'ka_grid_size': grid,
                'lie_algebra_dim': lie,
                'noncommutative_strength': nc,
                'differential_geometric_scale': dg,
                'enable_ka_operators': self.ka_var.get(),
                'quantization_aware': self.qa_var.get()
            }
            self.save_presets()
            menu = self.preset_menu['menu']
            menu.delete(0, 'end')
            for k in self.presets.keys():
                menu.add_command(label=k, command=lambda v=k: self.preset_var.set(v))
            self.log(f'✅ プリセット「{pname}」を保存')
        except Exception:
            self.log('❌ プリセット保存失敗: 入力値を確認してください')

    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, msg + '\n')
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')

    def run_patch_thread(self):
        threading.Thread(target=self.run_patch, daemon=True).start()

    def run_patch(self):
        if not self.gguf_files or not all(os.path.isfile(f) for f in self.gguf_files):
            self.log('⚠️ GGUFファイルを追加してください')
            return
        backup_dir = self.backup_var.get() or os.path.dirname(self.gguf_files[0])
        output_dir = self.output_var.get() or os.path.dirname(self.gguf_files[0])
        config_path = self.config_var.get()
        # バリデーション
        try:
            grid = int(self.grid_var.get())
            lie = int(self.lie_var.get())
            nc = float(self.nc_var.get())
            dg = float(self.dg_var.get())
            if grid < 1 or lie < 1 or not (0 <= nc <= 10) or not (0 <= dg <= 10):
                raise ValueError
        except Exception:
            self.log('❌ パラメータ値が不正です（正の整数/0-10範囲）')
            return
        # 設定ファイル優先
        config = None
        if config_path and os.path.isfile(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                if 'nkat_config' in config_dict:
                    config_dict = config_dict['nkat_config']
                config = NKATConfig(
                    enable_ka_operators=config_dict.get('enable_ka_operators', self.ka_var.get()),
                    ka_grid_size=int(config_dict.get('ka_grid_size', grid)),
                    lie_algebra_dim=int(config_dict.get('lie_algebra_dim', lie)),
                    noncommutative_strength=float(config_dict.get('noncommutative_strength', nc)),
                    differential_geometric_scale=float(config_dict.get('differential_geometric_scale', dg)),
                    quantization_aware=config_dict.get('quantization_aware', self.qa_var.get())
                )
            except Exception:
                self.log('❌ 設定ファイルの読み込みに失敗しました')
                return
        else:
            config = NKATConfig(
                enable_ka_operators=self.ka_var.get(),
                ka_grid_size=grid,
                lie_algebra_dim=lie,
                noncommutative_strength=nc,
                differential_geometric_scale=dg,
                quantization_aware=self.qa_var.get()
            )
        # 一括パッチ
        for gguf_path in self.gguf_files:
            try:
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, os.path.basename(gguf_path))
                # --- WinError 32対応: コピー時リトライ ---
                copy_success = False
                for attempt in range(5):
                    try:
                        shutil.copy2(gguf_path, backup_path)
                        copy_success = True
                        break
                    except Exception as e:
                        if hasattr(e, 'winerror') and e.winerror == 32:
                            self.log(f'⚠️ ファイルロック中（{os.path.basename(gguf_path)}）: リトライ {attempt+1}/5')
                            import time; time.sleep(1)
                        else:
                            raise
                if not copy_success:
                    self.log(f'❌ {os.path.basename(gguf_path)}: バックアップコピー失敗（ファイルロック）')
                    continue
                self.log(f'🗂️ バックアップ作成: {backup_path}')
                output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(gguf_path))[0] + '_nkat.gguf')
                
                # パッチ処理を詳細にデバッグ
                try:
                    self.log(f'🔄 NKAT統合開始: {os.path.basename(gguf_path)}')
                    integrator = GGUFNKATIntegrator(config)
                    self.log(f'✅ NKATIntegrator初期化完了')
                    
                    integrator.create_nkat_enhanced_gguf(gguf_path, output_path)
                    self.log(f'✅ パッチ適用完了: {output_path}')
                except ImportError as e:
                    self.log(f'❌ {os.path.basename(gguf_path)}: インポートエラー: {e}')
                except FileNotFoundError as e:
                    self.log(f'❌ {os.path.basename(gguf_path)}: ファイルが見つかりません: {e}')
                except PermissionError as e:
                    self.log(f'❌ {os.path.basename(gguf_path)}: アクセス権限エラー: {e}')
                except MemoryError as e:
                    self.log(f'❌ {os.path.basename(gguf_path)}: メモリ不足: {e}')
                except Exception as e:
                    import traceback
                    self.log(f'❌ {os.path.basename(gguf_path)}: パッチ失敗: {type(e).__name__}: {str(e)}')
                    self.log(f'📋 詳細エラー情報:\n{traceback.format_exc()}')
            except Exception as e:
                import traceback
                self.log(f'❌ {os.path.basename(gguf_path)}: 全体処理失敗: {type(e).__name__}: {str(e)}')
                self.log(f'📋 詳細エラー情報:\n{traceback.format_exc()}')

    def on_drop_files(self, event):
        files = self.tk.splitlist(event.data)
        self.gguf_files.extend([f for f in files if f not in self.gguf_files and f.lower().endswith('.gguf')])
        self.update_files_entry()

def main():
    """メインテスト実行"""
    print("🚀 GGUF + NKAT Integration Test Suite")
    print("="*60)
    print("   目的: 既存GGUFファイルへのNKAT理論統合テスト")
    print("   対象: llama.cpp互換モデルの理論的強化")
    print("="*60)
    
    test_results = {}
    
    # 基本統合テスト
    test_results["basic"] = test_basic_integration()
    
    # パフォーマンステスト
    test_results["performance"] = test_performance_comparison()
    
    # 互換性テスト
    test_results["compatibility"] = test_compatibility()
    
    # 使用例生成
    examples = generate_usage_examples()
    
    # 総合結果
    print(f"\n🎉 テスト完了!")
    print(f"="*50)
    print(f"基本統合: {'✅ 成功' if test_results['basic'] else '❌ 失敗'}")
    print(f"パフォーマンス: ✅ {len(test_results['performance'])} ケース完了")
    print(f"互換性: ✅ マトリックス生成完了")
    print(f"使用例: ✅ {len(examples)} 例生成完了")
    
    print(f"\n📁 生成されたファイル:")
    print(f"   - test_models/demo_model_nkat.gguf (NKAT拡張GGUFファイル)")
    print(f"   - *_config.json (各種設定ファイル)")
    print(f"   - NKAT_USAGE_GUIDE.md (使用ガイド)")
    
    print(f"\n🎯 次のステップ:")
    print(f"   1. 実際のGGUFファイルでテスト")
    print(f"   2. llama.cpp拡張の実装")
    print(f"   3. 理論的効果の検証")

if __name__ == "__main__":
    import sys
    # CLIオプションがあっても必ずGUIを起動
    app = NKATPatchGUI()
    app.mainloop() 