#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT Interactive Demo
実戦チャットテスト & リアルタイムチューニング環境
"""

import os
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging
from tqdm import tqdm

# Import NKAT components
from nkat_inference_engine import NKATInferenceEngine
from nkat_gguf_converter import NKATGGUFConverter

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NKATInteractiveDemo:
    """NKAT対話型デモ環境"""
    
    def __init__(self):
        self.engine = None
        self.current_model = None
        self.performance_history = []
        self.test_prompts = [
            "Hello, how are you today?",
            "Write a Python function to calculate fibonacci numbers.",
            "Explain the concept of neural networks in simple terms.",
            "Tell me a short story about a robot learning to love.",
            "What are the advantages of quantum computing?"
        ]
    
    def show_banner(self):
        """開始バナー表示"""
        print("\n" + "="*60)
        print("🚀 NKAT Interactive Demo & Chat Test Environment 🚀")
        print("="*60)
        print("📊 リアルタイムパフォーマンス測定 & パラメータ調整")
        print("🤖 実戦チャットテスト環境")
        print("🔧 RTX 3080 最適化済み")
        print("="*60 + "\n")
    
    def load_model(self, model_path: str) -> bool:
        """モデル読み込み"""
        try:
            logger.info(f"📂 モデル読み込み: {model_path}")
            self.engine = NKATInferenceEngine(model_path, use_cuda=True)
            success = self.engine.load_model()
            
            if success:
                self.current_model = model_path
                logger.info(f"✅ モデル読み込み完了")
                return True
            else:
                logger.error(f"❌ モデル読み込み失敗")
                return False
                
        except Exception as e:
            logger.error(f"❌ エラー: {e}")
            return False
    
    def quick_benchmark(self, seq_len: int = 256, iterations: int = 20) -> Dict[str, float]:
        """クイックベンチマーク"""
        print(f"\n🏁 クイックベンチマーク実行中... (seq_len={seq_len})")
        
        if not self.engine:
            print("❌ モデルが読み込まれていません")
            return {}
        
        results = self.engine.benchmark_inference(seq_len, iterations)
        self.performance_history.append({
            "timestamp": time.time(),
            "results": results,
            "config": self.engine.config.copy()
        })
        
        print(f"⚡ 速度: {results['tokens_per_second']:.1f} tok/s")
        print(f"⏱️  レイテンシ: {results['avg_latency_ms']:.2f} ms")
        print(f"🖥️  デバイス: {results['device']}")
        
        return results
    
    def adjust_parameters(self):
        """パラメータ調整メニュー"""
        if not self.engine:
            print("❌ モデルが読み込まれていません")
            return
        
        print("\n🔧 パラメータ調整メニュー")
        print("1. θ減衰率 (gamma) 調整")
        print("2. θ有効/無効切り替え")
        print("3. レイヤー減衰有効/無効")
        print("4. 現在の設定表示")
        print("5. 戻る")
        
        choice = input("\n選択 (1-5): ").strip()
        
        if choice == "1":
            current_gamma = self.engine.config["theta_gamma"]
            print(f"現在のγ: {current_gamma}")
            new_gamma = input("新しいγ値 (0.90-0.99): ").strip()
            try:
                gamma_val = float(new_gamma)
                if 0.90 <= gamma_val <= 0.99:
                    self.engine.config["theta_gamma"] = gamma_val
                    print(f"✅ γを{gamma_val}に設定")
                else:
                    print("❌ 範囲外です (0.90-0.99)")
            except ValueError:
                print("❌ 無効な値です")
        
        elif choice == "2":
            current = self.engine.config["theta_enabled"]
            self.engine.config["theta_enabled"] = not current
            status = "有効" if self.engine.config["theta_enabled"] else "無効"
            print(f"✅ θテンソルを{status}に設定")
        
        elif choice == "3":
            current = self.engine.config["layer_decay"]
            self.engine.config["layer_decay"] = not current
            status = "有効" if self.engine.config["layer_decay"] else "無効"
            print(f"✅ レイヤー減衰を{status}に設定")
        
        elif choice == "4":
            print("\n📋 現在の設定:")
            for key, value in self.engine.config.items():
                print(f"   {key}: {value}")
    
    def interactive_chat(self):
        """対話型チャットテスト"""
        if not self.engine:
            print("❌ モデルが読み込まれていません")
            return
        
        print("\n💬 対話型チャットテスト")
        print("テストプロンプトを選ぶか、独自のプロンプトを入力してください")
        print("'quit'で終了、'bench'でベンチマーク実行")
        
        while True:
            print("\n" + "-"*40)
            print("📝 テストプロンプト:")
            for i, prompt in enumerate(self.test_prompts, 1):
                print(f"{i}. {prompt}")
            print(f"{len(self.test_prompts)+1}. カスタムプロンプト")
            
            choice = input("\n選択またはコマンド: ").strip().lower()
            
            if choice == "quit":
                break
            elif choice == "bench":
                self.quick_benchmark()
                continue
            
            # プロンプト選択
            prompt = None
            try:
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.test_prompts):
                    prompt = self.test_prompts[choice_num - 1]
                elif choice_num == len(self.test_prompts) + 1:
                    prompt = input("カスタムプロンプト: ").strip()
            except ValueError:
                prompt = choice  # 直接入力として扱う
            
            if prompt:
                self.run_inference_test(prompt)
    
    def run_inference_test(self, prompt: str):
        """推論テスト実行"""
        print(f"\n🧠 推論テスト: '{prompt[:50]}...'")
        
        # ダミー推論（実際のテキスト生成は簡略化）
        start_time = time.time()
        
        # 疑似的な推論処理
        seq_len = len(prompt.split()) + 100  # 入力+生成予定tokens
        test_results = self.quick_benchmark(seq_len, 5)
        
        end_time = time.time()
        
        print(f"⏱️  推論時間: {(end_time - start_time)*1000:.1f} ms")
        print(f"📊 推定生成速度: {test_results.get('tokens_per_second', 0):.1f} tok/s")
        
        # 疑似生成結果表示
        dummy_response = f"[NKAT-Generated Response for: {prompt[:30]}...] This is a simulated response demonstrating NKAT inference capabilities."
        print(f"🤖 生成結果: {dummy_response}")
    
    def performance_analysis(self):
        """パフォーマンス分析表示"""
        if not self.performance_history:
            print("❌ ベンチマーク履歴がありません")
            return
        
        print("\n📈 パフォーマンス分析")
        print("-"*50)
        
        speeds = [h["results"]["tokens_per_second"] for h in self.performance_history]
        latencies = [h["results"]["avg_latency_ms"] for h in self.performance_history]
        
        print(f"📊 統計:")
        print(f"   平均速度: {np.mean(speeds):.1f} tok/s")
        print(f"   最高速度: {np.max(speeds):.1f} tok/s")
        print(f"   最低速度: {np.min(speeds):.1f} tok/s")
        print(f"   平均レイテンシ: {np.mean(latencies):.2f} ms")
        
        # 設定別分析
        theta_enabled_speeds = [h["results"]["tokens_per_second"] 
                               for h in self.performance_history 
                               if h["config"]["theta_enabled"]]
        theta_disabled_speeds = [h["results"]["tokens_per_second"] 
                                for h in self.performance_history 
                                if not h["config"]["theta_enabled"]]
        
        if theta_enabled_speeds and theta_disabled_speeds:
            speedup = np.mean(theta_enabled_speeds) / np.mean(theta_disabled_speeds)
            print(f"🔥 NKAT効果: {(speedup-1)*100:+.1f}% 速度変化")
    
    def export_results(self):
        """結果エクスポート"""
        if not self.performance_history:
            print("❌ エクスポートする結果がありません")
            return
        
        timestamp = int(time.time())
        filename = f"nkat_demo_results_{timestamp}.json"
        
        export_data = {
            "model": self.current_model,
            "timestamp": timestamp,
            "performance_history": self.performance_history,
            "summary": {
                "total_tests": len(self.performance_history),
                "avg_speed": np.mean([h["results"]["tokens_per_second"] for h in self.performance_history]),
                "device": self.performance_history[0]["results"]["device"] if self.performance_history else "unknown"
            }
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 結果をエクスポート: {filename}")
    
    def main_menu(self):
        """メインメニュー"""
        while True:
            print("\n" + "="*40)
            print("🎯 NKAT Demo メインメニュー")
            print("="*40)
            print("1. モデル読み込み")
            print("2. クイックベンチマーク")
            print("3. パラメータ調整")
            print("4. 対話型チャットテスト")
            print("5. パフォーマンス分析")
            print("6. 結果エクスポート")
            print("7. 終了")
            
            choice = input("\n選択 (1-7): ").strip()
            
            if choice == "1":
                model_path = input("モデルパス: ").strip()
                if model_path and os.path.exists(model_path):
                    self.load_model(model_path)
                else:
                    print("❌ ファイルが見つかりません")
            
            elif choice == "2":
                self.quick_benchmark()
            
            elif choice == "3":
                self.adjust_parameters()
            
            elif choice == "4":
                self.interactive_chat()
            
            elif choice == "5":
                self.performance_analysis()
            
            elif choice == "6":
                self.export_results()
            
            elif choice == "7":
                print("👋 デモ終了")
                break
            
            else:
                print("❌ 無効な選択です")

def main():
    """メイン実行"""
    demo = NKATInteractiveDemo()
    demo.show_banner()
    
    # コマンドライン引数があれば自動読み込み
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        if os.path.exists(model_path):
            demo.load_model(model_path)
        else:
            print(f"❌ モデルファイルが見つかりません: {model_path}")
    
    demo.main_menu()

if __name__ == "__main__":
    main() 