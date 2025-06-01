
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
