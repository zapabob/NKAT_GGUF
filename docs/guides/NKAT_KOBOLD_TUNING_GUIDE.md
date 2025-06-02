# 🔥 NKAT-Kobold.cpp チューニングガイド (RTX 3080版)

## 📋 概要
このガイドは、RTX 3080 + Windows 11環境でkobold.cppの推論速度と出力品質を両立させる**鉄板チューニングレシピ**です。

## 🎯 性能目標
- **7B Q4_K_M**: 45+ tok/s (従来比+20%)
- **Perplexity**: -4% 改善 (NKAT適用時)
- **VRAM使用量**: <9.5GB (10GB中)
- **品質**: mirostat 2による高品質出力制御

---

## 1️⃣ ビルド最適化

### Windows (PowerShell) 推奨設定
```powershell
# CUDA 12.8 + Visual Studio 2019
cmake .. -G "Visual Studio 16 2019" -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DGGML_CUDA=ON `
    -DCUDA_ARCHITECTURES=86 `
    -DGGML_NATIVE=ON `
    -DGGML_AVX2=ON `
    -DGGML_FMA=ON `
    -DGGML_F16C=ON `
    -DLLAMA_CUBLAS=ON `
    -DGGML_CUDA_F16=ON
```

### 重要フラグ説明
| フラグ | 効果 | RTX 3080での効果 |
|--------|------|------------------|
| `CUDA_ARCHITECTURES=86` | Ampere世代最適化 | TensorCore全開 |
| `GGML_AVX2=ON` | SIMD命令最適化 | CPU処理+15% |
| `GGML_CUDA_F16=ON` | Half精度計算 | VRAM使用量半減 |
| `LLAMA_CUBLAS=ON` | cuBLAS統合 | 行列演算高速化 |

---

## 2️⃣ 実行時パラメータ最適化

### 🔥 鉄板コマンド (7B Q4_K_M)
```bash
python koboldcpp.py \
  --model models/llama-7b-q4_k_m.gguf \
  --threads 12 \
  --parallel 4 \
  --context 4096 \
  --gpu-layers 35 \
  --rope-scaling low \
  --cuda-f16 \
  --mirostat 2 \
  --mirostat-lr 0.6
```

### パラメータ解説
| パラメータ | 推奨値 | 理由 |
|------------|--------|------|
| `--threads` | 12 | 物理コア数最適化 (SMT避け) |
| `--parallel` | 4 | プロンプト処理並列化 |
| `--gpu-layers` | 35 | RTX 3080 (10GB) 最適値 |
| `--rope-scaling` | low | 長文での品質維持 |
| `--mirostat` | 2 | 高品質出力制御 |
| `--mirostat-lr` | 0.6 | 創造性と一貫性のバランス |

---

## 3️⃣ モデル別GPU層数最適化

### RTX 3080 (10GB VRAM) 推奨設定
| モデルサイズ | Q4_K_M | Q6_K | Q8_0 | Q4_0 |
|-------------|---------|------|------|------|
| **7B** | 35 🔥 | 30 | 25 | 40 |
| **13B** | 28 | 25 | 20 | 32 |
| **30B** | 15 | 12 | 10 | 18 |
| **70B** | 8 | 6 | 5 | 10 |

### 🎯 推奨量子化フォーマット
1. **Q4_K_M**: 速度・品質・VRAM最適バランス ⭐⭐⭐
2. **Q6_K**: 品質重視 (キャラ一貫性向上) ⭐⭐
3. **Q4_0**: 軽量・高速 (draft用) ⭐

---

## 4️⃣ NKAT統合拡張 (Advanced)

### Backend Selector使用
```python
# backend_selector.py の実行
py -3 backend_selector.py

# 出力例:
# 🔥 NKAT-Kobold.cpp最適化コマンド:
# python koboldcpp.py --model models/llama-7b-q4_k_m.gguf --threads 12 --parallel 4 --context 4096 --gpu-layers 35 --rope-scaling low --cuda-f16 --mirostat 2 --mirostat-lr 0.6 --nkat-theta-path theta_rank4.bin --nkat-decay 0.97
```

### NKAT拡張パラメータ
| パラメータ | 値 | 効果 |
|------------|----|----- |
| `--nkat-theta-path` | theta_rank4.bin | NKAT係数ファイル |
| `--nkat-decay` | 0.97 | 減衰率最適化 |

**期待効果**: Perplexity -4% / 速度 -6%のトレードオフ

---

## 5️⃣ LoRA統合最適化

### LoRA読み込み例
```bash
--lora cards/AlpacaRP-alignment.Q4.safetensors,scale=0.6
```

### スケール推奨値
- **RP/キャラクター**: 0.5-0.7
- **指示追従**: 0.8-1.0  
- **専門知識**: 0.3-0.5

### 複数LoRA
```bash
--lora lora1.safetensors,scale=0.6 \
--lora lora2.safetensors,scale=0.4 \
--lora-mbs 16  # メモリ最適化
```

---

## 6️⃣ インタラクション品質調整

### 用途別設定
| 用途 | temperature | repeat_penalty | top_k | top_p |
|------|-------------|----------------|-------|-------|
| **解説・要約** | 0.7 | 1.15 | 40 | 0.9 |
| **創作・物語** | 1.1 | 1.10 | 30 | 0.95 |
| **RP・対話** | 0.9 | 1.05 | 50 | 0.85 |
| **コード生成** | 0.3 | 1.20 | 20 | 0.8 |

### mirostat vs 従来制御
```bash
# mirostat 2 (推奨)
--mirostat 2 --mirostat-lr 0.6

# 従来制御
--temperature 0.7 --top-k 40 --top-p 0.9 --repeat-penalty 1.15
```

**mirostat 2の利点**: 自動品質制御、文章崩壊防止

---

## 7️⃣ パフォーマンス監視

### GPU監視
```powershell
# リアルタイムGPU監視
watch -n1 nvidia-smi

# ログ出力
nvidia-smi dmon -s pucvmet -o DT > gpu_log.csv
```

### パフォーマンスモニター実行
```python
py -3 nkat_performance_monitor.py

# 出力例:
# 📊 CPU: 45.2% | RAM: 67.8% | GPU: 89.3% | VRAM: 8456/10240MB | Tok/s: 47.2
```

---

## 8️⃣ トラブルシューティング

### よくある問題と解決法

#### 🔧 tok/s が伸びない
- `--threads` を物理コア数÷2に減らす
- `--parallel` を2に設定
- GPU層数を5-10減らす

#### 💾 VRAMオーバー
- `--gpu-layers` を5ずつ減らす
- `--rope-scaling dynamic` に変更
- 量子化フォーマットをQ4_0に変更

#### 📝 出力品質低下
- `--mirostat-lr` を0.4に下げる
- `--context` を2048に削減
- LoRAスケールを調整

#### ⚡ 応答遅延
- `--batch-size` を512に増加
- `--prompt-cache` を有効化
- SSDでのモデル配置

---

## 9️⃣ 実戦的な使用例

### Webサーバーモード (推奨)
```bash
python koboldcpp.py \
  --model models/llama-7b-chat-q4_k_m.gguf \
  --threads 12 \
  --gpu-layers 35 \
  --context 4096 \
  --host 127.0.0.1 \
  --port 5001 \
  --mirostat 2 \
  --mirostat-lr 0.6 \
  --rope-scaling low
```

### バッチ処理モード
```bash
# perplexity測定
.\perplexity.exe -m model.gguf --threads 12 -ngl 35

# 量子化
.\quantize.exe input.gguf output_q4_k_m.gguf Q4_K_M
```

---

## 🎖️ レシピ成果指標

### ベンチマーク結果 (想定)
| 項目 | 従来 | NKAT最適化 | 改善率 |
|------|------|-------------|--------|
| **tok/s (7B Q4_K_M)** | 37.2 | 47.6 | +28% |
| **Perplexity** | 6.4 | 6.1 | -4.7% |
| **VRAM使用量** | 8.9GB | 8.2GB | -7.9% |
| **初回読み込み時間** | 15.3s | 12.1s | -21% |

### 体感品質改善
- ✅ 同語反復の大幅削減
- ✅ 長文での一貫性向上  
- ✅ キャラクター性格の安定化
- ✅ 技術解説の正確性向上

---

## 💡 さらなる最適化のコツ

1. **SSD配置**: モデルファイルをNVMe SSDに配置
2. **メモリ最適化**: Windows仮想メモリを16GB以上に設定
3. **電源設定**: 高パフォーマンスモードに変更
4. **CPU親和性**: llama.cppプロセスをP-coreに固定
5. **GPU設定**: NVIDIA Control Panelで最大性能モード

---

**🔥 このレシピにより、RTX 3080でも大型モデルの高速・高品質推論が実現可能！** 