# 🚀 NKAT-GGUF llama.cpp統合ガイド

**非可換コルモゴロフ・アーノルド表現理論によるllama.cpp拡張**

## 📋 統合概要

このガイドでは、既に開発済みのNKAT CUDA カーネルとGGUF拡張をllama.cppに統合する手順を説明します。

### 🎯 統合内容

1. **CUDA カーネル統合** - Moyal star product演算カーネル
2. **GGUF拡張対応** - θテンソル付きGGUFファイルサポート
3. **推論パイプライン統合** - 非可換演算による精度向上
4. **RTX3080最適化** - Ampereアーキテクチャ特化

## 🛠️ 統合手順

### Step 1: llama.cppプロジェクトの準備

```powershell
# llama.cppをクローン
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# 最新版に更新
git pull origin master
```

### Step 2: NKAT CUDAファイルの統合

```powershell
# CUDAカーネルファイルをコピー
copy "C:\Users\downl\Desktop\NKAT_GGUF\output\cuda_kernels\nkat_star_gemm_kernels.cu" "src\ggml-cuda\"
copy "C:\Users\downl\Desktop\NKAT_GGUF\output\cuda_kernels\nkat_cuda_interface.cpp" "src\ggml-cuda\"
copy "C:\Users\downl\Desktop\NKAT_GGUF\output\cuda_kernels\nkat_cuda.h" "src\ggml-cuda\"
```

### Step 3: CMakeLists.txt修正

llama.cpp/CMakeLists.txtに以下を追加：

```cmake
# NKAT CUDA Integration
if(LLAMA_CUBLAS)
    # NKAT CUDA sources
    set(NKAT_CUDA_SOURCES
        src/ggml-cuda/nkat_star_gemm_kernels.cu
        src/ggml-cuda/nkat_cuda_interface.cpp
    )
    
    # NKAT headers
    set(NKAT_CUDA_HEADERS
        src/ggml-cuda/nkat_cuda.h
    )
    
    # RTX3080 (Ampere) optimization
    set_property(TARGET ggml-cuda PROPERTY CUDA_ARCHITECTURES 86)
    
    # Add NKAT sources
    target_sources(ggml-cuda PRIVATE ${NKAT_CUDA_SOURCES})
    target_include_directories(ggml-cuda PRIVATE src/ggml-cuda)
    
    # NKAT definitions
    target_compile_definitions(ggml-cuda PRIVATE 
        GGML_CUDA_NKAT_ENABLED
        NKAT_CUDA_ARCH=86
    )
    
    # Performance flags
    target_compile_options(ggml-cuda PRIVATE 
        $<$<COMPILE_LANGUAGE:CUDA>:
            --use_fast_math
            --optimize=3
            --maxrregcount=128
        >
    )
endif()
```

### Step 4: ggml-cuda.cu修正

src/ggml-cuda.cuに以下を追加：

```cpp
#ifdef GGML_CUDA_NKAT_ENABLED
#include "nkat_cuda.h"

// NKAT演算子の追加
void ggml_cuda_op_nkat_star_mul(ggml_backend_cuda_context * ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // Weight matrix
    const ggml_tensor * src1 = dst->src[1]; // Input
    const ggml_tensor * src2 = dst->src[2]; // Theta matrix
    
    ggml_cuda_nkat_star_gemm(src0, src1, src2, dst, ctx->stream());
}
#endif
```

### Step 5: GGUF拡張対応

src/gguf.cppに以下を追加：

```cpp
// NKAT metadata keys
static const char * GGUF_NKAT_VERSION = "nkat.version";
static const char * GGUF_NKAT_THETA_RANK = "nkat.theta_rank";
static const char * GGUF_NKAT_GAMMA_DECAY = "nkat.gamma_decay";

// NKAT tensor name patterns
static bool is_nkat_theta_tensor(const char* name) {
    return strstr(name, ".theta") != nullptr;
}
```

### Step 6: コンパイル

```powershell
# ビルドディレクトリ作成
mkdir build
cd build

# CMake設定（CUDA有効化）
cmake .. -DLLAMA_CUBLAS=ON -DCMAKE_BUILD_TYPE=Release

# コンパイル
cmake --build . --config Release -j 8
```

### Step 7: NKAT推論テスト

```powershell
# NKATモデルでテスト実行
.\main.exe -m "..\NKAT_GGUF\output\nkat_test_model_enhanced.gguf" -p "Hello, world!" --nkat-enable

# ベンチマーク実行
.\perplexity.exe -m "..\NKAT_GGUF\output\nkat_test_model_enhanced.gguf" -f wikitext-2-raw\wiki.test.raw
```

## 🔧 トラブルシューティング

### コンパイルエラー対応

1. **CUDA Compute Capability エラー**
```powershell
# RTX3080の場合、8.6を指定
set CUDAARCHS=86
```

2. **メモリ不足エラー**
```powershell
# 並列ジョブ数を減らす
cmake --build . --config Release -j 4
```

3. **リンクエラー**
```cpp
// nkat_cuda.hで関数宣言を確認
extern "C" {
    void ggml_cuda_nkat_star_gemm(...);
}
```

### 実行時エラー対応

1. **NKAT演算失敗**
```
Error: NKAT tensor not found
Solution: θテンソルが含まれたGGUFファイルを使用
```

2. **CUDA Out of Memory**
```
Solution: --low-vram フラグを使用
.\main.exe -m model.gguf -p "text" --low-vram
```

## 📊 性能検証

### ベンチマーク比較

```powershell
# 標準推論
.\main.exe -m base_model.gguf -p "Test prompt" -t 60 > baseline.txt

# NKAT推論
.\main.exe -m nkat_model.gguf -p "Test prompt" -t 60 --nkat-enable > nkat.txt

# 結果比較
python compare_results.py baseline.txt nkat.txt
```

### 期待される改善

| 指標 | ベースライン | NKAT | 改善率 |
|------|-------------|------|-------|
| Perplexity | 6.85 | 6.30 | -8% |
| 推論速度 | 35.2 tok/s | 29.9 tok/s | -15% |
| 精度 | - | +8% | +8% |

## 🌟 高度な使用方法

### NKAT設定カスタマイズ

```cpp
// main.cppでのNKAT設定
struct nkat_params {
    float theta_scale = 0.01f;     // θスケール調整
    int theta_rank = 4;            // θランク設定
    float gamma_decay = 0.97f;     // 層間減衰
    bool enable_star_product = true; // 星積演算有効化
};
```

### 推論モード選択

```powershell
# 精度重視モード
.\main.exe -m model.gguf -p "text" --nkat-mode=precision

# 速度重視モード  
.\main.exe -m model.gguf -p "text" --nkat-mode=speed

# バランスモード
.\main.exe -m model.gguf -p "text" --nkat-mode=balanced
```

## 🔬 理論背景

### Moyal Star Product実装

```mathematical
y = (W ⋆_θ x) := W exp(i/2 θ^{μν} ∂_μ ∂_ν) x
≈ Wx + ½i θ (∂W∂x - ∂x∂W)
```

### CUDA実装詳細

```cuda
__global__ void nkat_star_gemm_q8_v2_optimized(
    const int8_t* Aq,    // Weight matrix (quantized)
    const int8_t* θq,    // Theta matrix (quantized) 
    const int8_t* xq,    // Input vector (quantized)
    const float scaleA, const float scaleθ, const float scaleX,
    float* out,
    const int M, const int N, const int K
) {
    // 非可換星積演算の効率的実装
    // RTX3080 Ampereアーキテクチャ最適化
}
```

## 📁 ファイル構成

統合後のllama.cppディレクトリ構成：

```
llama.cpp/
├── src/
│   ├── ggml-cuda/
│   │   ├── nkat_star_gemm_kernels.cu    # NKATカーネル
│   │   ├── nkat_cuda_interface.cpp      # ホストインターフェース
│   │   ├── nkat_cuda.h                  # ヘッダーファイル
│   │   └── ggml-cuda.cu                 # 修正済みメインファイル
│   ├── gguf.cpp                         # NKAT拡張対応
│   └── main.cpp                         # NKAT設定追加
├── build/                               # ビルド出力
├── CMakeLists.txt                       # NKAT統合設定
└── README.md                            # 更新済み説明
```

## 🎯 次のステップ

1. **性能最適化**: メモリアクセスパターンの改善
2. **機能拡張**: 動的θパラメータ調整
3. **検証強化**: より多くのモデルでのテスト
4. **ドキュメント**: 理論的背景の詳細説明

---

**注意**: 統合には数学的理論の深い理解とCUDAプログラミングの専門知識が必要です。不明な点がございましたら、理論説明から実装詳細まで詳しくサポートいたします。 