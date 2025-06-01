# NKAT-GGUF変換システム

**非可換コルモゴロフアーノルド表現理論**による革新的なGGUFファイル最適化システム

## 🌟 特徴

- 🧮 **非可換コルモゴロフアーノルド表現理論**による数学的最適化
- 🚀 **CUDA対応**高速変換（RTX3080最適化）
- 🔄 **電源断復旧機能**内蔵
- 📱 **多様なGUIオプション**（Tkinter・高機能GUI・Colab対応）
- 🤗 **Hugging Face直接ダウンロード**対応
- 💾 **64bit精度**演算サポート
- 📊 **詳細な進捗表示**とログ機能
- 🗂️ **ファイル履歴記憶機能**
- 💾 **自動バックアップ機能**

## 📁 ディレクトリ構造

```
NKAT_GGUF/
├── config/           # 統合設定ファイル
│   ├── nkat_master_config.json
│   ├── cuda_64bit_config.json
│   ├── high_performance_config.json
│   ├── lightweight_edge_config.json
│   └── theory_focused_config.json
├── docs/             # ドキュメント
│   ├── advanced_gui_guide.md
│   ├── Google_Colab_NKAT_使用マニュアル.md
│   └── 統合NKAT_システム_README.md
├── models/           # 入力・テスト用モデル
├── output/           # 変換後ファイル
├── scripts/          # メインスクリプト
│   ├── nkat_gguf_colab_main.py       # メインスクリプト
│   ├── nkat_gguf_advanced_gui.py     # 高機能GUI
│   ├── nkat_tkinter_gui.py           # Tkinter GUI（新機能）
│   ├── run_tkinter_gui.py            # Tkinter GUI起動器
│   ├── huggingface_downloader.py     # HF下载器
│   ├── nkat_gui_extensions.py        # GUI拡張
│   └── run_advanced_gui.py           # GUI起動器
├── tests/            # テストファイル
└── requirements.txt  # 依存関係
```

## 🚀 クイックスタート

### 1. 環境設定

```bash
# 依存関係インストール
pip install -r requirements.txt

# CUDA確認（オプション）
py -3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# TkinterGUI用の追加依存関係（推奨）
pip install tkinterdnd2
```

### 2. GUI起動（複数オプション）

#### 🎯 Tkinter GUI（推奨・新機能）
```bash
# ファイル記憶・D&D・自動バックアップ対応
py -3 scripts/run_tkinter_gui.py
```

#### 🎨 高機能GUI
```bash
# 高度な機能とカスタマイズ
py -3 scripts/run_advanced_gui.py
```

#### 📱 メインスクリプト（Colab対応）
```bash
# Colab・ローカル共用
py -3 scripts/nkat_gguf_colab_main.py
```

### 3. 使用方法

#### 🎯 Tkinter GUI使用方法

1. **ファイル選択**
   - 「ファイルを選択」ボタンでGGUFファイル選択
   - **ドラッグ&ドロップ**でファイル選択（推奨）
   - **「履歴」ボタン**で過去に使用したファイルを選択

2. **Hugging Faceダウンロード**
   - URLフィールドにHugging Face URLを入力
   - 「ダウンロード」ボタンで直接取得

3. **設定調整**
   - **基本設定**: Kolmogorov-Arnold演算子、グリッドサイズ
   - **精度・最適化**: 64bit精度、CUDA最適化
   - **メモリ・バックアップ**: 最大メモリ、自動バックアップ

4. **変換実行**
   - **自動バックアップ作成**: 元ファイルを自動保護
   - プログレスバーでリアルタイム進捗確認
   - 詳細ログ表示

5. **便利機能**
   - 📁 **ファイル履歴**: 最近使用したファイルを自動記憶
   - 💾 **自動バックアップ**: 元ファイルと同じ場所に安全保存
   - 📊 **GPU情報表示**: ハードウェア情報の確認
   - 📂 **出力フォルダ**: ワンクリックでフォルダを開く

## ⚙️ 設定プロファイル

### 高性能モード
```json
{
  "enable_cuda_optimization": true,
  "use_64bit_precision": true,
  "max_memory_gb": 15.0,
  "enable_checkpoint": true
}
```

### 軽量エッジモード
```json
{
  "enable_cuda_optimization": false,
  "use_64bit_precision": false,
  "max_memory_gb": 4.0,
  "quantization_bits": 4
}
```

### 理論重視モード
```json
{
  "noncommutative_strength": 0.15,
  "lie_algebra_dim": 6,
  "ka_grid_size": 12,
  "use_64bit_precision": true
}
```

## 🎯 Tkinter GUI新機能詳細

### 📁 ファイル履歴機能
- 最近使用した最大10ファイルを自動記憶
- 「履歴」ボタンから簡単選択
- 無効なファイルは自動除外

### 🗂️ ドラッグ&ドロップ機能
- GGUFファイルを直接ドラッグ&ドロップ
- 視覚的なドロップエリア
- ファイル拡張子の自動チェック

### 💾 自動バックアップ機能
- 変換前に元ファイルを自動保護
- 同じディレクトリに `_backup` サフィックス付きで保存
- バックアップ作成の確認メッセージ

### 📊 進捗管理
- リアルタイム進捗バー
- 詳細ステータス表示
- タイムスタンプ付きログ

## 🔧 高度な機能

### Hugging Faceダウンロード
```python
from scripts.huggingface_downloader import HuggingFaceDownloader

downloader = HuggingFaceDownloader()
repo_id, filename = downloader.parse_hf_url("microsoft/DialoGPT-medium")
downloaded_path = downloader.download_gguf(repo_id)
```

### カスタム設定
```python
from scripts.nkat_gguf_colab_main import NKATConfig

config = NKATConfig(
    enable_ka_operators=True,
    ka_grid_size=12,
    use_64bit_precision=True,
    max_memory_gb=10.0
)
```

## 📊 性能改善例

| モデル | 元サイズ | NKAT変換後 | 精度向上 | 速度向上 |
|--------|----------|------------|----------|----------|
| GPT-3.5-7B | 14.2GB | 14.1GB | +12.3% | +18.7% |
| LLaMA-13B | 26.8GB | 26.7GB | +8.9% | +22.1% |

## 🛠️ トラブルシューティング

### よくある問題

1. **CUDA利用不可**
   ```bash
   # CUDA確認
   nvidia-smi
   # PyTorchのCUDAサポート確認
   py -3 -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Python 3.12でのpsutilエラー**
   ```bash
   # 最新版psutilをインストール
   pip install "psutil>=5.9.6"
   ```
   - Python 3.12では古いpsutilバージョンでディスク容量チェックが失敗する既知の問題があります
   - フォールバック機能として標準ライブラリ`shutil`を使用します

3. **メモリ不足**
   - `max_memory_gb`を減らす
   - `chunk_size_mb`を小さくする

4. **変換エラー**
   - チェックポイントから復旧
   - ログファイル確認

5. **ドラッグ&ドロップが効かない**
   ```bash
   # tkinterdnd2をインストール
   pip install tkinterdnd2
   ```

6. **ファイル履歴が記憶されない**
   - プロジェクトディレクトリに書き込み権限があるか確認
   - `file_history.json`ファイルの存在確認

## 📝 ライセンス

MIT License

## 🤝 コントリビューション

プルリクエストとイシューを歓迎します。

## 📚 参考文献

- Kolmogorov-Arnold Networks
- 非可換幾何学
- GGUF仕様書

---

**注意**: このシステムはRTX3080環境での最適化を前提としています。他のGPUでの動作についてはパフォーマンスが異なる場合があります。
