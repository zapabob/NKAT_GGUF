# 🎨 NKAT-GGUF 高機能GUI使用ガイド

## 📚 目次
1. [概要](#概要)
2. [機能一覧](#機能一覧)
3. [セットアップ](#セットアップ)
4. [使用方法](#使用方法)
5. [詳細設定](#詳細設定)
6. [トラブルシューティング](#トラブルシューティング)
7. [高度な使用例](#高度な使用例)

## 📋 概要

NKAT-GGUF高機能GUI変換システムは、非可換コルモゴロフアーノルド表現理論（NKAT）を用いた高度なGGUF最適化システムです。

### 🎯 主な特徴
- **直感的なGUI**: ドラッグ&ドロップ対応の使いやすいインターフェース
- **Hugging Face連携**: URL入力による直接ダウンロード
- **自動バックアップ**: 変換前の安全なファイル保護
- **履歴管理**: 過去の変換記録を完全追跡
- **RTX3080最適化**: CUDA加速による高速処理
- **電源断リカバリー**: 処理中断からの自動復旧

## 🚀 機能一覧

### 📁 ファイル選択機能
| 機能 | 説明 | 対応形式 |
|------|------|----------|
| ドラッグ&ドロップ | ファイルを直接エリアにドロップ | .gguf |
| Hugging Face URL | HFモデルのURL入力でダウンロード | URL, repo_id |
| ファイルアップロード | ローカルファイルの選択 | .gguf |
| 履歴選択 | 過去に使用したファイルから選択 | 履歴データ |

### ⚙️ 設定管理機能
| カテゴリ | 設定項目 | 範囲/選択肢 |
|----------|----------|-------------|
| NKAT理論 | Kolmogorov-Arnold演算子 | 有効/無効 |
| NKAT理論 | グリッドサイズ | 4-16 |
| NKAT理論 | リー代数次元 | 2-8 |
| NKAT理論 | 非可換強度 | 0.01-1.0 |
| パフォーマンス | 64bit精度 | 有効/無効 |
| パフォーマンス | CUDA最適化 | 有効/無効 |
| パフォーマンス | 最大メモリ | 1-15GB |
| パフォーマンス | チャンクサイズ | 128-2048MB |

### 💾 バックアップ機能
- **自動バックアップ**: 変換前の自動ファイル保護
- **タイムスタンプ付き**: 日時情報を含む識別可能な名前
- **メタデータ保存**: バックアップの詳細情報
- **クリーンアップ**: 古いバックアップの自動削除

### 📚 履歴管理機能
- **変換履歴**: 過去の変換処理記録
- **ファイル情報**: サイズ、パス、ソース情報
- **検索機能**: 履歴からの効率的な検索
- **JSON形式**: 構造化されたデータ保存

## 🛠️ セットアップ

### 前提条件
- Google Colab環境（推奨）
- Python 3.7以上
- 最低4GBのRAM（RTX3080使用時は8GB推奨）

### インストール手順

#### 1. リポジトリクローン
```bash
git clone https://github.com/your-username/NKAT_GGUF.git
cd NKAT_GGUF
```

#### 2. 依存関係インストール
```bash
pip install -r requirements.txt
```

#### 3. Jupyter拡張機能有効化
```bash
jupyter nbextension enable --py widgetsnbextension
```

#### 4. CUDA環境セットアップ（オプション）
```python
# RTX3080使用時
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📝 使用方法

### クイックスタート

#### Google Colabでの起動
```python
# 1. ノートブックを開く
# NKAT_GGUF_Colab_QuickStart.ipynbを実行

# 2. または直接実行
exec(open("scripts/run_advanced_gui.py").read())
```

### 詳細な使用手順

#### ステップ1: ファイル選択

##### 方法A: ドラッグ&ドロップ
1. 画面上部の「📋 ドラッグ&ドロップエリア」を確認
2. GGUFファイルをエリアにドラッグ
3. 複数ファイル同時選択可能
4. ファイル情報が自動表示

##### 方法B: Hugging Face URL
1. 「📁 ファイル選択」タブを選択
2. 「🤗 Hugging Face ダウンロード」セクションに移動
3. URL入力（対応形式）:
   ```
   https://huggingface.co/microsoft/DialoGPT-large
   microsoft/DialoGPT-large
   
   # 複数URL（改行区切り）
   microsoft/DialoGPT-large
   microsoft/DialoGPT-medium
   rinna/japanese-gpt2-medium
   ```
4. 「📥 一括ダウンロード」クリック

##### 方法C: ローカルファイル
1. 「📁 ファイルアップロード」セクション
2. 「複数ファイル選択」ボタンクリック
3. ローカルのGGUFファイル選択

##### 方法D: 履歴選択
1. 「🕒 最近使用したファイル」ドロップダウン
2. 過去に使用したファイルから選択

#### ステップ2: 設定調整（オプション）

##### 基本設定
1. 「⚙️ 詳細設定」タブを選択
2. プリセット選択:
   - **デフォルト**: 標準的な設定
   - **高速処理**: 処理速度優先
   - **高品質**: 変換品質優先
   - **省メモリ**: メモリ使用量最小化
   - **RTX3080最適化**: GPU最適化設定

##### カスタム設定
- **NKAT理論設定**:
  ```
  Kolmogorov-Arnold演算子: 有効
  グリッドサイズ: 8-12（標準）
  リー代数次元: 4-6（標準）
  非可換強度: 0.3-0.7（標準）
  ```

- **パフォーマンス設定**:
  ```
  64bit精度: データ精度重視時は有効
  CUDA最適化: GPU使用時は有効
  最大メモリ: 利用可能メモリの70-80%
  チャンクサイズ: 512-1024MB（標準）
  ```

#### ステップ3: 変換実行
1. ファイル選択完了を確認
2. 「🚀 一括NKAT変換実行」ボタンクリック
3. プログレス表示で進捗確認
4. 変換ログで詳細確認
5. 完了後、最適化ファイルを取得

## ⚙️ 詳細設定

### NKAT理論パラメータ

#### Kolmogorov-Arnold演算子
```python
# 効果: 多変数関数の一変数関数への最適分解
# 推奨設定:
enable_ka_operators = True  # 高品質変換時
enable_ka_operators = False # 高速処理時
```

#### グリッドサイズ
```python
# 効果: 関数分解の解像度
# 範囲: 4-16
# 推奨:
ka_grid_size = 8   # バランス型
ka_grid_size = 12  # 高品質
ka_grid_size = 6   # 高速処理
```

#### リー代数次元
```python
# 効果: 変換群の自由度
# 範囲: 2-8
# 推奨:
lie_algebra_dim = 4  # 標準
lie_algebra_dim = 6  # 高精度
lie_algebra_dim = 3  # 軽量
```

#### 非可換強度
```python
# 効果: 非可換性の強さ
# 範囲: 0.01-1.0
# 推奨:
noncommutative_strength = 0.5   # バランス
noncommutative_strength = 0.8   # 高圧縮
noncommutative_strength = 0.2   # 安全性重視
```

### パフォーマンス最適化

#### メモリ管理
```python
# システムメモリに応じた設定
# 8GB環境
max_memory_gb = 6.0
chunk_size_mb = 512

# 16GB環境
max_memory_gb = 12.0
chunk_size_mb = 1024

# 32GB環境
max_memory_gb = 24.0
chunk_size_mb = 2048
```

#### CUDA最適化
```python
# RTX3080設定例
{
    "enable_cuda_optimization": True,
    "use_64bit_precision": True,
    "max_memory_gb": 8.0,
    "chunk_size_mb": 1024,
    "cuda_device": 0
}
```

### バックアップ設定

#### 自動バックアップ
```python
backup_config = {
    "auto_backup": True,           # 変換前自動バックアップ
    "backup_to_drive": True,       # Google Driveバックアップ
    "max_backups": 10,             # 最大保持数
    "cleanup_old": True            # 古いファイル自動削除
}
```

## 🔧 トラブルシューティング

### 一般的な問題

#### 🚨 GUIが表示されない
**症状**: インターフェースが空白または表示されない

**解決方法**:
```python
# 1. Jupyter拡張機能再有効化
!jupyter nbextension enable --py widgetsnbextension

# 2. ページ再読み込み
# ブラウザでページを更新

# 3. 環境確認
import ipywidgets as widgets
print(widgets.__version__)
```

#### 🚨 ドラッグ&ドロップが機能しない
**症状**: ファイルをドロップしても反応しない

**解決方法**:
```python
# 1. ブラウザサポート確認
# Chrome、Firefox、Edge推奨

# 2. ファイルサイズ確認
# 大きなファイル（>2GB）は別の方法を使用

# 3. JavaScript確認
# ブラウザの開発者ツールでエラー確認
```

#### 🚨 CUDA加速が利用できない
**症状**: GPU処理が開始されない

**解決方法**:
```python
# 1. CUDA環境確認
import torch
print(f"CUDA利用可能: {torch.cuda.is_available()}")
print(f"CUDAバージョン: {torch.version.cuda}")
print(f"GPUデバイス数: {torch.cuda.device_count()}")

# 2. GPU情報確認
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  メモリ: {props.total_memory // (1024**3)}GB")

# 3. CUDA再インストール
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 🚨 メモリ不足エラー
**症状**: OutOfMemoryError または system-killed

**解決方法**:
```python
# 1. メモリ使用量確認
import psutil
memory = psutil.virtual_memory()
print(f"総メモリ: {memory.total // (1024**3)}GB")
print(f"利用可能: {memory.available // (1024**3)}GB")
print(f"使用率: {memory.percent}%")

# 2. 設定調整
config_adjustments = {
    "max_memory_gb": 4.0,          # メモリ制限を下げる
    "chunk_size_mb": 256,          # チャンクサイズを小さく
    "use_64bit_precision": False,  # 32bit精度に変更
    "enable_checkpoint": True      # チェックポイント有効化
}

# 3. Google Colab Pro検討
# より多くのメモリとGPUが利用可能
```

#### 🚨 Hugging Faceダウンロードエラー
**症状**: HFからのダウンロードが失敗

**解決方法**:
```python
# 1. 認証確認
from huggingface_hub import login
login()  # HFトークンでログイン

# 2. リポジトリ存在確認
from huggingface_hub import hf_hub_url
try:
    url = hf_hub_url(repo_id="microsoft/DialoGPT-large", filename="pytorch_model.bin")
    print(f"リポジトリ確認: {url}")
except Exception as e:
    print(f"エラー: {e}")

# 3. 手動ダウンロード
!git lfs install
!git clone https://huggingface.co/microsoft/DialoGPT-large
```

### エラーログの確認

#### ログファイル場所
```
/content/nkat_conversion.log    # 変換ログ
/content/nkat_error.log         # エラーログ
/content/nkat_file_history.json # ファイル履歴
/content/nkat_backups/          # バックアップディレクトリ
```

#### ログ内容確認
```python
# 最新エラーログ確認
with open("/content/nkat_error.log", "r") as f:
    lines = f.readlines()
    for line in lines[-20:]:  # 最新20行
        print(line.strip())

# 変換ログ確認
with open("/content/nkat_conversion.log", "r") as f:
    content = f.read()
    print(content[-2000:])  # 最新2000文字
```

## 🎯 高度な使用例

### 例1: 大規模モデルの段階的変換

```python
# 設定: メモリ効率重視
config = {
    "max_memory_gb": 6.0,
    "chunk_size_mb": 256,
    "enable_checkpoint": True,
    "ka_grid_size": 6,          # 軽量設定
    "lie_algebra_dim": 3,
    "noncommutative_strength": 0.3
}

# 対象: 7B+パラメータモデル
large_models = [
    "microsoft/DialoGPT-large",
    "facebook/blenderbot-3B",
    "google/flan-t5-large"
]
```

### 例2: 高品質変換（RTX3080環境）

```python
# 設定: 品質最優先
config = {
    "enable_cuda_optimization": True,
    "use_64bit_precision": True,
    "max_memory_gb": 10.0,
    "chunk_size_mb": 1024,
    "ka_grid_size": 16,         # 最高解像度
    "lie_algebra_dim": 8,       # 最大次元
    "noncommutative_strength": 0.8,
    "enable_ka_operators": True
}
```

### 例3: 一括処理（複数ファイル）

```python
# HF URL一括入力例
batch_urls = """
microsoft/DialoGPT-small
microsoft/DialoGPT-medium
microsoft/DialoGPT-large
rinna/japanese-gpt2-small
rinna/japanese-gpt2-medium
"""

# 処理フロー:
# 1. URL入力エリアに上記をペースト
# 2. 「📥 一括ダウンロード」実行
# 3. 自動的に履歴に追加
# 4. 「🚀 一括NKAT変換実行」で全て処理
```

### 例4: カスタム最適化設定

```python
# モデルサイズ別最適化
optimization_profiles = {
    "small_models": {  # <1B パラメータ
        "ka_grid_size": 12,
        "lie_algebra_dim": 6,
        "noncommutative_strength": 0.6,
        "chunk_size_mb": 512
    },
    "medium_models": {  # 1B-3B パラメータ
        "ka_grid_size": 10,
        "lie_algebra_dim": 5,
        "noncommutative_strength": 0.5,
        "chunk_size_mb": 768
    },
    "large_models": {  # 3B+ パラメータ
        "ka_grid_size": 8,
        "lie_algebra_dim": 4,
        "noncommutative_strength": 0.4,
        "chunk_size_mb": 1024
    }
}
```

### 例5: 自動化スクリプト

```python
# 定期的な変換処理自動化
import schedule
import time

def auto_convert_new_models():
    """新しいモデルの自動変換"""
    # 1. 指定されたHFリポジトリをチェック
    # 2. 新しいモデルを検出
    # 3. 自動ダウンロード
    # 4. NKAT変換実行
    # 5. 結果をGoogle Driveに保存
    pass

# 毎日午前2時に実行
schedule.every().day.at("02:00").do(auto_convert_new_models)

while True:
    schedule.run_pending()
    time.sleep(3600)  # 1時間間隔でチェック
```

## 📞 サポート・コミュニティ

### 公式ドキュメント
- 技術仕様: `docs/technical_specification.md`
- API リファレンス: `docs/api_reference.md`
- 理論的背景: `docs/mathematical_foundation.md`

### コミュニティ
- GitHub Issues: バグ報告・機能要望
- Discord サーバー: リアルタイム質問・議論
- Reddit コミュニティ: 使用事例共有

### 開発者向け
- 貢献ガイド: `CONTRIBUTING.md`
- 開発環境セットアップ: `docs/development_setup.md`
- テストスイート実行: `pytest tests/`

---

**🎉 NKAT-GGUF高機能GUI変換システムで、数学的最適化による高品質なGGUF変換をお楽しみください！** 