@echo off
REM KoboldCPP最適化起動スクリプト
REM モデル: Vecteus-v1-IQ4_XS.gguf
REM 生成日時: C:\Users\downl\Desktop\NKAT_GGUF

echo 🚀 KoboldCPP最適化起動
echo モデル: Vecteus-v1-IQ4_XS.gguf
echo.

REM メモリ監視開始
echo 📊 システム情報:
systeminfo | findstr "Total Physical Memory"
echo.

REM KoboldCPP起動
echo 🔧 KoboldCPP起動中...
python koboldcpp.py --model "C:\Users\downl\Downloads\EasyNovelAssistant-main\EasyNovelAssistant-main\KoboldCpp\Vecteus-v1-IQ4_XS.gguf" --contextsize 4096 --blasbatchsize 256 --blasthreads 4 --port 5001 --skiplauncher --gpulayers 29 --usecublas normal 0 --nommap --usemlock False --noavx2 --failsafe

pause
