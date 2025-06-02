@echo off
echo 🔧 NKAT安定性修正済み推論実行
echo 設定レベル: balanced_stability
echo Gamma: 0.95, Rank: 6
echo.

py -3 nkat_inference_engine.py ^
  --model "models/test/test_large_NKAT_real.gguf" ^
  --benchmark ^
  --seq-len 512 ^
  --iterations 5 ^
  --theta-gamma 0.95

echo.
echo 推論完了
pause
