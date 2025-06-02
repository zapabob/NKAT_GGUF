@echo off
echo 🔧 NKAT安定性修正済み推論実行
echo 設定レベル: high_stability
echo Gamma: 0.97, Rank: 8
echo.

py -3 nkat_inference_engine.py ^
  --model "models/test/test_large_NKAT_real.gguf" ^
  --benchmark ^
  --seq-len 512 ^
  --iterations 5 ^
  --theta-gamma 0.97

echo.
echo 推論完了
pause
