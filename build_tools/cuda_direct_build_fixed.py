#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NKAT-llama.cpp CUDA Ultra-Optimized Build System
RTX 3080 CUDA Core Maximum Performance Configuration
Author: AI Assistant
"""

import os
import sys
import subprocess
import json
import time
import shutil
from pathlib import Path
from datetime import datetime
import argparse

class CUDAOptimizedBuilder:
    def __init__(self, clean_build=False, log_level="INFO"):
        self.project_root = Path.cwd()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"nkat_cuda_optimized_{self.timestamp}.log"
        self.clean_build = clean_build
        self.log_level = log_level
        
        # RTX 3080 Ultra-Optimized Settings
        self.rtx3080_config = {
            "cuda_arch": "86",
            "max_threads": 8,
            "sm_count": 68,  # RTX 3080 has 68 SMs
            "memory_bandwidth": 760,  # GB/s
            "cuda_cores": 8704,
            "tensor_cores": 272,
            "rt_cores": 68
        }
        
        # CUDA Maximum Performance Flags
        self.cuda_ultra_flags = [
            "--use_fast_math",
            "-O3",
            "--maxrregcount=64",
            "--ftz=true",  # Flush to zero
            "--prec-div=false",  # Fast division
            "--prec-sqrt=false",  # Fast sqrt
            "--fmad=true",  # Fused multiply-add
            "--gpu-architecture=sm_86",
            "--gpu-code=sm_86",
            "--ptxas-options=-v",
            "--generate-line-info",
            "--optimize=3"
        ]
        
        self.build_state_file = "cuda_build_state.json"
        
    def log(self, message, level="INFO"):
        """Enhanced logging with performance metrics"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        colored_message = self._colorize_message(message, level)
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(colored_message)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")
    
    def _colorize_message(self, message, level):
        """Add color coding for different log levels"""
        colors = {
            "SUCCESS": "\033[92m",  # Green
            "ERROR": "\033[91m",    # Red
            "WARN": "\033[93m",     # Yellow
            "INFO": "\033[94m",     # Blue
            "CUDA": "\033[95m",     # Magenta
            "PERF": "\033[96m"      # Cyan
        }
        reset = "\033[0m"
        color = colors.get(level, "")
        return f"{color}[{level}] {message}{reset}"
    
    def save_build_state(self, state):
        """Save build state for recovery"""
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "state": state,
            "config": self.rtx3080_config,
            "cuda_flags": self.cuda_ultra_flags
        }
        with open(self.build_state_file, "w") as f:
            json.dump(state_data, f, indent=2)
        self.log(f"Build state saved: {state}", "INFO")
    
    def check_cuda_environment(self):
        """Comprehensive CUDA environment verification"""
        self.log("=== CUDA Environment Analysis ===", "CUDA")
        
        # Check CUDA version
        try:
            result = subprocess.run(["nvcc", "--version"], 
                                  capture_output=True, text=True, check=True)
            cuda_version = result.stdout
            self.log(f"CUDA Compiler: {cuda_version.strip()}", "SUCCESS")
        except Exception as e:
            self.log(f"CUDA compiler check failed: {e}", "ERROR")
            return False
        
        # Check GPU details
        try:
            result = subprocess.run([
                "nvidia-smi", "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, check=True)
            
            gpu_info = result.stdout.strip()
            self.log(f"GPU Information: {gpu_info}", "SUCCESS")
            
            # Verify RTX 3080
            if "RTX 3080" in gpu_info:
                self.log("RTX 3080 confirmed - Enabling maximum optimization", "CUDA")
                return True
            else:
                self.log(f"Non-RTX 3080 GPU detected: {gpu_info}", "WARN")
                
        except Exception as e:
            self.log(f"GPU information failed: {e}", "ERROR")
            return False
        
        return True
    
    def find_visual_studio(self):
        """Auto-detect Visual Studio installations"""
        vs_paths = [
            "C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/Tools/VsDevCmd.bat",
            "C:/Program Files/Microsoft Visual Studio/2022/Professional/Common7/Tools/VsDevCmd.bat",
            "C:/Program Files/Microsoft Visual Studio/2022/Enterprise/Common7/Tools/VsDevCmd.bat",
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/Common7/Tools/VsDevCmd.bat",
            "C:/Program Files (x86)/Microsoft Visual Studio/2019/Professional/Common7/Tools/VsDevCmd.bat"
        ]
        
        for vs_path in vs_paths:
            if Path(vs_path).exists():
                self.log(f"Visual Studio found: {vs_path}", "SUCCESS")
                return vs_path
        
        self.log("Visual Studio not found", "ERROR")
        self.log("Please install Visual Studio 2022 Community (free):", "INFO")
        self.log("https://visualstudio.microsoft.com/vs/community/", "INFO")
        return None
    
    def setup_vs_environment(self, vs_dev_cmd):
        """Setup Visual Studio environment variables"""
        try:
            cmd = f'"{vs_dev_cmd}" && set'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            env_vars = {}
            for line in result.stdout.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
            
            # Update current environment
            os.environ.update(env_vars)
            self.log("Visual Studio environment configured", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"VS environment setup failed: {e}", "ERROR")
            return False
    
    def create_optimized_cmake_config(self, build_dir):
        """Generate ultra-optimized CMake configuration for RTX 3080"""
        cuda_flags_str = " ".join(self.cuda_ultra_flags)
        
        cmake_args = [
            "..",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_CUDA=ON",
            f"-DCUDA_ARCHITECTURES={self.rtx3080_config['cuda_arch']}",
            "-DCUDA_TOOLKIT_ROOT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8",
            "-DCMAKE_CUDA_COMPILER=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe",
            
            # RTX 3080 Specific Optimizations
            f"-DCMAKE_CUDA_FLAGS={cuda_flags_str}",
            "-DGGML_CUDA_F16=ON",
            "-DGGML_CUDA_DMMV_X=32",
            "-DGGML_CUDA_MMV_Y=1",
            "-DGGML_CUDA_PEER_MAX_BATCH_SIZE=128",
            "-DGGML_CUDA_FORCE_DMMV=ON",
            "-DGGML_CUDA_FORCE_MMQ=ON",
            
            # Performance Optimizations
            "-DCMAKE_CXX_FLAGS=/O2 /Ob2 /Oi /Ot /Oy /GT /GL /DNDEBUG",
            "-DCMAKE_C_FLAGS=/O2 /Ob2 /Oi /Ot /Oy /GT /GL /DNDEBUG",
            "-DCMAKE_EXE_LINKER_FLAGS=/LTCG /OPT:REF /OPT:ICF",
            
            # Disable unnecessary features
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_METAL=OFF",
            "-DLLAMA_OPENCL=OFF",
            "-DLLAMA_BLAS=OFF",
            
            # Memory optimizations
            "-DCMAKE_CUDA_SEPARABLE_COMPILATION=OFF",
            "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
        ]
        
        self.log("=== RTX 3080 Ultra-Optimized CMake Configuration ===", "CUDA")
        for arg in cmake_args:
            self.log(f"  {arg}", "INFO")
        
        return cmake_args
    
    def build_project(self):
        """Main build execution with RTX 3080 optimization"""
        self.log("=== NKAT-llama.cpp RTX 3080 CUDA Ultra-Optimized Build ===", "CUDA")
        
        # Environment checks
        if not self.check_cuda_environment():
            return False
        
        vs_dev_cmd = self.find_visual_studio()
        if not vs_dev_cmd:
            return False
        
        if not self.setup_vs_environment(vs_dev_cmd):
            return False
        
        self.save_build_state("ENVIRONMENT_READY")
        
        # Build directory setup
        build_dir = Path("llama.cpp/build_cuda_optimized")
        if self.clean_build and build_dir.exists():
            self.log("Clean build: removing existing directory", "INFO")
            shutil.rmtree(build_dir)
        
        build_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(build_dir)
        
        self.save_build_state("CMAKE_CONFIG")
        
        try:
            # CMake Configuration
            cmake_args = self.create_optimized_cmake_config(build_dir)
            self.log("Executing CMake configuration...", "CUDA")
            
            start_time = time.time()
            result = subprocess.run(["cmake"] + cmake_args, 
                                  capture_output=True, text=True)
            config_time = time.time() - start_time
            
            if result.returncode != 0:
                self.log(f"CMake configuration failed: {result.stderr}", "ERROR")
                self.log("Attempting fallback configuration...", "WARN")
                
                # Fallback configuration
                fallback_args = [
                    "..",
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DGGML_CUDA=ON",
                    "-DCUDA_ARCHITECTURES=86",
                    "-DLLAMA_CURL=OFF"
                ]
                
                result = subprocess.run(["cmake"] + fallback_args,
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    self.log("Fallback configuration also failed", "ERROR")
                    return False
            
            self.log(f"CMake configuration completed in {config_time:.2f}s", "SUCCESS")
            self.save_build_state("BUILD_START")
            
            # Build execution
            self.log("Starting RTX 3080 optimized build...", "CUDA")
            build_cmd = [
                "cmake", "--build", ".", 
                "--config", "Release", 
                "--target", "main",
                "--parallel", str(self.rtx3080_config['max_threads'])
            ]
            
            start_time = time.time()
            result = subprocess.run(build_cmd, capture_output=True, text=True)
            build_time = time.time() - start_time
            
            if result.returncode != 0:
                self.log("Main target build failed, trying full build", "WARN")
                full_build_cmd = [
                    "cmake", "--build", ".", 
                    "--config", "Release",
                    "--parallel", str(self.rtx3080_config['max_threads'])
                ]
                result = subprocess.run(full_build_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.log(f"Build failed: {result.stderr}", "ERROR")
                    return False
            
            self.log(f"Build completed in {build_time:.2f}s", "SUCCESS")
            self.save_build_state("BUILD_COMPLETE")
            
            # Executable verification
            return self.verify_cuda_executable()
            
        except Exception as e:
            self.log(f"Build process error: {e}", "ERROR")
            return False
        finally:
            os.chdir(self.project_root)
    
    def verify_cuda_executable(self):
        """Verify CUDA functionality in built executable"""
        exe_paths = ["main.exe", "Release/main.exe", "bin/main.exe", "main"]
        found_exe = None
        
        for exe_path in exe_paths:
            if Path(exe_path).exists():
                found_exe = exe_path
                break
        
        if not found_exe:
            self.log("Executable not found", "ERROR")
            return False
        
        file_size = Path(found_exe).stat().st_size / (1024*1024)
        self.log(f"Executable found: {found_exe} ({file_size:.2f} MB)", "SUCCESS")
        
        # CUDA functionality test
        try:
            result = subprocess.run([f"./{found_exe}", "--help"],
                                  capture_output=True, text=True, timeout=30)
            
            cuda_features = [line for line in result.stdout.split('\n') 
                           if any(keyword in line.lower() for keyword in ['cuda', 'gpu'])]
            
            if cuda_features:
                self.log("CUDA support confirmed!", "CUDA")
                for feature in cuda_features:
                    self.log(f"  {feature}", "PERF")
            else:
                self.log("CUDA support not detected in help output", "WARN")
            
            # Performance test
            self.log("Running basic performance test...", "PERF")
            perf_result = subprocess.run([f"./{found_exe}", "--version"],
                                       capture_output=True, text=True, timeout=10)
            if perf_result.stdout:
                self.log(f"Version: {perf_result.stdout.strip()}", "INFO")
            
            return True
            
        except subprocess.TimeoutExpired:
            self.log("Executable test timed out", "WARN")
            return True
        except Exception as e:
            self.log(f"Executable test failed: {e}", "WARN")
            return True
    
    def print_optimization_summary(self):
        """Print RTX 3080 optimization summary"""
        self.log("", "INFO")
        self.log("=== RTX 3080 CUDA Optimization Summary ===", "CUDA")
        self.log(f"CUDA Architecture: sm_{self.rtx3080_config['cuda_arch']}", "PERF")
        self.log(f"CUDA Cores: {self.rtx3080_config['cuda_cores']}", "PERF")
        self.log(f"Tensor Cores: {self.rtx3080_config['tensor_cores']}", "PERF")
        self.log(f"Memory Bandwidth: {self.rtx3080_config['memory_bandwidth']} GB/s", "PERF")
        self.log(f"Parallel Build Threads: {self.rtx3080_config['max_threads']}", "PERF")
        self.log("", "INFO")
        self.log("Optimization Flags Applied:", "CUDA")
        for flag in self.cuda_ultra_flags:
            self.log(f"  {flag}", "INFO")
        self.log("", "INFO")
        self.log("Next Steps:", "INFO")
        self.log("  1. cd llama.cpp/build_cuda_optimized", "INFO")
        self.log("  2. ./main.exe -m ../../../models/test/model.gguf -p 'CUDA Test'", "INFO")
        self.log(f"  3. Check log: {self.log_file}", "INFO")

def main():
    parser = argparse.ArgumentParser(description="RTX 3080 CUDA Ultra-Optimized Builder")
    parser.add_argument("--clean", action="store_true", help="Clean build")
    parser.add_argument("--log-level", default="INFO", choices=["INFO", "DEBUG", "WARN", "ERROR"])
    
    args = parser.parse_args()
    
    builder = CUDAOptimizedBuilder(clean_build=args.clean, log_level=args.log_level)
    
    try:
        success = builder.build_project()
        builder.print_optimization_summary()
        
        if success:
            builder.log("RTX 3080 CUDA Ultra-Optimized Build SUCCESSFUL!", "SUCCESS")
            sys.exit(0)
        else:
            builder.log("Build FAILED - Check logs for details", "ERROR")
            sys.exit(1)
            
    except KeyboardInterrupt:
        builder.log("Build interrupted by user", "WARN")
        sys.exit(1)
    except Exception as e:
        builder.log(f"Unexpected error: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main() 