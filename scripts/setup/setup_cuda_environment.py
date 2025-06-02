#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTX 3080 CUDA Environment Setup and Optimization
Visual Studio Integration Automation
"""

import os
import sys
import subprocess
import winreg
from pathlib import Path
from datetime import datetime

class RTX3080CUDAEnvironment:
    def __init__(self):
        self.cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8"
        self.vs_paths = [
            "C:/Program Files/Microsoft Visual Studio/2022/Community",
            "C:/Program Files/Microsoft Visual Studio/2022/Professional", 
            "C:/Program Files/Microsoft Visual Studio/2022/Enterprise",
            "C:/Program Files/Microsoft Visual Studio/2022/BuildTools"
        ]
        
        # RTX 3080 Ultimate Performance Settings
        self.rtx3080_ultimate = {
            "arch": "sm_86",
            "compute_capability": "8.6",
            "cuda_cores": 8704,
            "tensor_cores": 272,
            "memory_bandwidth": 760,  # GB/s
            "l2_cache": 6,  # MB
            "max_threads_per_sm": 1536,
            "max_blocks_per_sm": 16,
            "warp_size": 32,
            "max_shared_memory": 99  # KB per SM
        }
        
    def log(self, message, level="INFO"):
        """Enhanced logging"""
        colors = {
            "SUCCESS": "\033[92m",
            "ERROR": "\033[91m", 
            "WARN": "\033[93m",
            "INFO": "\033[94m",
            "CUDA": "\033[95m",
            "PERF": "\033[96m"
        }
        reset = "\033[0m"
        color = colors.get(level, "")
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"{color}[{timestamp}] [{level}] {message}{reset}")
    
    def check_cuda_installation(self):
        """Verify CUDA 12.8 installation"""
        self.log("=== CUDA 12.8 Installation Verification ===", "CUDA")
        
        # Check CUDA path
        if not Path(self.cuda_path).exists():
            self.log(f"CUDA path not found: {self.cuda_path}", "ERROR")
            return False
        
        # Check nvcc
        nvcc_path = Path(self.cuda_path) / "bin" / "nvcc.exe"
        if not nvcc_path.exists():
            self.log(f"nvcc.exe not found: {nvcc_path}", "ERROR")
            return False
        
        # Check CUDA libraries
        cuda_libs = ["cublas", "cudart", "cufft", "curand", "cusparse"]
        lib_path = Path(self.cuda_path) / "lib" / "x64"
        
        missing_libs = []
        for lib in cuda_libs:
            lib_file = lib_path / f"{lib}.lib"
            if not lib_file.exists():
                missing_libs.append(lib)
        
        if missing_libs:
            self.log(f"Missing CUDA libraries: {missing_libs}", "WARN")
        else:
            self.log("All essential CUDA libraries found", "SUCCESS")
        
        return True
    
    def detect_visual_studio(self):
        """Detect Visual Studio installations"""
        self.log("=== Visual Studio Detection ===", "INFO")
        
        found_vs = []
        for vs_path in self.vs_paths:
            if Path(vs_path).exists():
                # Check for essential components
                vcvars_path = Path(vs_path) / "Common7" / "Tools" / "VsDevCmd.bat"
                msbuild_path = Path(vs_path) / "MSBuild" / "Current" / "Bin" / "MSBuild.exe"
                
                if vcvars_path.exists() and msbuild_path.exists():
                    found_vs.append(vs_path)
                    self.log(f"Visual Studio found: {vs_path}", "SUCCESS")
        
        if not found_vs:
            self.log("No Visual Studio installation detected", "WARN")
            self.log("Waiting for Visual Studio installation...", "INFO")
            return None
        
        return found_vs[0]  # Return first found installation
    
    def setup_environment_variables(self):
        """Setup optimal environment variables for RTX 3080"""
        self.log("=== RTX 3080 Environment Variables Setup ===", "PERF")
        
        cuda_env_vars = {
            "CUDA_PATH": self.cuda_path,
            "CUDA_TOOLKIT_ROOT_DIR": self.cuda_path,
            "CUDA_HOME": self.cuda_path,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",  # Primary GPU
            "CUDA_CACHE_PATH": str(Path.home() / ".nv" / "ComputeCache"),
            
            # RTX 3080 Performance Optimizations
            "CUDA_LAUNCH_BLOCKING": "0",  # Async launches for performance
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",  # Max concurrent kernels
            "CUDA_AUTO_BOOST": "1",  # Enable GPU Boost
            "CUDA_FORCE_PTX_JIT": "0",  # Use compiled SASS when available
            
            # Memory optimizations
            "CUDA_MALLOC_HEAP_SIZE": str(256 * 1024 * 1024),  # 256MB heap
            "CUDA_STACK_SIZE": str(8 * 1024),  # 8KB stack per thread
            
            # Compiler optimizations
            "CUDA_ENABLE_COREDUMP_ON_EXCEPTION": "0",
            "CUDA_ENABLE_CPU_MEMORY_FALLBACK": "0"
        }
        
        # Apply environment variables
        for key, value in cuda_env_vars.items():
            os.environ[key] = value
            self.log(f"Set {key}={value}", "INFO")
        
        # Update PATH
        cuda_bin = str(Path(self.cuda_path) / "bin")
        current_path = os.environ.get("PATH", "")
        if cuda_bin not in current_path:
            os.environ["PATH"] = f"{cuda_bin};{current_path}"
            self.log(f"Added to PATH: {cuda_bin}", "SUCCESS")
    
    def create_cuda_optimization_header(self):
        """Create RTX 3080 optimization header file"""
        self.log("Creating RTX 3080 optimization header...", "CUDA")
        
        header_content = f'''#pragma once
/* 
 * RTX 3080 CUDA Ultra-Optimization Header
 * Generated automatically for maximum performance
 */

#ifndef RTX3080_CUDA_OPTIMIZATIONS_H
#define RTX3080_CUDA_OPTIMIZATIONS_H

// RTX 3080 Hardware Specifications
#define RTX3080_CUDA_ARCH               {self.rtx3080_ultimate["arch"]}
#define RTX3080_COMPUTE_CAPABILITY      {self.rtx3080_ultimate["compute_capability"]}
#define RTX3080_CUDA_CORES              {self.rtx3080_ultimate["cuda_cores"]}
#define RTX3080_TENSOR_CORES            {self.rtx3080_ultimate["tensor_cores"]}
#define RTX3080_MEMORY_BANDWIDTH_GBPS   {self.rtx3080_ultimate["memory_bandwidth"]}
#define RTX3080_L2_CACHE_MB             {self.rtx3080_ultimate["l2_cache"]}
#define RTX3080_MAX_THREADS_PER_SM      {self.rtx3080_ultimate["max_threads_per_sm"]}
#define RTX3080_MAX_BLOCKS_PER_SM       {self.rtx3080_ultimate["max_blocks_per_sm"]}
#define RTX3080_WARP_SIZE               {self.rtx3080_ultimate["warp_size"]}
#define RTX3080_MAX_SHARED_MEMORY_KB    {self.rtx3080_ultimate["max_shared_memory"]}

// Optimal Thread Block Configurations
#define RTX3080_OPTIMAL_BLOCK_SIZE_1D   256
#define RTX3080_OPTIMAL_BLOCK_SIZE_2D_X 16
#define RTX3080_OPTIMAL_BLOCK_SIZE_2D_Y 16
#define RTX3080_OPTIMAL_GRID_SIZE       ((RTX3080_CUDA_CORES + RTX3080_OPTIMAL_BLOCK_SIZE_1D - 1) / RTX3080_OPTIMAL_BLOCK_SIZE_1D)

// Performance Compiler Directives
#ifdef __CUDACC__
#define RTX3080_FORCE_INLINE    __forceinline__ __device__
#define RTX3080_GLOBAL          __global__
#define RTX3080_SHARED          __shared__
#define RTX3080_RESTRICT        __restrict__

// Fast Math Functions (RTX 3080 optimized)
#define RTX3080_FAST_DIV(a, b)  __fdividef(a, b)
#define RTX3080_FAST_SQRT(x)    __fsqrt_rn(x)
#define RTX3080_FAST_EXP(x)     __expf(x)
#define RTX3080_FAST_LOG(x)     __logf(x)
#define RTX3080_FAST_SIN(x)     __sinf(x)
#define RTX3080_FAST_COS(x)     __cosf(x)

// Memory Access Patterns (Optimized for RTX 3080 L2 Cache)
#define RTX3080_CACHE_LINE_SIZE 128
#define RTX3080_MEMORY_ALIGN    __align__(RTX3080_CACHE_LINE_SIZE)

// Warp-level Primitives
#define RTX3080_WARP_REDUCE_SUM(val) \\
    do {{ \\
        val += __shfl_down_sync(0xFFFFFFFF, val, 16); \\
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);  \\
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);  \\
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);  \\
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);  \\
    }} while(0)

// Tensor Core Utilization Helpers
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 860  // RTX 3080 supports Tensor Cores
#define RTX3080_TENSOR_CORES_AVAILABLE 1
#include <mma.h>
using namespace nvcuda;
#endif
#endif

#endif // __CUDACC__

// CMake Integration Macros
#define RTX3080_CMAKE_CUDA_FLAGS "-arch=sm_86 --use_fast_math -O3 --maxrregcount=64 --ptxas-options=-v"
#define RTX3080_CMAKE_CXX_FLAGS  "/O2 /Ob2 /Oi /Ot /Oy /GT /GL /DNDEBUG /DRTX3080_OPTIMIZED"

#endif // RTX3080_CUDA_OPTIMIZATIONS_H
'''
        
        output_path = Path("llama.cpp") / "rtx3080_optimizations.h"
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(header_content)
        
        self.log(f"RTX 3080 optimization header created: {output_path}", "SUCCESS")
        return output_path
    
    def wait_for_visual_studio(self, timeout_minutes=30):
        """Wait for Visual Studio installation completion"""
        self.log(f"Waiting for Visual Studio installation (timeout: {timeout_minutes} min)...", "INFO")
        
        import time
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        while time.time() - start_time < timeout_seconds:
            vs_path = self.detect_visual_studio()
            if vs_path:
                self.log("Visual Studio installation detected!", "SUCCESS")
                return vs_path
            
            # Check every 30 seconds
            time.sleep(30)
            elapsed = int((time.time() - start_time) / 60)
            self.log(f"Still waiting... ({elapsed}/{timeout_minutes} min)", "INFO")
        
        self.log("Timeout waiting for Visual Studio installation", "WARN")
        return None
    
    def run_setup(self):
        """Main setup routine"""
        self.log("=== RTX 3080 CUDA Environment Setup Starting ===", "CUDA")
        
        # 1. Check CUDA
        if not self.check_cuda_installation():
            self.log("CUDA installation issues detected", "ERROR")
            return False
        
        # 2. Setup environment variables
        self.setup_environment_variables()
        
        # 3. Create optimization header
        self.create_cuda_optimization_header()
        
        # 4. Wait for or detect Visual Studio
        vs_path = self.detect_visual_studio()
        if not vs_path:
            vs_path = self.wait_for_visual_studio()
        
        if vs_path:
            self.log(f"RTX 3080 CUDA environment ready with Visual Studio: {vs_path}", "SUCCESS")
            self.log("Ready to execute RTX 3080 optimized build!", "PERF")
            return True
        else:
            self.log("Visual Studio not available - manual installation required", "WARN")
            return False

def main():
    print("🚀 RTX 3080 CUDA Environment Setup & Visual Studio Integration")
    print("=" * 70)
    
    env_setup = RTX3080CUDAEnvironment()
    success = env_setup.run_setup()
    
    if success:
        print("\n✅ RTX 3080 CUDA Environment Setup Complete!")
        print("🎯 Ready for maximum performance CUDA build!")
        print("\nNext step:")
        print("   py -3 cuda_direct_build_fixed.py --clean")
    else:
        print("\n⚠️ Setup incomplete - Visual Studio installation required")
        print("💡 Install Visual Studio 2022 Community and run this script again")
    
    return success

if __name__ == "__main__":
    sys.exit(0 if main() else 1) 