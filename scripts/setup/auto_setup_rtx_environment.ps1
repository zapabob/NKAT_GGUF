# RTX30/RTX40シリーズ向け自動環境セットアップスクリプト
[CmdletBinding()]
param(
    [Parameter(Mandatory=$false)]
    [switch]$Force,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipCUDA,
    
    [Parameter(Mandatory=$false)]
    [switch]$SkipPython
)

Write-Host "NKAT-GGUF Auto Setup for RTX30/RTX40 Series" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Green

# 管理者権限チェック
function Test-IsAdmin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-IsAdmin)) {
    Write-Host "Administrator privileges required for some installations" -ForegroundColor Yellow
    Write-Host "   Please run as Administrator for best results" -ForegroundColor Yellow
}

# GPU検出
Write-Host "Detecting GPU..." -ForegroundColor Cyan
try {
    $gpuInfo = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    Write-Host "   GPU Detected: $gpuInfo" -ForegroundColor Green
    
    if ($gpuInfo -match "RTX (30|40)") {
        Write-Host "RTX 30/40 series GPU detected" -ForegroundColor Green
    } else {
        Write-Host "No RTX 30/40 series GPU detected" -ForegroundColor Yellow
        Write-Host "   This setup is optimized for RTX 30/40 series" -ForegroundColor Yellow
    }
} catch {
    Write-Host "NVIDIA GPU not detected or nvidia-smi not available" -ForegroundColor Red
    if (-not $Force) {
        Write-Host "   Use -Force to continue anyway" -ForegroundColor Yellow
        exit 1
    }
}

# CUDA Toolkit チェック・インストール
if (-not $SkipCUDA) {
    Write-Host "`nChecking CUDA Toolkit..." -ForegroundColor Cyan
    
    $cudaPaths = @(
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
    )
    
    $cudaFound = $false
    foreach ($path in $cudaPaths) {
        if (Test-Path "$path\bin\nvcc.exe") {
            Write-Host "CUDA Toolkit found: $path" -ForegroundColor Green
            $env:CUDA_PATH = $path
            $cudaFound = $true
            break
        }
    }
    
    if (-not $cudaFound) {
        Write-Host "CUDA Toolkit not found" -ForegroundColor Yellow
        Write-Host "   Please install CUDA Toolkit 12.x from:" -ForegroundColor Yellow
        Write-Host "   https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
        
        if (-not $Force) {
            Write-Host "   Use -Force to continue without CUDA" -ForegroundColor Yellow
            exit 1
        }
    }
}

# Visual Studio Build Tools チェック
Write-Host "`nChecking Visual Studio Build Tools..." -ForegroundColor Cyan
$vsPaths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools",
    "C:\Program Files\Microsoft Visual Studio\2022\Community",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional"
)

$vsFound = $false
foreach ($path in $vsPaths) {
    if (Test-Path "$path\VC\Auxiliary\Build\vcvarsall.bat") {
        Write-Host "Visual Studio Build Tools found: $path" -ForegroundColor Green
        $vsFound = $true
        break
    }
}

if (-not $vsFound) {
    Write-Host "Visual Studio Build Tools not found" -ForegroundColor Yellow
    Write-Host "   Installing Visual Studio Build Tools..." -ForegroundColor Yellow
    
    # Build Tools自動インストール
    try {
        $buildToolsUrl = "https://aka.ms/vs/17/release/vs_buildtools.exe"
        $installer = "$env:TEMP\vs_buildtools.exe"
        
        Write-Host "   Downloading Build Tools installer..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri $buildToolsUrl -OutFile $installer
        
        Write-Host "   Installing Build Tools (this may take a while)..." -ForegroundColor Yellow
        Start-Process -FilePath $installer -ArgumentList "--quiet", "--wait", "--add", "Microsoft.VisualStudio.Workload.VCTools", "--add", "Microsoft.VisualStudio.Component.VC.CMake.Project" -Wait
        
        Remove-Item $installer -Force
        Write-Host "Visual Studio Build Tools installed" -ForegroundColor Green
    } catch {
        Write-Host "Failed to install Build Tools automatically" -ForegroundColor Red
        Write-Host "   Please install manually from: https://visualstudio.microsoft.com/downloads/" -ForegroundColor Yellow
    }
}

# Python環境チェック
if (-not $SkipPython) {
    Write-Host "`nChecking Python environment..." -ForegroundColor Cyan
    
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "   Python version: $pythonVersion" -ForegroundColor Green
        
        # 必要なパッケージインストール
        Write-Host "   Installing Python dependencies..." -ForegroundColor Yellow
        
        $packages = @(
            "torch", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121",
            "transformers",
            "optuna", 
            "matplotlib",
            "tqdm",
            "psutil",
            "numpy",
            "scipy"
        )
        
        py -3 -m pip install --upgrade pip
        py -3 -m pip install @packages
        
        Write-Host "Python dependencies installed" -ForegroundColor Green
    } catch {
        Write-Host "Python not found or installation failed" -ForegroundColor Red
        Write-Host "   Please install Python 3.8+ from: https://www.python.org/downloads/" -ForegroundColor Yellow
    }
}

# CMake チェック
Write-Host "`nChecking CMake..." -ForegroundColor Cyan
try {
    $cmakeVersion = cmake --version
    Write-Host "CMake found: $($cmakeVersion.Split("`n")[0])" -ForegroundColor Green
} catch {
    Write-Host "CMake not found" -ForegroundColor Yellow
    Write-Host "   Installing CMake..." -ForegroundColor Yellow
    
    try {
        # CMake自動インストール
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install Kitware.CMake
            Write-Host "CMake installed via winget" -ForegroundColor Green
        } else {
            Write-Host "   Please install CMake manually from: https://cmake.org/download/" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Failed to install CMake automatically" -ForegroundColor Red
    }
}

# Git LFS チェック
Write-Host "`nChecking Git LFS..." -ForegroundColor Cyan
try {
    $gitLfsVersion = git lfs version
    Write-Host "Git LFS found: $gitLfsVersion" -ForegroundColor Green
} catch {
    Write-Host "Git LFS not found" -ForegroundColor Yellow
    Write-Host "   Installing Git LFS..." -ForegroundColor Yellow
    
    try {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install GitHub.GitLFS
            Write-Host "Git LFS installed" -ForegroundColor Green
        } else {
            Write-Host "   Please install Git LFS manually from: https://git-lfs.github.io/" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "Failed to install Git LFS automatically" -ForegroundColor Red
    }
}

# 環境変数設定
Write-Host "`nSetting up environment variables..." -ForegroundColor Cyan

if ($env:CUDA_PATH) {
    $env:PATH = "$env:CUDA_PATH\bin;" + $env:PATH
    Write-Host "   CUDA_PATH set to: $env:CUDA_PATH" -ForegroundColor Green
}

# 設定ファイル作成
Write-Host "`nCreating configuration files..." -ForegroundColor Cyan

$envConfig = @{
    "setup_date" = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    "cuda_path" = $env:CUDA_PATH
    "rtx_optimized" = $true
    "last_check" = (Get-Date).ToString("yyyy-MM-dd")
}

$configJson = $envConfig | ConvertTo-Json -Depth 2
$configPath = "environment_config.json"
$configJson | Out-File -FilePath $configPath -Encoding UTF8

Write-Host "Environment configuration saved to: $configPath" -ForegroundColor Green

# セットアップ完了
Write-Host "`nSetup completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "`nSummary:" -ForegroundColor Cyan
Write-Host "   GPU Detection: $(if ($gpuInfo) { 'OK' } else { 'FAILED' })" -ForegroundColor White
Write-Host "   CUDA Toolkit: $(if ($cudaFound) { 'OK' } else { 'FAILED' })" -ForegroundColor White
Write-Host "   Visual Studio: $(if ($vsFound) { 'OK' } else { 'FAILED' })" -ForegroundColor White
Write-Host "   Python: $(if (-not $SkipPython) { 'OK' } else { 'Skipped' })" -ForegroundColor White

Write-Host "`nNext Steps:" -ForegroundColor Cyan
Write-Host "   1. Organize repository:" -ForegroundColor White
Write-Host "      py -3 repository_cleanup_organizer.py" -ForegroundColor Yellow
Write-Host "   2. Build the project:" -ForegroundColor White  
Write-Host "      .\scripts\build\universal_build.ps1" -ForegroundColor Yellow
Write-Host "   3. Run benchmarks:" -ForegroundColor White
Write-Host "      py -3 scripts\benchmark\comprehensive_benchmark.py" -ForegroundColor Yellow

if (-not $cudaFound -or -not $vsFound) {
    Write-Host "`nWarning: Some components missing" -ForegroundColor Yellow
    Write-Host "   Please install missing components and re-run setup" -ForegroundColor Yellow
} 