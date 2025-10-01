@echo off
setlocal
rem ============================================================================
rem PHIQ Elastic KV Cache - Windows Build Script
rem Production-Grade CUDA Compilation for GTX 1070 (SM 6.1)
rem ============================================================================

echo PHIQ Elastic KV Cache - Windows Build Script
echo ============================================

rem Check for CUDA
where nvcc >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: CUDA Toolkit not found. Please install CUDA 11.8 or higher.
    exit /b 1
)

rem Check for Visual Studio
if not exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    if not exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        echo Error: Visual Studio 2019 not found.
        exit /b 1
    )
)

rem Set environment variables
set BUILD_TYPE=Release
set CUDA_ARCH=61
set OUTPUT_DIR=build

echo Build Type: %BUILD_TYPE%
echo CUDA Architecture: SM_%CUDA_ARCH%
echo Output Directory: %OUTPUT_DIR%
echo.

rem Set up Visual Studio environment
echo Setting up Visual Studio 2019 environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul
if %ERRORLEVEL% NEQ 0 (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
)

rem Create build directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%
cd %OUTPUT_DIR%

rem Direct NVCC compilation (fallback if CMake not available)
echo Compiling with NVCC...
nvcc.exe ^
    -arch=sm_%CUDA_ARCH% ^
    -O3 ^
    --use_fast_math ^
    --maxrregcount=64 ^
    -std=c++17 ^
    -Xcompiler "/EHsc /MD /O2 /fp:fast /GS- /favor:INTEL64" ^
    -DOPTIMAL_BLOCK_SIZE=256 ^
    -DVECTOR_WIDTH=4 ^
    -DTHREADS_PER_BLOCK=256 ^
    ..\src\elastic_kv_cli.cu ^
    -o elastic_kv_cli.exe

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Build completed successfully!
    echo Executable: %OUTPUT_DIR%\elastic_kv_cli.exe
    dir elastic_kv_cli.exe
    echo.

    rem Run quick test
    echo Running quick test...
    elastic_kv_cli.exe --seq=1024 --compress=2 --reps=10 --json
    echo.
    echo Build and test completed successfully!
) else (
    echo Error: Build failed
    exit /b 1
)

pause