@echo off
echo === Activating VS 2022 Developer Environment ===
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate VS developer environment
    exit /b 1
)

echo === Setting CUDA environment ===
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8
set TORCH_CUDA_ARCH_LIST=8.9
set DISTUTILS_USE_SDK=1

echo CUDA_HOME=%CUDA_HOME%
echo TORCH_CUDA_ARCH_LIST=%TORCH_CUDA_ARCH_LIST%

echo === Compiling Deformable Attention CUDA Extension ===
cd /d %~dp0models\dino\ops

conda run -n pytorch python setup.py build develop
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo === Compilation FAILED! Trying build_ext --inplace instead ===
    conda run -n pytorch python setup.py build_ext --inplace
)

echo.
echo === Done ===
