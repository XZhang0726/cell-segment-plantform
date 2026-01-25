@echo off
chcp 65001 >nul
echo ========================================
echo   细胞分割平台启动脚本
echo ========================================
echo.

echo [1/3] 检查环境...
call conda activate cellpose_gpu
if errorlevel 1 (
    echo.
    echo ❌ 错误：cellpose_gpu环境不存在！
    echo.
    echo 请先创建环境，运行以下命令：
    echo   mamba create -n cellpose_gpu python=3.12 -y
    echo   conda activate cellpose_gpu
    echo   mamba install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
    echo   pip install cellpose streamlit loguru opencv-python scikit-image pandas openpyxl
    echo.
    pause
    exit /b 1
)

echo ✅ 环境已激活：cellpose_gpu
echo.

echo [2/3] 进入项目目录...
cd /d "%~dp0"
echo ✅ 当前目录：%CD%
echo.

echo [3/3] 启动Streamlit应用...
echo.
echo ========================================
echo   应用启动中...
echo   浏览器将自动打开
echo   按 Ctrl+C 停止应用
echo ========================================
echo.

REM 设置环境变量确保浏览器自动打开
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

REM 启动应用（不使用headless模式）
start /B streamlit run app_enhanced.py --server.headless=false

REM 等待3秒让服务器启动
timeout /t 3 /nobreak >nul

REM 手动打开浏览器
start http://localhost:8501

pause
