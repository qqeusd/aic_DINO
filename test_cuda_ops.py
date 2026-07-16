"""测试 CUDA 算子是否正确编译和导入"""
import sys
import os

# === 关键：torch 必须先被 import，它的 DLL 目录才能被注册到搜索路径 ===
# CUDA 扩展 .pyd 依赖 c10.dll, torch_cpu.dll 等，这些由 torch 提供

# 用脚本所在目录推导项目根目录（消除硬编码路径）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = _SCRIPT_DIR
_OPS_DIR = os.path.join(_PROJECT_ROOT, 'models', 'dino', 'ops')

# 额外添加 CUDA toolkit DLL 目录（Windows 专用）
if sys.platform == 'win32':
    # 尝试查找系统中的 CUDA bin 目录
    cuda_candidates = [
        os.environ.get('CUDA_PATH', ''),
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
    ]
    for cuda_bin in cuda_candidates:
        if cuda_bin and os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)
            print(f"📁 注册 CUDA DLL 路径: {cuda_bin}")
            break

# 先 import torch — 它会自动注册自己的 lib 目录到 DLL 搜索路径
print("📦 导入 torch...")
import torch
print(f"   PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

# 注册 torch lib 到 DLL 搜索路径
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if sys.platform == 'win32' and os.path.isdir(torch_lib):
    os.add_dll_directory(torch_lib)

# 现在尝试导入编译好的 CUDA 扩展
# 自动检测编译产物目录
_ops_build_dir = os.path.join(_OPS_DIR, 'build')
_built_libs = []
if os.path.isdir(_ops_build_dir):
    for _d in os.listdir(_ops_build_dir):
        _build_path = os.path.join(_ops_build_dir, _d, 'lib')
        if os.path.isdir(_build_path):
            _built_libs.append(_build_path)
        _build_path_direct = os.path.join(_ops_build_dir, _d)
        if os.path.isdir(_build_path_direct):
            # 检查目录名是否包含 .so 或 .pyd
            for _f in os.listdir(_build_path_direct):
                if _f.endswith('.so') or _f.endswith('.pyd'):
                    _built_libs.append(_build_path_direct)
                    break

for _p in _built_libs:
    sys.path.insert(0, _p)
sys.path.insert(0, _OPS_DIR)

# 1. 导入 .pyd / .so
try:
    import MultiScaleDeformableAttention as MSDA
    print(f"✅ MultiScaleDeformableAttention 导入成功")
    print(f"   函数: ms_deform_attn_forward, ms_deform_attn_backward")
except ImportError as e:
    print(f"❌ 导入失败: {e}")

    # 详细诊断
    print(f"   已在 sys.path 中添加:")
    for _p in _built_libs:
        _search_file = os.path.join(_p, 'MultiScaleDeformableAttention')
        print(f"   - {_p}/ ({'有文件' if any(f.startswith('MultiScaleDeformableAttention') for f in os.listdir(_p) if os.path.isfile(os.path.join(_p, f))) else '空目录'})")

    # 列出 torch lib 下的关键 DLL (Windows)
    if sys.platform == 'win32':
        for dll_name in ['c10.dll', 'torch_cpu.dll', 'torch_cuda.dll', 'cudart64_110.dll']:
            dll_path = os.path.join(torch_lib, dll_name)
            print(f"   {'✅' if os.path.exists(dll_path) else '❌'} {dll_name}")

    sys.exit(1)

# 使用 DINO 项目的正常包导入路径
sys.path.insert(0, _PROJECT_ROOT)

try:
    from models.dino.ops.modules import MSDeformAttn
    print("✅ MSDeformAttn 导入成功 (via models.dino.ops.modules)")
except Exception as e:
    print(f"❌ MSDeformAttn 导入失败: {e}")
    sys.exit(1)

# 4. 简单功能测试
try:
    attn = MSDeformAttn(d_model=256, n_levels=4, n_heads=8, n_points=4).cuda()
    print("✅ MSDeformAttn 实例化成功 (CUDA)")

    # 构造测试输入跑一次 forward
    B, N, C = 1, 100, 256
    value = torch.randn(B, N, 8, C // 8).cuda()  # [B, sum(HW), nheads, head_dim]
    spatial_shapes = torch.tensor([[20, 20], [10, 10], [5, 5], [3, 3]]).cuda()
    level_start_index = torch.tensor([0, 400, 500, 525]).cuda()
    sampling_loc = torch.randn(B, N, 8, 4, 4, 2).cuda()
    attn_weight = torch.randn(B, N, 8, 4, 4).softmax(-1).cuda()

    out = attn(value, spatial_shapes, level_start_index, sampling_loc, attn_weight)
    print(f"✅ MSDeformAttn forward 成功! 输出 shape: {out.shape}")
    print(f"   输入: value={value.shape}, 输出: {out.shape}")
except Exception as e:
    print(f"❌ MSDeformAttn 功能测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 所有 CUDA 算子模块验证通过!")
