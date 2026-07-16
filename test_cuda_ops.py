"""测试 CUDA 算子是否正确编译和导入"""
import sys
import os

# === 关键：torch 必须先被 import，它的 DLL 目录才能被注册到搜索路径 ===
# CUDA 扩展 .pyd 依赖 c10.dll, torch_cpu.dll 等，这些由 torch 提供

# 额外添加 CUDA toolkit DLL 目录
cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
if os.path.isdir(cuda_bin):
    os.add_dll_directory(cuda_bin)
    print(f"📁 注册 DLL 路径: {cuda_bin}")

# 先 import torch — 它会自动注册自己的 lib 目录到 DLL 搜索路径
print("📦 导入 torch...")
import torch
print(f"   PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")

# 注册 torch lib 到 DLL 搜索路径（torch 已自动处理，但显式加一次）
torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if os.path.isdir(torch_lib):
    os.add_dll_directory(torch_lib)

# 现在尝试导入编译好的 CUDA 扩展
ops_path = r'D:\files\MODEL\transformer-detection\DINO-main\models\dino\ops'
build_path = os.path.join(ops_path, 'build', 'lib.win-amd64-cpython-312')
sys.path.insert(0, build_path)
sys.path.insert(0, ops_path)

# 1. 导入 .pyd
try:
    import MultiScaleDeformableAttention as MSDA
    print(f"✅ MultiScaleDeformableAttention.pyd 导入成功")
    print(f"   函数: ms_deform_attn_forward, ms_deform_attn_backward")
except ImportError as e:
    print(f"❌ .pyd 导入失败: {e}")

    # 详细诊断
    pyd_file = os.path.join(build_path, 'MultiScaleDeformableAttention.cp312-win_amd64.pyd')
    print(f"   .pyd 存在: {os.path.exists(pyd_file)}")
    if os.path.exists(pyd_file):
        print(f"   文件大小: {os.path.getsize(pyd_file):,} bytes")

    # 列出 torch lib 下的关键 DLL
    import glob
    torch_dlls = glob.glob(os.path.join(torch_lib, '*.dll'))
    print(f"   torch lib DLL 数量: {len(torch_dlls)}")
    for dll_name in ['c10.dll', 'torch_cpu.dll', 'torch_cuda.dll', 'cudart64_110.dll']:
        dll_path = os.path.join(torch_lib, dll_name)
        print(f"   {'✅' if os.path.exists(dll_path) else '❌'} {dll_name}")

    sys.exit(1)

# 使用 DINO 项目的正常包导入路径
sys.path.insert(0, r'D:\files\MODEL\transformer-detection\DINO-main')

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
