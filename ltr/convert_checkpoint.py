import torch

# 你的模型 checkpoint 路径
src = "checkpoint.pth.tar"
dst = "checkpoint.pth"

ckpt = torch.load(src, map_location="cpu")

# 提取实际的权重
state_dict = ckpt["net"]

# 保存成标准 .pth
torch.save(state_dict, dst)

print("转换完成！已输出：", dst)

