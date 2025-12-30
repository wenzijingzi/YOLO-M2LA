import matplotlib.pyplot as plt

# 数据
categories = ["Pedestrian", "Cyclist", "Car", "Truck", "Van"]
images = [2227, 1982, 7928, 699, 4545]
instances = [4279, 2874, 41425, 763, 6796]

# 颜色（与之前一致）
colors = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2"]

# 创建画布
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# 左图 - Images
wedges1, texts1, autotexts1 = axes[0].pie(
    images,  autopct="%1.1f%%", colors=colors,
    textprops={'fontsize': 16}  # 设置饼图内字体大小
)
axes[0].set_title("Category Distribution by Images", fontsize=18)

# 添加图例（右下角，避免重叠）
axes[0].legend(
    wedges1, categories, title="Categories", loc="lower right",
    bbox_to_anchor=(1.3, -0.05), fontsize=16
)

# 右图 - Instances
wedges2, texts2, autotexts2 = axes[1].pie(
    instances,  autopct="%1.1f%%", colors=colors,
    textprops={'fontsize': 16}
)
axes[1].set_title("Category Distribution by Instances", fontsize=18)

# 添加图例
axes[1].legend(
    wedges2, categories, title="Categories", loc="lower right",
    bbox_to_anchor=(1.3, -0.05), fontsize=16
)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig("E:/track/YOLOv11.3/bfvd_piecharts.png", dpi=300, bbox_inches="tight")
plt.show()

