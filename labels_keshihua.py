import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
from glob import glob
from PIL import Image, ImageDraw

category = ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Van']
num_classes = len(category)

# ✅ 手动指定颜色 (RGB 0-255 转换为 [0,1] 区间)
custom_rgb = [
    (139, 140, 140),    # Pedestrian
    (230, 216, 211),    # Cyclist
    (167, 121, 121),    # Car
    (186, 194, 212),    # Truck
    (156, 158, 137)     # Van
]
colors = [(r/255, g/255, b/255) for r, g, b in custom_rgb]

# ✅ 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

def get_image_size(label_path):
    """获取标注文件对应图像的尺寸"""
    img_path = label_path.replace('\\labels\\', '\\images\\').replace('/labels/', '/images/')
    img_path = img_path.replace('.txt', '.jpg')
    try:
        img = Image.open(img_path)
        return img.width, img.height
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

def plot_labels(labels, names=(), save_dir='', img_width=None, img_height=None):
    if labels.ndim == 1:
        if len(labels) % 5 == 0:
            labels = labels.reshape(-1, 5)
        else:
            raise ValueError("Invalid labels format. Expected length divisible by 5.")

    print('Plotting labels... ')
    c, b = labels[:, 0], labels[:, 1:].transpose()
    nc = int(c.max() + 1)
    x = pd.DataFrame(b.transpose(), columns=['x', 'y', 'width', 'height'])

    # ✅ 自定义颜色字典
    color_dict_matplotlib = {i: colors[i] for i in range(num_classes)}
    color_dict_pil = {i: tuple(int(c * 255) for c in colors[i]) for i in range(num_classes)}

    # seaborn correlogram
    sn.pairplot(x, corner=True, diag_kind='auto', kind='hist',
                diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(os.path.join(save_dir, 'labels_correlogram.jpg'), dpi=200)
    plt.close()

    matplotlib.use('svg')
    fig, ax = plt.subplots(2, 2, figsize=(9, 9), tight_layout=True)

    # 左上：类别分布饼状图
    class_counts = np.bincount(c.astype(int), minlength=nc)
    wedges, texts, autotexts = ax[0, 0].pie(
        class_counts,
        labels=names if 0 < len(names) < 30 else None,
        autopct='%1.1f%%',
        colors=[color_dict_matplotlib.get(i, 'gray') for i in range(nc)],
        textprops={'fontsize': 16, 'fontfamily': 'Times New Roman'}
    )
    ax[0, 0].set_title('Class Distribution', fontsize=16, fontfamily='Times New Roman')
    # ax[0, 0].legend(wedges, names, title="Categories", loc="center left",
    #                 bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12, prop={'family': 'Times New Roman'})

    # 左下：位置分布(x,y)
    sn.histplot(x, x='x', y='y', ax=ax[1, 0], bins=50, pmax=0.9,
                color=(167 / 255, 121 / 255, 121 / 255))
    ax[1, 0].set_xlabel("x", fontsize=16, fontfamily='Times New Roman')
    ax[1, 0].set_ylabel("y", fontsize=16, fontfamily='Times New Roman')
    ax[1, 0].tick_params(axis='both', labelsize=11)

    # 右下：尺寸分布(width,height)
    sn.histplot(x, x='width', y='height', ax=ax[1, 1], bins=50, pmax=0.9,
                color=(167 / 255, 121 / 255, 121 / 255))
    ax[1, 1].set_xlabel("Width", fontsize=16, fontfamily='Times New Roman')
    ax[1, 1].set_ylabel("Height", fontsize=16, fontfamily='Times New Roman')
    ax[1, 1].tick_params(axis='both', labelsize=11)

    # 右上：边界框可视化
    if img_width and img_height:
        converted_labels = labels.copy()
        converted_labels[:, 1:] = xywh2xyxy(converted_labels[:, 1:]) * [img_width, img_height, img_width, img_height]
        img = Image.fromarray(np.ones((img_height, img_width, 3), dtype=np.uint8) * 255)
        for cls, *box in converted_labels[:1000]:
            cls_idx = int(cls)
            draw_color = color_dict_pil.get(cls_idx, (0, 0, 0))
            ImageDraw.Draw(img).rectangle(box, width=1, outline=draw_color)
        ax[0, 1].imshow(img)
    else:
        ax[0, 1].text(0.5, 0.5, "Image dimensions not available",
                      ha='center', va='center', transform=ax[0, 1].transAxes,
                      fontsize=16, fontfamily='Times New Roman')
    ax[0, 1].axis('off')

    # ✅ 统一标题字体
    for a in ax.flat:
        a.title.set_fontsize(16)
        a.title.set_fontfamily("Times New Roman")
        for s in ['top', 'right', 'left', 'bottom']:
            a.spines[s].set_visible(False)

    plt.savefig(os.path.join(save_dir, 'labels.pdf'), dpi=200)
    matplotlib.use('Agg')
    plt.close()

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

# 主程序
if __name__ == "__main__":
    all_files = glob(r'E:/track/dataset/YT/all/labels/val/*.txt')
    shapes, ids, img_sizes = [], [], []

    for file in all_files:
        if file.endswith('classes.txt'):
            continue
        size = get_image_size(file)
        if size:
            img_sizes.append(size)
        with open(file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split()
                if not line:
                    continue
                ids.append(int(line[0]))
                shapes.append(list(map(float, line[1:])))

    ids = np.array(ids)
    shapes = np.array(shapes)
    lbs = np.column_stack((ids, shapes))

    valid_sizes = [s for s in img_sizes if s is not None]
    if valid_sizes:
        avg_width = int(sum(s[0] for s in valid_sizes) / len(valid_sizes))
        avg_height = int(sum(s[1] for s in valid_sizes) / len(valid_sizes))
        print(f"Using average image size: {avg_width}x{avg_height}")
    else:
        avg_width, avg_height = 1920, 1080
        print("No image dimensions found, using default Full HD size (1920x1080)")

    plot_labels(labels=lbs, names=np.array(category),
                save_dir='.', img_width=avg_width, img_height=avg_height)
