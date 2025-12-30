# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
# import cv2
# import os
# import numpy as np
#
# def save_results_no_conf(results, save_dir, names):
#     os.makedirs(save_dir, exist_ok=True)
#
#     # 为每个类别生成固定颜色 (BGR 格式，范围 0-255)
#     num_classes = len(names)
#     rng = np.random.default_rng(42)  # 固定种子，保证颜色一致
#     colors = rng.integers(0, 255, size=(num_classes, 3))
#
#     for r in results:
#         im = r.orig_img.copy()
#         boxes = r.boxes.xyxy.cpu().numpy()   # [x1,y1,x2,y2]
#         cls_ids = r.boxes.cls.cpu().numpy().astype(int)
#
#         for box, cls_id in zip(boxes, cls_ids):
#             x1, y1, x2, y2 = map(int, box)
#             label = names[cls_id]   # 类别名
#             color = tuple(map(int, colors[cls_id]))  # 取对应颜色
#
#             cv2.rectangle(im, (x1, y1), (x2, y2), color, 3)
#             cv2.putText(im, label, (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#
#         # 保存文件
#         filename = os.path.basename(r.path)
#         save_path = os.path.join(save_dir, filename)
#         cv2.imwrite(save_path, im)
#
#
#
#
# if __name__ == '__main__':
#     model = YOLO('E:/track/YOLOv11/runs/train/exp61/weights/best.pt')
#
#     results = model.predict(
#         source='E:/Dsesktop/figure/fig/detect_fig/.',
#         imgsz=640,
#         save=False   # 不用 Ultralytics 默认保存
#     )
#
#     # 自定义保存（每类不同颜色，只显示类别名）
#     save_dir = "runs/detect/exp14"
#     save_results_no_conf(results, save_dir, model.names)
#
#     print(f"✅ 结果已保存到 {save_dir}，每个类别使用不同颜色，只显示类别名。")


import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import cv2
import os
import numpy as np

# ========== 自定义圆角矩形函数 ==========
def draw_rounded_rectangle(img, pt1, pt2, color, thickness=2, r=8):
    """
    绘制圆角矩形
    pt1: 左上角坐标 (x1, y1)
    pt2: 右下角坐标 (x2, y2)
    r: 圆角半径
    thickness: 线条粗细
    color: BGR颜色 (0-255)
    """
    x1, y1 = pt1
    x2, y2 = pt2

    # 创建一个副本以避免边缘过度叠加
    overlay = img.copy()

    # 绘制直线部分
    cv2.line(overlay, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(overlay, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(overlay, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(overlay, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # 绘制四个圆角
    cv2.ellipse(overlay, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    # 合并图层（避免出现绘制痕迹）
    cv2.addWeighted(overlay, 1.0, img, 0, 0, img)

# ========== 主检测保存函数 ==========
def save_results_no_conf(results, save_dir, names, box_thickness=3, corner_radius=8):
    """
    仅显示类别名，使用圆角矩形框绘制检测结果
    :param results: YOLO 模型预测结果
    :param save_dir: 保存目录
    :param names: 类别名称字典
    :param box_thickness: 边框粗细
    :param corner_radius: 圆角半径
    """
    os.makedirs(save_dir, exist_ok=True)

    # 为每个类别生成固定颜色 (BGR 格式，范围 0-255)
    num_classes = len(names)
    rng = np.random.default_rng(42)  # 固定随机种子
    colors = rng.integers(0, 255, size=(num_classes, 3))

    for r in results:
        im = r.orig_img.copy()
        boxes = r.boxes.xyxy.cpu().numpy()   # [x1,y1,x2,y2]
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)

        for box, cls_id in zip(boxes, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            label = names[cls_id]   # 类别名
            color = tuple(map(int, colors[cls_id]))  # 对应颜色

            # 画圆角矩形框
            draw_rounded_rectangle(im, (x1, y1), (x2, y2),
                                   color=color, thickness=box_thickness, r=corner_radius)

            # 在框上方写类别文字
            cv2.putText(im, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # 保存结果图片
        filename = os.path.basename(r.path)
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, im)

    print(f"✅ 检测结果已保存到 {save_dir}，使用圆角矩形框，每个类别不同颜色。")

# ========== 主程序 ==========
if __name__ == '__main__':
    model = YOLO('E:/track/YOLOv11/runs/train/exp65/weights/best.pt')

    results = model.predict(
        source='E:/Dsesktop/figure/bus_vs_car_view/检测/.',
        imgsz=640,
        save=False   # 不使用 Ultralytics 默认保存
    )

    save_dir = "runs/detect/exp18"
    save_results_no_conf(results, save_dir, model.names,
                         box_thickness=3, corner_radius=10)

