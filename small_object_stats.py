import os
import glob

# ==== 参数 ====
VAL_DIR = r"E:\track\dataset\YT\all\labels\val"   # 改成你的 val 目录
IMAGE_W, IMAGE_H = 1920, 1080
THRESH = 32 * 32            # 面积阈值

def parse_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        _, cx, cy, w, h = map(float, parts[:5])
        return w, h
    except:
        return None

def main():
    txt_files = glob.glob(os.path.join(VAL_DIR, "*.txt"))
    total, small = 0, 0

    for f in txt_files:
        with open(f, "r", encoding="utf-8") as rf:
            for line in rf:
                rec = parse_line(line)
                if rec is None:
                    continue
                w_norm, h_norm = rec
                w_px = w_norm * IMAGE_W
                h_px = h_norm * IMAGE_H
                area = w_px * h_px
                total += 1
                if area < THRESH:
                    small += 1

    ratio = small / total if total > 0 else 0
    print(f"{ratio:.6f}")

if __name__ == "__main__":
    main()
