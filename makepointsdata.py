import os
import numpy as np
import pandas as pd
from matplotlib.path import Path
import matplotlib.pyplot as plt

# =========================
# 配置
# =========================
CONFIG = {
    "matrix_boundary_file": r"matrix_boundary.csv",
    "circles_file": r"circles.csv",
    # 采样模式：规则网格
    "sampling": {
        # 网格分辨率：越大点越密
        "nx": 1200,
        "ny": 1200,
    },
    # 输出文件
    "output_points_file": r"points_xy.dat",
    # 是否画一张检查几何 + 采样点的示意图
    "draw_preview": True,
}


# =========================
# 读取基体多边形边界
# =========================
def load_matrix_boundary(path):
    """
    matrix_boundary.csv:
        可为逗号/空格/Tab 分隔的两列：
        x y
    无表头。
    """
    df = pd.read_csv(
        path,
        header=None,
        sep=None,  # 自动识别分隔符
        engine="python",
    )
    if df.shape[1] < 2:
        raise ValueError("matrix_boundary_file 必须至少有 2 列 (x, y).")

    xb = df.iloc[:, 0].to_numpy(dtype=float)
    yb = df.iloc[:, 1].to_numpy(dtype=float)
    return xb, yb


# =========================
# 读取圆：夹杂 & 孔洞
# =========================
def load_circles_with_type(csv_path):
    """
    circles.csv:
        xc, yc, R, type

        type = 1 -> 夹杂（inclusion）
        type = 0 -> 孔洞（hole）
    """
    data = np.loadtxt(csv_path, delimiter=",")
    data = np.atleast_2d(data)

    if data.shape[1] != 4:
        raise ValueError("circles_file 必须是 4 列: xc, yc, R, type")

    xc = data[:, 0]
    yc = data[:, 1]
    R = data[:, 2]
    ctype = data[:, 3]

    inclusions = []
    holes = []

    for i in range(len(ctype)):
        t = int(ctype[i])
        if t == 1:
            inclusions.append((xc[i], yc[i], R[i]))
        elif t == 0:
            holes.append((xc[i], yc[i], R[i]))
        else:
            # 其他类型先忽略
            pass

    return inclusions, holes


# =========================
# 生成采样点 xy + 区域标记
# =========================
def generate_sampling_points(cfg):
    xb, yb = load_matrix_boundary(cfg["matrix_boundary_file"])
    inc_circles, hole_circles = load_circles_with_type(cfg["circles_file"])

    # ---- 基体多边形 Path ----
    poly = Path(np.column_stack([xb, yb]))

    # ---- 规则网格包住所有基体边界 ----
    nx = cfg["sampling"]["nx"]
    ny = cfg["sampling"]["ny"]

    xmin, xmax = np.min(xb), np.max(xb)
    ymin, ymax = np.min(yb), np.max(yb)

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(xs, ys)
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    # ---- 先选出在基体多边形内的点 ----
    in_matrix_poly = poly.contains_points(pts)

    # ---- 标记孔洞：在任一孔洞圆内的点全部剔除 ----
    in_any_hole = np.zeros(len(pts), dtype=bool)
    for xc, yc, R in hole_circles:
        in_any_hole |= (pts[:, 0] - xc) ** 2 + (pts[:, 1] - yc) ** 2 <= R**2

    valid = in_matrix_poly & (~in_any_hole)

    pts_valid = pts[valid]

    # ---- 区域标记：0 = 基体, 1 = 夹杂 ----
    region = np.zeros(len(pts_valid), dtype=int)
    if inc_circles:
        in_any_inclusion = np.zeros(len(pts_valid), dtype=bool)
        for xc, yc, R in inc_circles:
            in_any_inclusion |= (pts_valid[:, 0] - xc) ** 2 + (
                pts_valid[:, 1] - yc
            ) ** 2 <= R**2
        region[in_any_inclusion] = 1

    # ---- 保存到 CSV ----
    df_out = pd.DataFrame(
        {
            "x": pts_valid[:, 0],
            "y": pts_valid[:, 1],
        }
    )
    out_path = cfg["output_points_file"]
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(out_path, index=False, header=False, sep=",")

    print(
        f"Total number of points: {len(df_out)}, Number of inclusion points: {np.sum(region == 1)}"
    )
    print(f"✔ Sampling points have been saved to: {out_path}")

    return pts_valid, region, (xb, yb), inc_circles, hole_circles


# =========================
# 可选：画一张检查图
# =========================
def preview_points(pts_valid, region, xb, yb, inc_circles, hole_circles):
    fig, ax = plt.subplots(figsize=(6, 4))

    # 基体边界
    ax.plot(xb, yb, "k-", lw=1.0, label="Matrix boundary")

    # 夹杂圆
    for xc, yc, R in inc_circles:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(xc + R * np.cos(theta), yc + R * np.sin(theta), "g-", lw=1.0)

    # 孔洞圆
    for xc, yc, R in hole_circles:
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(xc + R * np.cos(theta), yc + R * np.sin(theta), "r--", lw=1.0)

    # 采样点：基体 & 夹杂
    mask_mat = region == 0
    mask_inc = region == 1
    ax.scatter(
        pts_valid[mask_mat, 0],
        pts_valid[mask_mat, 1],
        s=1,
        c="C0",
        alpha=0.4,
        label="Matrix points",
    )
    ax.scatter(
        pts_valid[mask_inc, 0],
        pts_valid[mask_inc, 1],
        s=1,
        c="C1",
        alpha=0.6,
        label="Inclusion points",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Sampling points (matrix + inclusions, holes removed)")
    plt.tight_layout()
    plt.show()


# =========================
# main
# =========================
if __name__ == "__main__":
    pts_valid, region, (xb, yb), inc_circles, hole_circles = generate_sampling_points(
        CONFIG
    )

    if CONFIG.get("draw_preview", False):
        preview_points(pts_valid, region, xb, yb, inc_circles, hole_circles)
