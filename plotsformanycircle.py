import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.path import Path
from scipy.interpolate import griddata
from collections import Counter
import json
import argparse

# =========================
# 全局配置
# =========================
CONFIG_DEFAULT = {
    # 场数据文件（含 x-coord / y-coord / 各物理量列）
    "field_file": r"output_file.dat",
    # 基体外边界（多边形），两列 x y，逗号 / 空格 / Tab 分隔均可
    "matrix_boundary_file": r"matrix_boundary.csv",
    # 圆（夹杂 + 孔洞），格式：xc,yc,R,type ；type=0: 孔洞, type=1: 夹杂
    "circles_file": r"circles.csv",
    # 要画的物理量列表
    "plots": [
        # 示例：只给 vmin/vmax（自动均匀分级）
        {
            "column": "vonMises",
            "vmin": 0.5,
            "vmax": 770.2,
            # 如需自定义色块边界（必须单调，长度 = 颜色数+1），可加：
            # "cbar_levels": [v0, v1, ..., vN],
        },
        # 可以继续添加其它物理量：
        # {
        #     "column": "p1",
        #     "cbar_levels": [2.010, 26.95, ..., 301.3],
        # },
    ],
    # 规则网格插值分辨率（越大越细腻，代价是更慢）
    "grid": {
        "nx": 2000,
        "ny": 2000,
        "method": "cubic",  # "cubic" / "linear" / "nearest"
    },
    # 颜色列表（你的经典色带）
    "colors": [
        "#0000ff",
        "#005dff",
        "#00b9ff",
        "#00ffe8",
        "#00ff8b",
        "#00ff2e",
        "#2eff00",
        "#8bff00",
        "#e8ff00",
        "#ffb900",
        "#ff5d00",
        "#ff0000",
    ],
    # 图像尺寸与导出
    "figure": {
        "width_cm": 8,
        "height_cm": 6,
        "dpi": 600,
        "jpg_pad_mm": 1,
        "svg_pad_mm": 0,
    },
    # 字体
    "font": {
        "use_windows_tnr": True,
        "base_size_pt": 9,
    },
    # 色条刻度：True 用边界 levels，False 用区间中点
    "colorbar_tick_on_edges": True,
    # 清洗日志（打印多少个典型坏 token）
    "log_invalid_samples": 5,
}


# ---------- 工具：递归合并 dict（子字典也能覆盖） ----------
def deep_update(base, updates):
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_update(base[k], v)
        else:
            base[k] = v
    return base


# ---------- 从 JSON 读取配置，只要求写到 grid 为止 ----------
def load_config():
    parser = argparse.ArgumentParser(description="Plot HT-FEM fields for many circles.")
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()

    # 先拷贝一份默认配置
    cfg = json.loads(json.dumps(CONFIG_DEFAULT))

    cfg_path = os.path.abspath(args.config)
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        # 用用户的配置覆盖（只写 field_file / plots / grid 也没问题）
        cfg = deep_update(cfg, user_cfg)
    else:
        print(f"⚠ config.json 未找到：{cfg_path}，使用内置默认配置")

    # 处理相对路径（相对于 config.json 所在目录）
    base_dir = os.path.dirname(cfg_path)
    for key in ["field_file", "matrix_boundary_file", "circles_file"]:
        val = cfg.get(key)
        if isinstance(val, str) and val != "" and not os.path.isabs(val):
            cfg[key] = os.path.join(base_dir, val)

    return cfg


# =========================
# 字体：Times New Roman
# =========================
def use_times_new_roman_from_windows(base_size_pt=7, enable=True):
    if not enable:
        mpl.rcParams.update(
            {
                "font.size": base_size_pt,
                "axes.titlesize": base_size_pt,
                "axes.labelsize": base_size_pt,
                "xtick.labelsize": base_size_pt,
                "ytick.labelsize": base_size_pt,
                "legend.fontsize": base_size_pt,
                "figure.titlesize": base_size_pt,
                "svg.fonttype": "none",
                "pdf.fonttype": 42,
            }
        )
        return font_manager.FontProperties(size=base_size_pt)

    win_fonts_dir = "/mnt/c/Windows/Fonts"
    ttf_candidates = [
        "times.ttf",
        "timesbd.ttf",
        "timesi.ttf",
        "timesbi.ttf",
        "TIMES.TTF",
        "TIMESBD.TTF",
        "TIMESI.TTF",
        "TIMESBI.TTF",
        "Times New Roman.ttf",
        "Times New Roman Bold.ttf",
        "Times New Roman Italic.ttf",
        "Times New Roman Bold Italic.ttf",
    ]

    loaded = []
    for name in ttf_candidates:
        path = os.path.join(win_fonts_dir, name)
        if os.path.exists(path):
            try:
                font_manager.fontManager.addfont(path)
                loaded.append(path)
            except Exception:
                pass

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": base_size_pt,
            "axes.titlesize": base_size_pt,
            "axes.labelsize": base_size_pt,
            "xtick.labelsize": base_size_pt,
            "ytick.labelsize": base_size_pt,
            "legend.fontsize": base_size_pt,
            "figure.titlesize": base_size_pt,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    if loaded:
        return font_manager.FontProperties(fname=loaded[0], size=base_size_pt)
    else:
        return font_manager.FontProperties(family="Times New Roman", size=base_size_pt)


FP_TNR = use_times_new_roman_from_windows(
    base_size_pt=CONFIG_DEFAULT["font"]["base_size_pt"],
    enable=CONFIG_DEFAULT["font"]["use_windows_tnr"],
)


# =========================
# 工具：字符串转 float
# =========================
def _coerce_float(val):
    if isinstance(val, str):
        s = val.strip()
        if s == "" or s.lower() in {"nan", "+nan", "-nan"}:
            return np.nan
        if s.lower() in {"inf", "+inf"}:
            return np.inf
        if s.lower() == "-inf":
            return -np.inf
        # Fortran D 指数
        s = s.replace("D", "E").replace("d", "E")
        try:
            return float(s)
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan


def _log_cleaning_stats(df_before, df_after, colname, cfg):
    n0 = len(df_before)
    n1 = len(df_after)
    dropped = n0 - n1
    pct = (dropped / n0 * 100.0) if n0 > 0 else 0.0
    print(
        f"   - Loaded rows: {n0}, after cleaning: {n1}, "
        f"dropped: {dropped} ({pct:.4f}%) for column '{colname}'"
    )
    invalid_tokens = []
    for c in ["x-coord", "y-coord", colname]:
        if c in df_before.columns:
            raw_col = df_before[c].astype(str)
            co = raw_col.map(_coerce_float)
            invalid_tokens.extend(
                list(raw_col[co.isna()].head(cfg["log_invalid_samples"]))
            )
    if invalid_tokens:
        counts = Counter(invalid_tokens)
        topN = counts.most_common(cfg["log_invalid_samples"])
        print(f"   - Sample invalid tokens: {topN}")


# =========================
# 读取基体多边形边界
# =========================
def load_matrix_boundary(path):
    """
    matrix_boundary.csv / .dat:
        - 逗号分隔:  x,y
        - 或空格/Tab :  x   y
    """
    df = pd.read_csv(
        path,
        header=None,  # 无表头
        sep=None,  # 自动识别分隔符
        engine="python",
    )

    if df.shape[1] < 2:
        raise ValueError("matrix_boundary_file must have at least 2 columns (x, y).")

    xb = df.iloc[:, 0].to_numpy(dtype=float)
    yb = df.iloc[:, 1].to_numpy(dtype=float)
    return xb, yb


# =========================
# 读取圆（夹杂 + 孔洞）
# =========================
def load_circles(csv_path):
    """
    circles.csv:
        每行: xc,yc,R,type
        其中 type = 0: 孔洞, type = 1: 夹杂
    """
    data = np.loadtxt(csv_path, delimiter=",")
    data = np.atleast_2d(data)

    if data.shape[1] < 3:
        raise ValueError("circles.csv must have at least 3 columns: xc,yc,R[,type]")

    xc = data[:, 0]
    yc = data[:, 1]
    R = data[:, 2]

    if data.shape[1] >= 4:
        types = data[:, 3]
    else:
        # 没给 type 时，统一当作夹杂（不挖孔）
        types = np.ones_like(R)

    return xc, yc, R, types


# =========================
# 一次性读取 & 清洗所有物理量
# =========================
def load_and_prepare_all(cfg: dict):
    file_path = cfg["field_file"]
    plot_cols = [item["column"] for item in cfg["plots"]]

    data_raw = pd.read_csv(
        file_path,
        sep=r"\s+",
        engine="python",
        dtype=str,
        comment="#",
        keep_default_na=False,
        on_bad_lines="skip",
    )

    for req in ("x-coord", "y-coord"):
        if req not in data_raw.columns:
            raise ValueError(f"Required column '{req}' not found in {file_path}.")
    for col in plot_cols:
        if col not in data_raw.columns:
            raise ValueError(f"Column '{col}' not found in {file_path}.")

    df0 = data_raw.copy()

    # x / y 一次性转 float
    data_raw["x-coord"] = data_raw["x-coord"].map(_coerce_float)
    data_raw["y-coord"] = data_raw["y-coord"].map(_coerce_float)

    # 所有待画列统一转 float
    for col in plot_cols:
        data_raw[col] = data_raw[col].map(_coerce_float)

    # inf -> NaN
    data_num = data_raw.replace([np.inf, -np.inf], np.nan)

    cleaned = {}

    for col in plot_cols:
        df_before = df0[["x-coord", "y-coord", col]].copy()
        df_col = data_num[["x-coord", "y-coord", col]].copy()

        data_clean = df_col.dropna(subset=["x-coord", "y-coord", col])
        _log_cleaning_stats(df_before, data_clean, col, cfg)

        if len(data_clean) < 3:
            raise ValueError(
                f"Not enough valid points for '{col}': got {len(data_clean)} after cleaning."
            )

        x_clean = data_clean["x-coord"].to_numpy(float)
        y_clean = data_clean["y-coord"].to_numpy(float)
        z_clean = data_clean[col].to_numpy(float)

        col_min = float(np.nanmin(z_clean))
        col_max = float(np.nanmax(z_clean))

        cleaned[col] = {
            "x": x_clean,
            "y": y_clean,
            "z": z_clean,
            "min": col_min,
            "max": col_max,
        }

    return cleaned


# =========================
# 主绘图：规则网格 + 几何掩膜 + 孔洞白圆
# =========================
def plot_contour_geo_grid(
    x_clean,
    y_clean,
    z_clean,
    column_name,
    xb,
    yb,
    circles,  # (xc, yc, R, type)
    data_min=None,
    data_max=None,
    custom_min_value=None,
    custom_max_value=None,
    grid_nx=1000,
    grid_ny=1000,
    grid_method="cubic",
    colors=None,
    fig_width_cm=8,
    fig_height_cm=6,
    dpi=600,
    jpg_pad_mm=1,
    svg_pad_mm=0,
    colorbar_tick_on_edges=True,
    cbar_levels=None,
):
    # ------------------------------------------------
    # 0. 颜色 & 色阶
    # ------------------------------------------------
    if colors is None:
        colors = CONFIG_DEFAULT["colors"]
    cmap = mcolors.ListedColormap(colors)

    # 色条等级
    if cbar_levels is not None and len(cbar_levels) >= 2:
        levels = np.sort(np.array(cbar_levels, dtype=float))
        min_value = float(levels[0])
        max_value = float(levels[-1])
    else:
        if custom_min_value is not None and custom_max_value is not None:
            min_value = float(custom_min_value)
            max_value = float(custom_max_value)
        elif (data_min is not None) and (data_max is not None):
            min_value = float(data_min)
            max_value = float(data_max)
        else:
            min_value = float(np.nanmin(z_clean))
            max_value = float(np.nanmax(z_clean))

        levels = np.linspace(min_value, max_value, len(colors) + 1)

    # ------------------------------------------------
    # 1. 基体多边形 Path
    # ------------------------------------------------
    poly = Path(np.column_stack([xb, yb]))

    # 所有计算点（不再先裁掉边界）
    x_all = x_clean
    y_all = y_clean
    z_all = z_clean

    if len(x_all) < 10:
        raise ValueError("Too few data points for plotting.")

    # 计算点所在几何区域（点级别）
    pts = np.column_stack([x_all, y_all])
    inside_matrix_pts = poly.contains_points(pts)

    xc_arr, yc_arr, R_arr, type_arr = circles

    inside_hole_pts = np.zeros_like(x_all, dtype=bool)
    inside_inclusion_pts = np.zeros_like(x_all, dtype=bool)

    for xc, yc, R, t in zip(xc_arr, yc_arr, R_arr, type_arr):
        mask = (x_all - xc) ** 2 + (y_all - yc) ** 2 <= R**2
        if int(t) == 0:
            inside_hole_pts |= mask
        elif int(t) == 1:
            inside_inclusion_pts |= mask

    # 矩阵采样点 = 在基体内 & 不在孔洞 & 不在夹杂
    matrix_pts_mask = inside_matrix_pts & (~inside_hole_pts) & (~inside_inclusion_pts)
    x_mat = x_all[matrix_pts_mask]
    y_mat = y_all[matrix_pts_mask]
    z_mat = z_all[matrix_pts_mask]

    # 夹杂采样点 = 在基体内 & 在夹杂内（type=1）
    inclusion_pts_mask = inside_matrix_pts & inside_inclusion_pts
    x_inc = x_all[inclusion_pts_mask]
    y_inc = y_all[inclusion_pts_mask]
    z_inc = z_all[inclusion_pts_mask]

    if len(x_mat) < 3:
        raise ValueError("Too few matrix points after geometric classification.")
    # 夹杂点可能为空（特别是没有夹杂的算例），后面单独判断

    # ------------------------------------------------
    # 2. 网格生成 & 几何掩膜（网格级别）
    # ------------------------------------------------
    xmin, xmax = xb.min(), xb.max()
    ymin, ymax = yb.min(), yb.max()

    grid_x, grid_y = np.meshgrid(
        np.linspace(xmin, xmax, grid_nx),
        np.linspace(ymin, ymax, grid_ny),
    )

    # 基体内网格
    inside_matrix = poly.contains_points(
        np.column_stack([grid_x.ravel(), grid_y.ravel()])
    ).reshape(grid_x.shape)

    # 孔洞 & 夹杂 区域网格掩膜
    inside_hole = np.zeros_like(grid_x, dtype=bool)
    inside_inclusion = np.zeros_like(grid_x, dtype=bool)

    for xc, yc, R, t in zip(xc_arr, yc_arr, R_arr, type_arr):
        mask_circle = (grid_x - xc) ** 2 + (grid_y - yc) ** 2 <= R**2
        if int(t) == 0:
            inside_hole |= mask_circle
        elif int(t) == 1:
            inside_inclusion |= mask_circle

    # 矩阵网格点：在基体内 & 不在孔洞 & 不在夹杂
    matrix_grid_mask = inside_matrix & (~inside_hole) & (~inside_inclusion)
    # 夹杂网格点：在基体内 & 在夹杂
    inclusion_grid_mask = inside_matrix & inside_inclusion
    # 整个有材料区域（矩阵+夹杂）
    domain_mask = inside_matrix & (~inside_hole)

    # ------------------------------------------------
    # 3. 矩阵相插值：平滑 + 最近邻兜底
    # ------------------------------------------------
    grid_z_mat_smooth = griddata(
        (x_mat, y_mat),
        z_mat,
        (grid_x, grid_y),
        method=grid_method,
    )
    grid_z_mat_nn = griddata(
        (x_mat, y_mat),
        z_mat,
        (grid_x, grid_y),
        method="nearest",
    )
    grid_z_mat = np.where(np.isnan(grid_z_mat_smooth), grid_z_mat_nn, grid_z_mat_smooth)

    # ------------------------------------------------
    # 4. 夹杂相插值：只用最近邻，把内部场“延伸”到边界
    # ------------------------------------------------
    if len(x_inc) >= 3:
        # 先用线性插值，让夹杂内部场分布更顺滑
        grid_z_inc_smooth = griddata(
            (x_inc, y_inc),
            z_inc,
            (grid_x, grid_y),
            method="linear",  # ★ 线性插值
        )
        # 再用最近邻给 NaN 打补丁（比如凸包外的一小圈）
        grid_z_inc_nn = griddata(
            (x_inc, y_inc),
            z_inc,
            (grid_x, grid_y),
            method="nearest",
        )
        grid_z_inc = np.where(
            np.isnan(grid_z_inc_smooth),
            grid_z_inc_nn,
            grid_z_inc_smooth,
        )
    else:
        # 没有夹杂点：理论上不会用到 inclusion_grid_mask，安全兜底
        grid_z_inc = grid_z_mat.copy()

    # ------------------------------------------------
    # 5. 合并矩阵 + 夹杂，孔洞挖空
    # ------------------------------------------------
    grid_z = np.full_like(grid_x, np.nan, dtype=float)

    # 矩阵区域用矩阵场
    grid_z[matrix_grid_mask] = grid_z_mat[matrix_grid_mask]
    # 夹杂区域用夹杂场（最近邻延伸到边界）
    grid_z[inclusion_grid_mask] = grid_z_inc[inclusion_grid_mask]

    # 限制数值范围
    grid_z = np.clip(grid_z, min_value, max_value)
    # 域外（基体外 + 孔洞内）全部 mask
    grid_z = np.ma.masked_where(~domain_mask, grid_z)

    # ------------------------------------------------
    # 6. 画图 + 色条
    # ------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(fig_width_cm / 2.54, fig_height_cm / 2.54),
        dpi=dpi,
    )

    contour = ax.contourf(
        grid_x, grid_y, grid_z, levels=levels, cmap=cmap, antialiased=False
    )

    # 色条：科学计数法 + 三位有效数字
    cbar = fig.colorbar(contour, ax=ax, label=None, shrink=0.6)
    if colorbar_tick_on_edges:
        tick_pos = levels
    else:
        tick_pos = (levels[:-1] + levels[1:]) / 2.0

    tick_labels = [f"{v:.3e}" for v in tick_pos]
    cbar.set_ticks(tick_pos)
    cbar.outline.set_linewidth(0.3)
    cbar.ax.tick_params(width=0.3, labelsize=CONFIG_DEFAULT["font"]["base_size_pt"])
    cbar.ax.set_yticklabels(tick_labels, fontproperties=FP_TNR)

    # ------------------------------------------------
    # 7. 孔洞画成纯白圆（再次保证是完美圆）
    # ------------------------------------------------
    for xc, yc, R, t in zip(xc_arr, yc_arr, R_arr, type_arr):
        if int(t) == 0:
            circ = plt.Circle(
                (xc, yc),
                R,
                facecolor="white",
                edgecolor="none",
                zorder=50,
            )
            ax.add_patch(circ)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout()

    out_jpg = f"{column_name.split('.')[0]}.jpg"
    out_svg = out_jpg.replace(".jpg", ".svg")
    out_pdf = out_jpg.replace(".jpg", ".pdf")

    fig.savefig(
        out_jpg,
        format="jpg",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=jpg_pad_mm / 25.4,
    )
    fig.savefig(
        out_svg,
        format="svg",
        bbox_inches="tight",
        pad_inches=svg_pad_mm / 25.4,
    )
    fig.savefig(
        out_pdf,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"♥️ Saved: {out_jpg}, {out_svg}, {out_pdf}")


# =========================
# 批量入口
# =========================
def run_from_config(cfg: dict, cleaned: dict):
    g = cfg["grid"]
    f = cfg["figure"]

    # 读取几何：基体边界 + 圆
    xb, yb = load_matrix_boundary(cfg["matrix_boundary_file"])
    xc, yc, R, types = load_circles(cfg["circles_file"])
    circles = (xc, yc, R, types)

    for item in cfg["plots"]:
        column = item["column"]
        vmin = item.get("vmin", None)
        vmax = item.get("vmax", None)
        cbar_levels = item.get("cbar_levels", None)

        if column not in cleaned:
            raise ValueError(f"Column '{column}' not found in cleaned data cache.")

        col_data = cleaned[column]
        print(f"==> Drawing [{column}] ...")

        plot_contour_geo_grid(
            x_clean=col_data["x"],
            y_clean=col_data["y"],
            z_clean=col_data["z"],
            column_name=column,
            xb=xb,
            yb=yb,
            circles=circles,
            data_min=col_data["min"],
            data_max=col_data["max"],
            custom_min_value=vmin,
            custom_max_value=vmax,
            grid_nx=g["nx"],
            grid_ny=g["ny"],
            grid_method=g["method"],
            colors=cfg["colors"],
            fig_width_cm=f["width_cm"],
            fig_height_cm=f["height_cm"],
            dpi=f["dpi"],
            jpg_pad_mm=f["jpg_pad_mm"],
            svg_pad_mm=f["svg_pad_mm"],
            colorbar_tick_on_edges=cfg["colorbar_tick_on_edges"],
            cbar_levels=cbar_levels,
        )


# =========================
# main
# =========================
if __name__ == "__main__":
    cfg = load_config()
    cleaned_data = load_and_prepare_all(cfg)  # 只清洗一次
    run_from_config(cfg, cleaned_data)
