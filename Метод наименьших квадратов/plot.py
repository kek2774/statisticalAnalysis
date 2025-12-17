import os
import math
import glob
from typing import List, Tuple, Optional

# ==========================
# НАСТРОЙКИ "без консоли"
# ==========================
SHOW_WINDOWS = True
SAVE_COMBINED_PNG = False
DPI = 170
OUT_SUBDIR = ""  # например "Plots" -> сохранит в Out/Plots; "" -> прямо Out
COMBINED_NAME = "MLS_Combined.png"

# Если в имени есть weibull/вейбул -> считаем, что это Вейбулл и делаем X=lgN
USE_LG_X_FOR_WEIBULL_NAME = True

SHIFT = 5.0  # "+5" как в C++ / методичке

# Тики вероятности как на скриншотах/примере
P_TICKS_NORMAL = [0.01, 0.025, 0.05, 0.10, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99]
P_TICKS_WEIBULL = [
    0.001,
    0.01,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.70,
    0.80,
    0.90,
    0.95,
    0.975,
    0.99,
    0.999,
]


# -----------------------------
# Чтение формата .xout
# -----------------------------
def read_xout(path: str) -> Tuple[List[List[float]], List[List[float]]]:
    with open(path, "r", encoding="utf-8") as f:
        tokens = f.read().strip().split()

    if not tokens:
        raise ValueError(f"Пустой файл: {path}")

    it = iter(tokens)
    try:
        nc = int(next(it))
    except StopIteration:
        raise ValueError("Некорректный .xout: нет числа серий (nc)")

    sizes: List[int] = []
    for _ in range(nc):
        try:
            sizes.append(int(next(it)))
        except StopIteration:
            raise ValueError("Некорректный .xout: не хватает размеров серий")

    Xs: List[List[float]] = []
    for i in range(nc):
        s = sizes[i]
        arr: List[float] = []
        for _ in range(s):
            try:
                arr.append(float(next(it)))
            except StopIteration:
                raise ValueError(f"Некорректный .xout: не хватает X для серии {i}")
        Xs.append(arr)

    Ys: List[List[float]] = []
    for i in range(nc):
        s = sizes[i]
        arr: List[float] = []
        for _ in range(s):
            try:
                arr.append(float(next(it)))
            except StopIteration:
                raise ValueError(f"Некорректный .xout: не хватает Y для серии {i}")
        Ys.append(arr)

    try:
        extra = next(it)
        raise ValueError(
            f"Некорректный .xout: лишние данные после чтения (например: {extra})"
        )
    except StopIteration:
        pass

    return Xs, Ys


def sort_by_y(x: List[float], y: List[float]) -> Tuple[List[float], List[float]]:
    pairs = sorted(zip(y, x))
    ys = [t[0] for t in pairs]
    xs = [t[1] for t in pairs]
    return xs, ys


def detect_is_weibull_from_name(path: str) -> bool:
    name = os.path.basename(path).lower()
    return ("weibull" in name) or ("вейбул" in name)


def pick_two_xouts(out_dir: str) -> Tuple[Optional[str], Optional[str]]:
    all_xouts = sorted(glob.glob(os.path.join(out_dir, "*.xout")))
    if not all_xouts:
        return None, None

    normal = None
    weibull = None
    for p in all_xouts:
        low = os.path.basename(p).lower()
        if normal is None and ("normal" in low or "норм" in low):
            normal = p
        if weibull is None and ("weibull" in low or "вейбул" in low):
            weibull = p

    if normal and weibull:
        return normal, weibull

    if len(all_xouts) >= 2:
        return all_xouts[0], all_xouts[1]

    return all_xouts[0], None


def format_float_ru(val: float, digits: int = 2) -> str:
    s = f"{val:.{digits}f}"
    return s.replace(".", ",")


def format_percent_ru(p: float) -> str:
    val = p * 100.0
    if val < 1.0:
        s = f"{val:.1f}"
    else:
        if abs(val - round(val)) < 1e-12:
            s = f"{int(round(val))}"
        else:
            s = f"{val:.1f}"
    return s.replace(".", ",") + "%"


# --------- inv_norm_cdf (Acklam) ----------
def inv_norm_cdf(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        return num / den

    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        num = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        return num / den

    q = p - 0.5
    r = q * q
    num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    return num / den


def y_from_p_normal(p: float) -> float:
    return SHIFT + inv_norm_cdf(p)


def y_from_p_weibull(p: float) -> float:
    # как в C++: 5 + ln ln(1/(1-p))
    return SHIFT + math.log(math.log(1.0 / (1.0 - p)))


def set_excel_like_grid(ax) -> None:
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="-", alpha=0.55)
    ax.grid(True, which="minor", linestyle="-", alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)


def autosize_and_center_figure(
    fig, width_frac: float = 0.92, height_frac: float = 0.82
) -> None:
    sw = None
    sh = None
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.destroy()
    except Exception:
        sw = None
        sh = None

    if not sw or not sh:
        return

    w = int(sw * width_frac)
    h = int(sh * height_frac)
    x = int((sw - w) / 2)
    y = int((sh - h) / 2)

    try:
        dpi = float(fig.get_dpi())
        fig.set_size_inches(w / dpi, h / dpi, forward=True)
    except Exception:
        pass

    try:
        mgr = fig.canvas.manager
        if hasattr(mgr, "window"):
            win = mgr.window
            if hasattr(win, "setGeometry"):  # Qt
                win.setGeometry(x, y, w, h)
                return
            if hasattr(win, "wm_geometry"):  # Tk
                win.wm_geometry(f"{w}x{h}+{x}+{y}")
                return
            if hasattr(win, "SetSize") and hasattr(win, "SetPosition"):  # Wx
                win.SetSize((w, h))
                win.SetPosition((x, y))
                return
    except Exception:
        pass


def plot_into_axis(ax, xout_path: str) -> None:
    import matplotlib.ticker as mticker

    Xs, Ys = read_xout(xout_path)
    if len(Xs) < 4 or len(Ys) < 4:
        raise ValueError(f"Ожидалось 4 серии в .xout, но найдено: {len(Xs)}")

    # Ожидаемый формат:
    # Xs[0]=X_emp, Xs[1]=X_low, Xs[2]=X_hat, Xs[3]=X_up
    # Ys[0]=Y_emp, Ys[1]=Y_grid (для линий), остальные могут дублировать
    X_emp, X_low, X_hat, X_up = Xs[0], Xs[1], Xs[2], Xs[3]
    Y_emp = Ys[0]
    Y_grid = Ys[1]

    # Сортируем линии по Y
    X_low_s, Y_grid_s = sort_by_y(X_low, Y_grid)
    X_hat_s, _ = sort_by_y(X_hat, Y_grid)
    X_up_s, _ = sort_by_y(X_up, Y_grid)

    is_weibull = USE_LG_X_FOR_WEIBULL_NAME and detect_is_weibull_from_name(xout_path)

    # X: для Вейбулла делаем lgN
    if is_weibull:
        all_x = X_emp + X_low_s + X_hat_s + X_up_s
        if all((math.isfinite(v) and v > 0.0) for v in all_x):
            X_emp = [math.log10(v) for v in X_emp]
            X_low_s = [math.log10(v) for v in X_low_s]
            X_hat_s = [math.log10(v) for v in X_hat_s]
            X_up_s = [math.log10(v) for v in X_up_s]
        ax.set_xlabel("lgx")
        dist_name = "Вейбулл"
    else:
        ax.set_xlabel("x")
        dist_name = "Нормальное"

    # Рисуем
    ax.fill_betweenx(Y_grid_s, X_low_s, X_up_s, alpha=0.22, label="ДИ 95%")
    ax.plot(X_hat_s, Y_grid_s, linewidth=2.0, label="Оценка")
    ax.scatter(X_emp, Y_emp, s=28, marker="o", label="KM (события)")

    set_excel_like_grid(ax)

    # Автолимиты X
    allx = [v for v in (X_emp + X_low_s + X_hat_s + X_up_s) if math.isfinite(v)]
    if allx:
        xmin, xmax = min(allx), max(allx)
        if xmin == xmax:
            xmin -= 1.0
            xmax += 1.0
        else:
            pad = 0.06 * (xmax - xmin)
            xmin -= pad
            xmax += pad
        ax.set_xlim(xmin, xmax)

    ax.set_title(dist_name)

    # Формат осей с запятой
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, pos: format_float_ru(x, 2))
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, pos: format_float_ru(y, 2))
    )

    if is_weibull:
        ax.set_ylabel("ln(-ln(1-p)) + 5")

        # Локаторы (как в примере)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))

        ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

        # Справа: P%
        axr = ax.twinx()
        axr.set_ylim(ax.get_ylim())
        axr.set_ylabel("P%")

        y_ticks = [y_from_p_weibull(p) for p in P_TICKS_WEIBULL]
        axr.set_yticks(y_ticks)
        axr.set_yticklabels([format_percent_ru(p) for p in P_TICKS_WEIBULL])
        axr.grid(False)
    else:
        # Normal: слева сразу P%
        ax.set_ylabel("P%")
        y_ticks = [y_from_p_normal(p) for p in P_TICKS_NORMAL]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([format_percent_ru(p) for p in P_TICKS_NORMAL])

    ax.legend(loc="best")


def main() -> None:
    import matplotlib.pyplot as plt

    root = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(root, "Out")

    if not os.path.isdir(out_dir):
        print(f"[ОШИБКА] Не найдена папка: {out_dir}")
        print(
            "Положи plot.py в каталог над Out/, чтобы было: .../plot.py и .../Out/*.xout"
        )
        return

    left_path, right_path = pick_two_xouts(out_dir)
    if left_path is None:
        print(f"[ОШИБКА] В папке {out_dir} не найдено ни одного *.xout")
        return

    fig, axes = plt.subplots(1, 2, dpi=DPI)

    try:
        plot_into_axis(axes[0], left_path)
    except Exception as e:
        axes[0].axis("off")
        axes[0].text(
            0.02,
            0.5,
            f"Ошибка:\n{os.path.basename(left_path)}\n{e}",
            transform=axes[0].transAxes,
            va="center",
            ha="left",
        )

    if right_path is not None:
        try:
            plot_into_axis(axes[1], right_path)
        except Exception as e:
            axes[1].axis("off")
            axes[1].text(
                0.02,
                0.5,
                f"Ошибка:\n{os.path.basename(right_path)}\n{e}",
                transform=axes[1].transAxes,
                va="center",
                ha="left",
            )
    else:
        axes[1].axis("off")
        axes[1].text(
            0.02,
            0.5,
            "В папке Out найден только один .xout",
            transform=axes[1].transAxes,
            va="center",
            ha="left",
        )

    fig.tight_layout()
    autosize_and_center_figure(fig)

    if SAVE_COMBINED_PNG:
        save_dir = out_dir if not OUT_SUBDIR else os.path.join(out_dir, OUT_SUBDIR)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, COMBINED_NAME)
        fig.savefig(out_path, bbox_inches="tight")
        print(f"[OK] Сохранено: {out_path}")

    if SHOW_WINDOWS:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
