"""
simulation.py — Monte Carlo Insight Simulator 엔진
"""
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. 분포 샘플링
# ─────────────────────────────────────────────
def sample_distribution(dist: str, min_val: float, max_val: float, n: int) -> np.ndarray:
    """변수 하나에 대해 n개의 샘플을 생성합니다."""
    dist = dist.lower()
    if dist == "균등":
        return np.random.uniform(min_val, max_val, n)
    elif dist == "정규":
        mean = (min_val + max_val) / 2
        std = (max_val - min_val) / 6  # 99.7% 범위 = ±3σ
        samples = np.random.normal(mean, std, n)
        return np.clip(samples, min_val, max_val)
    elif dist == "삼각":
        mode = (min_val + max_val) / 2
        return np.random.triangular(min_val, mode, max_val, n)
    else:
        raise ValueError(f"알 수 없는 분포: {dist}")


# ─────────────────────────────────────────────
# 2. 시뮬레이션 실행
# ─────────────────────────────────────────────
def run_simulation(variables: list[dict], n_iter: int) -> pd.DataFrame:
    """
    variables: [{'name': str, 'min': float, 'max': float, 'dist': str, 'weight': float}, ...]
    반환: 각 변수 컬럼 + 'result' 컬럼을 가진 DataFrame
    """
    data = {}
    for var in variables:
        data[var["name"]] = sample_distribution(
            var["dist"], var["min"], var["max"], n_iter
        )

    df = pd.DataFrame(data)

    # result = 가중합 (weight 미지정 시 균등 가중)
    weights = np.array([var.get("weight", 1.0) for var in variables], dtype=float)
    weights /= weights.sum()
    values = np.column_stack([df[var["name"]].values for var in variables])
    df["result"] = values @ weights

    return df


# ─────────────────────────────────────────────
# 3. 자동 수렴 감지
# ─────────────────────────────────────────────
def auto_convergence(
    variables: list[dict],
    tol: float = 1e-3,
    min_iter: int = 1_000,
    max_iter: int = 100_000,
    chunk: int = 1_000,
) -> tuple[pd.DataFrame, list[float]]:
    """
    평균값 변화가 tol 미만으로 안정화되면 시뮬레이션을 중단합니다.
    반환: (최종 DataFrame, running_means 리스트)
    """
    all_rows = []
    running_means = []
    prev_mean = None

    n_done = 0
    while n_done < max_iter:
        batch_df = run_simulation(variables, chunk)
        all_rows.append(batch_df)
        n_done += chunk

        cur_mean = pd.concat(all_rows)["result"].mean()
        running_means.append(cur_mean)

        if n_done >= min_iter and prev_mean is not None:
            if abs(cur_mean - prev_mean) / (abs(prev_mean) + 1e-12) < tol:
                break
        prev_mean = cur_mean

    return pd.concat(all_rows, ignore_index=True), running_means


# ─────────────────────────────────────────────
# 4. 신뢰 구간 계산
# ─────────────────────────────────────────────
def calc_confidence_interval(results: np.ndarray) -> dict:
    """5th / 95th percentile 기반 90% CI 반환."""
    return {
        "p5": float(np.percentile(results, 5)),
        "p95": float(np.percentile(results, 95)),
        "mean": float(np.mean(results)),
        "median": float(np.median(results)),
        "std": float(np.std(results)),
    }


# ─────────────────────────────────────────────
# 5. 수렴 추이 (고정 횟수 버전용)
# ─────────────────────────────────────────────
def calc_running_mean(results: np.ndarray, n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """결과 배열에서 running mean 추이를 균등 간격으로 샘플링해 반환합니다."""
    total = len(results)
    indices = np.linspace(100, total, n_points, dtype=int)
    means = [results[:i].mean() for i in indices]
    return indices, np.array(means)


# ─────────────────────────────────────────────
# 6. 민감도 분석 (토네이도 차트용)
# ─────────────────────────────────────────────
def sensitivity_analysis(df: pd.DataFrame, target: str = "result") -> pd.Series:
    """
    각 입력 변수와 결과값 사이의 Pearson 상관계수를 계산합니다.
    절댓값 내림차순으로 정렬된 Series를 반환합니다.
    """
    input_cols = [c for c in df.columns if c != target]
    corr = df[input_cols].corrwith(df[target])
    return corr.reindex(corr.abs().sort_values(ascending=False).index)
