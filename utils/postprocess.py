# utils/postprocess.py
import numpy as np

def apply_scurve_adjustment(pred_cfu, midpoint=6.0, steepness=1.0):
    """
    예측된 CFU를 log10 스케일로 변환 후, S-curve 보정
    midpoint: log10 기준값 (ex. 6 → 1e6 CFU 근처에서 보정 중심)
    steepness: 곡선 기울기
    """
    pred_cfu = max(pred_cfu, 1.0)  # log10 안정성
    log_val = np.log10(pred_cfu)

    # Sigmoid 보정 (0~1 사이 factor)
    factor = 1 / (1 + np.exp(-steepness * (log_val - midpoint)))

    # log-scale에서 보정 적용
    adjusted_log = log_val * (0.5 + factor)  # 0.5~1.5배 스케일
    adjusted = 10**adjusted_log
    return adjusted
