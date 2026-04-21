#!/usr/bin/env python3
"""
加权最小均方误差（WMMSE）预编码算法实现

参考：Shi, Q., et al. "An iteratively weighted MMSE approach to distributed sum-utility 
maximization for a MIMO interfering broadcast channel." IEEE Transactions on Signal Processing, 2011.
"""

import numpy as np
from typing import Tuple, Optional


def generate_channel(num_users: int, num_antennas: int, seed: Optional[int] = None) -> np.ndarray:
    """
    生成瑞利衰落信道矩阵
    
    Args:
        num_users: 用户数 K
        num_antennas: 基站天线数 N_t
        seed: 随机种子
        
    Returns:
        H: 信道矩阵 (K x N_t)，每个元素服从CN(0,1)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 复高斯信道：实部和虚部独立同分布
    real_part = np.random.randn(num_users, num_antennas) / np.sqrt(2)
    imag_part = np.random.randn(num_users, num_antennas) / np.sqrt(2)
    H = real_part + 1j * imag_part
    
    return H


def wmmse_precoding(
    H: np.ndarray,
    P_max: float,
    weights: Optional[np.ndarray] = None,
    noise_power: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-6,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    WMMSE预编码算法
    
    Args:
        H: 信道矩阵 (K x N_t)
        P_max: 最大发射功率
        weights: 用户权重 (K,)，默认等权重
        noise_power: 噪声功率
        max_iter: 最大迭代次数
        tol: 收敛容差
        verbose: 是否打印迭代信息
        
    Returns:
        V: 预编码矩阵 (N_t x K)
        U: 接收滤波器 (K x 1)
        obj_history: 目标函数值历史
    """
    K, N_t = H.shape
    
    # 默认权重
    if weights is None:
        weights = np.ones(K)
    
    # 初始化预编码矩阵（匹配滤波器）
    V = np.sqrt(P_max / K) * np.random.randn(N_t, K) + 1j * np.random.randn(N_t, K)
    V = V / np.linalg.norm(V, 'fro') * np.sqrt(P_max)
    
    # 初始化接收滤波器
    U = np.zeros(K, dtype=complex)
    
    # 初始化权重矩阵（对角矩阵）
    W = np.diag(weights)
    
    obj_history = []
    
    for iter in range(max_iter):
        # 步骤1: 固定V,W，更新U（MMSE接收机）
        for k in range(K):
            # 计算干扰加噪声协方差矩阵
            R_k = noise_power * np.eye(N_t)
            for j in range(K):
                if j != k:
                    h_k = H[k, :].reshape(-1, 1)  # (N_t, 1)
                    v_j = V[:, j].reshape(-1, 1)  # (N_t, 1)
                    R_k += (h_k @ h_k.conj().T) * (v_j @ v_j.conj().T)
            
            h_k = H[k, :].reshape(-1, 1)
            v_k = V[:, k].reshape(-1, 1)
            
            # MMSE接收机
            u_k = np.linalg.inv(R_k) @ h_k @ v_k
            u_k = u_k / (1 + v_k.conj().T @ h_k.conj().T @ np.linalg.inv(R_k) @ h_k @ v_k)
            U[k] = u_k.item()
        
        # 步骤2: 固定U,W，更新V
        # 构建矩阵 A 和 B
        A = np.zeros((N_t, N_t), dtype=complex)
        B = np.zeros((N_t, K), dtype=complex)
        
        for k in range(K):
            h_k = H[k, :].reshape(-1, 1)  # (N_t, 1)
            w_k = weights[k]
            u_k = U[k]
            
            A += w_k * np.abs(u_k)**2 * (h_k @ h_k.conj().T)
            B[:, k:k+1] = w_k * u_k.conj() * h_k
        
        # 添加正则化项确保可逆
        A += 1e-8 * np.eye(N_t)
        
        # 更新预编码矩阵
        V_new = np.linalg.inv(A) @ B
        
        # 功率归一化
        power = np.linalg.norm(V_new, 'fro')**2
        if power > P_max:
            V_new = V_new / np.sqrt(power) * np.sqrt(P_max)
        
        # 计算目标函数值
        sum_utility = 0.0
        for k in range(K):
            h_k = H[k, :].reshape(-1, 1)
            v_k = V_new[:, k].reshape(-1, 1)
            e_k = 1 + np.abs(h_k.conj().T @ v_k)**2 / noise_power
            sum_utility += weights[k] * np.log2(e_k)
        
        obj_history.append(sum_utility.item())
        
        # 检查收敛
        if iter > 0:
            improvement = abs(obj_history[-1] - obj_history[-2]) / abs(obj_history[-2] + 1e-10)
            if improvement < tol:
                if verbose:
                    print(f"迭代 {iter+1}: 目标值 = {obj_history[-1]:.6f}, 收敛")
                break
        
        if verbose and (iter % 10 == 0 or iter == max_iter - 1):
            print(f"迭代 {iter+1}: 目标值 = {obj_history[-1]:.6f}")
        
        V = V_new
    
    return V, U.reshape(-1, 1), obj_history


def calculate_sum_rate(H: np.ndarray, V: np.ndarray, noise_power: float = 1.0) -> float:
    """
    计算和速率
    
    Args:
        H: 信道矩阵 (K x N_t)
        V: 预编码矩阵 (N_t x K)
        noise_power: 噪声功率
        
    Returns:
        sum_rate: 和速率 (bps/Hz)
    """
    K = H.shape[0]
    sum_rate = 0.0
    
    for k in range(K):
        h_k = H[k, :].reshape(-1, 1)  # (N_t, 1)
        v_k = V[:, k].reshape(-1, 1)  # (N_t, 1)
        
        # 信号功率
        signal_power = np.abs(h_k.conj().T @ v_k)**2
        
        # 干扰功率
        interference_power = 0.0
        for j in range(K):
            if j != k:
                v_j = V[:, j].reshape(-1, 1)
                interference_power += np.abs(h_k.conj().T @ v_j)**2
        
        # SINR
        sinr = signal_power / (interference_power + noise_power)
        
        # 速率
        sum_rate += np.log2(1 + sinr.item())
    
    return sum_rate


def zero_forcing_precoding(H: np.ndarray, P_max: float) -> np.ndarray:
    """
    迫零预编码（用于对比）
    
    Args:
        H: 信道矩阵 (K x N_t)
        P_max: 最大发射功率
        
    Returns:
        V_zf: 迫零预编码矩阵 (N_t x K)
    """
    K, N_t = H.shape
    
    # 迫零预编码：V = H^H (H H^H)^{-1}
    if K <= N_t:
        V_zf = H.conj().T @ np.linalg.inv(H @ H.conj().T)
    else:
        # 用户数多于天线数，使用伪逆
        V_zf = np.linalg.pinv(H)
    
    # 功率归一化
    power = np.linalg.norm(V_zf, 'fro')**2
    V_zf = V_zf / np.sqrt(power) * np.sqrt(P_max)
    
    return V_zf


def main():
    """主函数：演示WMMSE算法"""
    # 仿真参数
    num_users = 4
    num_antennas = 8
    P_max = 10.0  # 10W
    noise_power = 0.1
    seed = 42
    
    print("=== WMMSE预编码算法演示 ===")
    print(f"系统配置: {num_users}用户, {num_antennas}天线")
    print(f"功率约束: {P_max} W, 噪声功率: {noise_power}")
    
    # 生成信道
    H = generate_channel(num_users, num_antennas, seed)
    print(f"信道矩阵形状: {H.shape}")
    
    # 运行WMMSE算法
    print("\n运行WMMSE算法...")
    V_wmmse, U_wmmse, obj_history = wmmse_precoding(
        H, P_max, noise_power=noise_power, max_iter=50, tol=1e-6, verbose=True
    )
    
    # 计算性能
    sum_rate_wmmse = calculate_sum_rate(H, V_wmmse, noise_power)
    print(f"\nWMMSE和速率: {sum_rate_wmmse:.4f} bps/Hz")
    
    # 对比：迫零预编码
    V_zf = zero_forcing_precoding(H, P_max)
    sum_rate_zf = calculate_sum_rate(H, V_zf, noise_power)
    print(f"迫零预编码和速率: {sum_rate_zf:.4f} bps/Hz")
    
    # 对比：匹配滤波器
    V_mf = H.conj().T
    power_mf = np.linalg.norm(V_mf, 'fro')**2
    V_mf = V_mf / np.sqrt(power_mf) * np.sqrt(P_max)
    sum_rate_mf = calculate_sum_rate(H, V_mf, noise_power)
    print(f"匹配滤波器和速率: {sum_rate_mf:.4f} bps/Hz")
    
    print("\n=== 演示完成 ===")
    
    return {
        'H': H,
        'V_wmmse': V_wmmse,
        'U_wmmse': U_wmmse,
        'obj_history': obj_history,
        'sum_rates': {
            'WMMSE': sum_rate_wmmse,
            'ZF': sum_rate_zf,
            'MF': sum_rate_mf
        }
    }


if __name__ == "__main__":
    results = main()