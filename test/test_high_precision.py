#!/usr/bin/env python3
"""
高精度 DFT 测试：验证 D3BJ 色散校正、IAO 电荷和 CM5 电荷
"""

import sys
sys.path.insert(0, '/home/sulixian/FastORCA')

from pyscf import gto, dft
from producer.dft_calculator import DFTCalculator
from consumer.feature_extractor import FeatureExtractor

print("=" * 60)
print("高精度 DFT 测试：苯环 (c1ccccc1)")
print("=" * 60)

# 1. 测试 DFT 计算（含 D3BJ 色散校正）
print("\n--- 1. 测试 DFT 计算（def2-svp + D3BJ）---")

calc = DFTCalculator(functional="B3LYP", basis="def2-svp")
mol = calc.from_smiles("c1ccccc1")

print(f"分子: 苯环 (C6H6)")
print(f"基组: def2-svp")
print(f"原子数: {mol.natm}")

# 执行 DFT 计算
mf = calc.run_sp("benzene_test", mol)

print(f"\n✅ DFT 计算完成")
print(f"总能量: {mf.e_tot:.6f} Hartree")
print(f"D3BJ 色散校正: {'已启用' if hasattr(mf, 'disp') and mf.disp else '未启用'}")

# 导出波函数
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    result = calc.calculate_and_export("benzene_test", mol, tmpdir)
    
    # 2. 测试高级电荷分析
    print("\n--- 2. 测试高级电荷分析 ---")
    
    extractor = FeatureExtractor()
    features = extractor.extract_all_features(result["pkl_file"], "benzene_test")
    
    if features["success"]:
        print("✅ 特征提取成功\n")
        
        # 打印电荷信息
        print("原子符号:", features["features"]["atom_symbols"])
        print()
        
        print("电荷对比:")
        print("-" * 60)
        print(f"{'原子':<6} {'IAO':<12} {'CM5':<12} {'Hirshfeld':<12} {'Mulliken':<12}")
        print("-" * 60)
        
        for i, symbol in enumerate(features["features"]["atom_symbols"]):
            iao = features["features"]["charge_iao"][i]
            cm5 = features["features"]["charge_cm5"][i]
            hirsh = features["features"]["hirshfeld_charges"][i]
            mull = features["features"]["mulliken_charges"][i]
            print(f"{symbol:<6} {iao:<12.4f} {cm5:<12.4f} {hirsh:<12.4f} {mull:<12.4f}")
        
        print("-" * 60)
        print(f"{'总和':<6} {sum(features['features']['charge_iao']):<12.4f} {sum(features['features']['charge_cm5']):<12.4f} {sum(features['features']['hirshfeld_charges']):<12.4f} {sum(features['features']['mulliken_charges']):<12.4f}")
        
        print("\n物理分析:")
        print("- IAO 电荷: NPA 的平替，物理意义清晰，推荐用于机器学习")
        print("- CM5 电荷: Hirshfeld 的修正版，改善了对电负性的描述")
        print("- 苯环中 C 原子应带轻微负电荷或接近中性，H 原子带正电荷")
        
        # 验证 D3BJ
        print(f"\n--- 3. D3BJ 色散校正验证 ---")
        print(f"色散校正开启: {features['features']['dispersion_correction']}")
        
    else:
        print(f"❌ 特征提取失败: {features['error']}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
