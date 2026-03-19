from consumer.feature_extractor import FeatureExtractor
from utils.output_schema import UnifiedOutputBuilder


def test_align_elf_to_bond_indices_representative_molecules():
    """
    轻量验证 5 个代表分子：
    - 使用 RDKit bond_indices 作为 target
    - 人工构造 ELF bond pairs（刻意缺失/偏差）模拟 Mayer 阈值导致的集合差异
    - 验证对齐后长度始终等于 target，且统计信息合理
    """
    extractor = FeatureExtractor()

    cases = [
        ("ethanol", [[0, 1], [1, 2]]),
        ("benzene", [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]),
        ("pyridine", [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]]),
        ("acetic_acid", [[0, 1], [1, 2], [1, 3]]),
        ("caffeine_like", [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9]]),
    ]

    for name, bond_indices in cases:
        assert bond_indices, f"{name}: expected non-empty bond_indices"

        # 人工制造“ELF 仅覆盖部分键 + 额外非拓扑键”的场景
        kept = bond_indices[: max(1, len(bond_indices) // 2)]
        extra = [[0, len(bond_indices)]]
        elf_pairs = [tuple(b) for b in kept + extra]
        elf_midpoints = [0.5 + 0.01 * i for i in range(len(elf_pairs))]

        aligned, stats = extractor._align_elf_to_bond_indices(
            elf_midpoints=elf_midpoints,
            elf_bond_pairs=elf_pairs,
            bond_indices=bond_indices,
            molecule_id=name,
        )

        assert len(aligned) == len(bond_indices), f"{name}: alignment length mismatch"
        assert stats["raw_count"] == len(elf_midpoints), f"{name}: raw_count mismatch"
        assert stats["aligned_count"] <= len(bond_indices), f"{name}: invalid aligned_count"
        assert stats["dropped_count"] >= 0, f"{name}: invalid dropped_count"


def test_validate_lengths_no_mismatch_after_zero_fill_alignment():
    """
    覆盖 ELF 提取失败时的 zero-fill 对齐行为：
    只要 elf_bond_midpoint 长度与 bond_indices 一致，就不应产生 mismatch reason。
    """
    extractor = FeatureExtractor()
    builder = UnifiedOutputBuilder(molecule_id="m1", smiles="CCO")
    builder.set_molecule_info(natm=3)
    builder.set_atom_features(atomic_number=[6, 6, 8], charge_mulliken=[0.0, 0.0, 0.0])
    builder.set_bond_features(
        bond_indices=[[0, 1], [1, 2]],
        bond_orders_mayer=[1.0, 1.0],
        bond_orders_wiberg=[0.9, 0.9],
        elf_bond_midpoint=[0.0, 0.0],
        bond_types_rdkit=["SINGLE", "SINGLE"],
    )

    valid, reasons = extractor._validate_feature_lengths(builder, molecule_id="m1")
    assert valid is True
    assert "bond_feature_length_mismatch_elf_bond_midpoint" not in reasons
