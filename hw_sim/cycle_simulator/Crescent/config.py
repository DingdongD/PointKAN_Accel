from dataclasses import dataclass

@dataclass
class SimConfig:
    seed: int = 42
    num_points: int = 256
    num_queries: int = 32
    leaf_size: int = 16
    top_tree_height: int = 2
    radius: float = 0.22
    max_neighbors: int = 24

    # tree-buffer SRAM (for kd-tree nodes)
    tree_buf_capacity_bytes: int = 256 * 1024
    tree_buf_num_banks: int = 8
    tree_buf_bank_width_bytes: int = 32
    tree_buf_ports_per_bank: int = 1
    tree_buf_access_latency_cycles: int = 1
    tree_buf_read_energy_pj: float = 4.0
    tree_buf_write_energy_pj: float = 4.5

    # feature-buffer SRAM (for gathered point features)
    feat_dim: int = 32
    feat_buf_capacity_bytes: int = 512 * 1024
    feat_buf_num_banks: int = 8
    feat_buf_bank_width_bytes: int = 32
    feat_buf_ports_per_bank: int = 1
    feat_buf_access_latency_cycles: int = 1
    feat_buf_read_energy_pj: float = 5.0
    feat_buf_write_energy_pj: float = 5.5

    # neighbor-index / result SRAM
    index_buf_capacity_bytes: int = 128 * 1024
    index_buf_num_banks: int = 4
    index_buf_bank_width_bytes: int = 16
    index_buf_ports_per_bank: int = 1
    index_buf_access_latency_cycles: int = 1
    index_buf_read_energy_pj: float = 2.0
    index_buf_write_energy_pj: float = 2.5

    # BCE
    enable_bce_tree: bool = True
    enable_bce_feat: bool = True
    elision_height_tree: int = 4
    elision_height_feat: int = 1000  # feature gather is not tree-level aware; use synthetic large level threshold

    # aggregation behavior
    aggregate_mode: str = 'mean'  # mean|max
    fill_missing_with_zero: bool = True

    # simplistic DRAM model
    dram_burst_bytes: int = 64
    dram_read_energy_pj_per_burst: float = 60.0
    dram_write_energy_pj_per_burst: float = 70.0
