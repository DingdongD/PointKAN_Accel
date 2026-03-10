from reconfig_sim.dispatch.dispatcher import HardwareConfig, ModeAwareDispatcher
from reconfig_sim.dispatch.policies import FixedModePolicy, RuleBasedPolicy
from reconfig_sim.workloads.op_desc import OpDesc, conv2d_to_im2col_op


def build_ops():
    return [
        conv2d_to_im2col_op("stem_conv", batch=1, c_in=3, h_in=224, w_in=224, c_out=64, k_h=7, k_w=7, stride=2, padding=3),
        OpDesc("q_proj", "attention_proj", m=197, n=768, k=768, has_static_weight=True),
        OpDesc("qk_matmul", "attention_qk", m=197, n=197, k=64, has_static_weight=False),
        OpDesc("av_matmul", "attention_av", m=197, n=64, k=197, has_static_weight=False),
        OpDesc("ffn_up", "linear", m=197, n=3072, k=768, has_static_weight=True),
    ]


def main():
    hw = HardwareConfig(arr_r=8, arr_c=8)
    ops = build_ops()

    for name, policy in {
        "all_ws": FixedModePolicy("WS"),
        "all_os": FixedModePolicy("OS"),
        "rule_based": RuleBasedPolicy(),
    }.items():
        dispatcher = ModeAwareDispatcher(hw, policy)
        result = dispatcher.dispatch(ops)
        print(f"[{name}] cycles={result.total_cycles} energy={result.total_energy:.2f} mode_hist={result.mode_hist}")
        for tile, stats in list(zip(result.schedule, result.tile_stats))[:5]:
            print(" ", tile, stats.cycles, stats.energy)
        print()


if __name__ == "__main__":
    main()
