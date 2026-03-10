from reconfig_sim.dispatch.dispatcher import HardwareConfig, ModeAwareDispatcher
from reconfig_sim.dispatch.policies import FixedModePolicy, RuleBasedPolicy
from reconfig_sim.workloads.op_desc import OpDesc


def main():
    hw = HardwareConfig(arr_r=4, arr_c=4)
    ops = [
        OpDesc("fc1", "linear", m=16, n=16, k=16, has_static_weight=True),
        OpDesc("qk", "attention_qk", m=32, n=32, k=64, has_static_weight=False),
    ]

    res_ws = ModeAwareDispatcher(hw, FixedModePolicy("WS")).dispatch(ops)
    res_os = ModeAwareDispatcher(hw, FixedModePolicy("OS")).dispatch(ops)
    res_rb = ModeAwareDispatcher(hw, RuleBasedPolicy()).dispatch(ops)

    assert res_ws.mode_hist["WS"] > 0
    assert res_os.mode_hist["OS"] > 0
    assert res_rb.mode_hist["WS"] > 0 and res_rb.mode_hist["OS"] > 0
    print("dispatcher smoke test passed")


if __name__ == "__main__":
    main()
