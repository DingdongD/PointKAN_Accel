# Dispatcher / Policy patch

## Added files
- `reconfig_sim/workloads/op_desc.py`
- `reconfig_sim/dispatch/policies.py`
- `reconfig_sim/dispatch/dispatcher.py`
- `reconfig_sim/examples/policy_demo.py`

## Intended integration points
1. Replace any old `AUTO` selector with `RuleBasedPolicy`.
2. For experiments, instantiate one of:
   - `FixedModePolicy("WS")`
   - `FixedModePolicy("OS")`
   - `RuleBasedPolicy()`
3. Pass the policy into `ModeAwareDispatcher(hw, policy)`.
4. Use `dispatch(ops, overrides={...})` for manual override.

## Example op-type mapping
- `linear`, `conv2d_im2col`, `attention_proj` -> `WS`
- `attention_qk`, `attention_av`, `gemm_dynamic` -> `OS`

This matches the design intent: most static-weight operators use WS, while dynamic attention GEMMs use OS.
