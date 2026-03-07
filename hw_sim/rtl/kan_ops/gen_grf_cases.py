import json
import math
import struct
from typing import List, Dict


def float_to_fp32_hex(x: float) -> str:
    """Python float -> IEEE754 single-precision hex string like 32'h3f800000"""
    b = struct.pack(">f", float(x))
    u = struct.unpack(">I", b)[0]
    return f"32'h{u:08x}"


def float_to_fp32_u32(x: float) -> int:
    b = struct.pack(">f", float(x))
    return struct.unpack(">I", b)[0]


def fp32_u32_to_float(u: int) -> float:
    b = struct.pack(">I", u & 0xFFFFFFFF)
    return struct.unpack(">f", b)[0]


def grf_reference(x: float, a: List[float], b: List[float]) -> float:
    """
    P = a0 + a1*x + a2*x^2 + ... + a5*x^5
    Q = 1 + b0*x + b1*x^2 + b2*x^3 + b3*x^4
    y = P / Q
    """
    x1 = x
    x2 = x1 * x1
    x3 = x2 * x1
    x4 = x2 * x2
    x5 = x3 * x2

    p = a[0] + a[1] * x1 + a[2] * x2 + a[3] * x3 + a[4] * x4 + a[5] * x5
    q = 1.0 + b[0] * x1 + b[1] * x2 + b[2] * x3 + b[3] * x4
    return p / q


def make_cases() -> List[Dict]:
    """
    你可以在这里自由扩展测试样例
    """
    cases = []

    # case 0: y = 1 + x
    x = 0.5
    a = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0, 0.0]
    cases.append({"name": "y=1+x @ x=0.5", "x": x, "a": a, "b": b})

    # case 1: y = 1 + x
    x = 1.0
    a = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0, 0.0]
    cases.append({"name": "y=1+x @ x=1.0", "x": x, "a": a, "b": b})

    # case 2: y = x^2
    x = 2.0
    a = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0, 0.0]
    cases.append({"name": "y=x^2 @ x=2.0", "x": x, "a": a, "b": b})

    # case 3: y = (1+x+x^2)/(1+x)
    x = 1.0
    a = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0, 0.0]
    cases.append({"name": "(1+x+x^2)/(1+x) @ x=1.0", "x": x, "a": a, "b": b})

    # case 4: 一个一般 GRF
    x = 0.75
    a = [0.3, -0.8, 1.2, -0.5, 0.25, 0.1]
    b = [0.2, -0.1, 0.05, 0.01]
    cases.append({"name": "general_case_0", "x": x, "a": a, "b": b})

    # case 5: 再来一个一般 GRF
    x = -0.5
    a = [0.8, 0.1, -0.3, 0.6, -0.2, 0.05]
    b = [-0.15, 0.02, -0.01, 0.005]
    cases.append({"name": "general_case_1", "x": x, "a": a, "b": b})

    # 预先算 reference 和 hex
    for i, c in enumerate(cases):
        y_ref = grf_reference(c["x"], c["a"], c["b"])
        c["case_id"] = i
        c["y_ref"] = y_ref
        c["x_hex"] = float_to_fp32_hex(c["x"])
        c["y_ref_hex"] = float_to_fp32_hex(y_ref)
        c["a_hex"] = [float_to_fp32_hex(v) for v in c["a"]]
        c["b_hex"] = [float_to_fp32_hex(v) for v in c["b"]]

    return cases


def write_json(cases: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)


def write_memh(cases: List[Dict], path: str) -> None:
    """
    每行一个 case:
    x a0 a1 a2 a3 a4 a5 b0 b1 b2 b3 y_ref
    都是 8位hex，不带 32'h 前缀
    """
    with open(path, "w", encoding="utf-8") as f:
        for c in cases:
            nums = [
                float_to_fp32_u32(c["x"]),
                *[float_to_fp32_u32(v) for v in c["a"]],
                *[float_to_fp32_u32(v) for v in c["b"]],
                float_to_fp32_u32(c["y_ref"]),
            ]
            line = " ".join(f"{u:08x}" for u in nums)
            f.write(line + "\n")


def print_tb_snippets(cases: List[Dict]) -> None:
    print("\n=== Copy these send_sample() calls into testbench ===\n")
    for c in cases:
        a = c["a_hex"]
        b = c["b_hex"]
        print(f"// case {c['case_id']}: {c['name']}, y_ref={c['y_ref']}")
        print("send_sample(")
        print(f"    {c['x_hex']},")
        print(f"    {a[0]}, {a[1]}, {a[2]}, {a[3]}, {a[4]}, {a[5]},")
        print(f"    {b[0]}, {b[1]}, {b[2]}, {b[3]}")
        print(");\n")


def main() -> None:
    cases = make_cases()
    write_json(cases, "grf_cases.json")
    write_memh(cases, "grf_cases.memh")
    print_tb_snippets(cases)

    print("Generated:")
    print("  - grf_cases.json")
    print("  - grf_cases.memh")


if __name__ == "__main__":
    main()