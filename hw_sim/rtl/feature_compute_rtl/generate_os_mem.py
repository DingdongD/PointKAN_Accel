import numpy as np

def to_hex(val, width_hex, signed=False, bits=8):
    if signed and val < 0:
        val = (1 << bits) + val
    return format(val & ((1 << bits) - 1), f"0{width_hex}x")

def generate_os_mem(
    row=8,
    col=8,
    k=8,
    word_size=16,
    signed=True,
    seed=42,
    out_dir="."
):
    np.random.seed(seed)

    if signed:
        low = -(2 ** (word_size - 1))
        high = 2 ** (word_size - 1)
        A = np.random.randint(low, high, size=(row, k), dtype=np.int64)
        B = np.random.randint(low, high, size=(k, col), dtype=np.int64)
    else:
        low = 0
        high = 2 ** word_size
        A = np.random.randint(low, high, size=(row, k), dtype=np.int64)
        B = np.random.randint(low, high, size=(k, col), dtype=np.int64)

    C = A.dot(B)

    hex_w_ab = (word_size + 3) // 4

    # 改成 64-bit golden
    c_bits = 64
    hex_w_c = (c_bits + 3) // 4

    with open(f"{out_dir}/A.mem", "w") as fa:
        for r in range(row):
            for kk in range(k):
                fa.write(to_hex(int(A[r, kk]), hex_w_ab, signed=signed, bits=word_size) + "\n")

    with open(f"{out_dir}/B.mem", "w") as fb:
        for kk in range(k):
            for c in range(col):
                fb.write(to_hex(int(B[kk, c]), hex_w_ab, signed=signed, bits=word_size) + "\n")

    with open(f"{out_dir}/C_full.mem", "w") as fc:
        for r in range(row):
            for c in range(col):
                fc.write(to_hex(int(C[r, c]), hex_w_c, signed=True, bits=c_bits) + "\n")

    print("A =")
    print(A)
    print("B =")
    print(B)
    print("C = A x B =")
    print(C)
    print("Generated: A.mem, B.mem, C_full.mem (64-bit golden)")

if __name__ == "__main__":
    generate_os_mem(
        row=8,
        col=8,
        k=8,
        word_size=16,
        signed=True,
        seed=42,
        out_dir="C:\\Digital_Projects\\VerilogProjs\\HelloVerilog\\PointAcc\\feature_compute"
    )