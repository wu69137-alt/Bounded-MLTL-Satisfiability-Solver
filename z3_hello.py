 # 檔名: z3_hello.py
from z3 import *

def step1_hello_world():
    print("--- Step 1: Solving Simple Constraints ---")
    
    # 1. 宣告變數 (BitVec 就像 Verilog 的 reg [7:0])
    # 我們用 8-bit，剛好模擬一個 byte 的運算
    x = BitVec('x', 8) 
    y = BitVec('y', 8)
    
    # 2. 建立 Solver (求解器)
    s = Solver()
    
    # 3. 加入規則 (Constraints)
    # 想像你在寫 Verilog 的 assertion
    s.add(x + y == 100)      # 兩數相加等於 100
    s.add(x % 2 == 0)        # x 是偶數
    s.add(y % 2 == 0)        # y 是偶數
    s.add(x != y)            # x 不等於 y (多加一個條件)

    # 4. 求解 (Check Satisfiability)
    print("Solver 正在思考中...")
    result = s.check()
    
    if result == sat:
        print("找到解了! (Satisfiable)")
        m = s.model()
        # 取得 x 和 y 的具體數值
        print(f"x = {m[x]}")
        print(f"y = {m[y]}")
        
        # 額外小知識：在 MobaXterm 裡看到這個就像看到 Simulation Pass!
    else:
        print("無解 (Unsatisfiable)")

if __name__ == "__main__":
    step1_hello_world()
