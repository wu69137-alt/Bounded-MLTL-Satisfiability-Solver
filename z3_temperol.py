# 檔名: z3_temporal.py
from z3 import *

def step2_temporal_unrolling():
    print("--- Step 2: Temporal Unrolling (模擬波形) ---")
    s = Solver()
    
    # 1. 定義時間長度 (Simulation Time)
    # 我們想看 10 個時間點的變化
    steps = 10
    
    # 2. 宣告波形 (Waveform)
    # 建立一個 list，裡面有 10 個 4-bit 的變數
    # trace[0] 就是 t=0 的值, trace[1] 就是 t=1 的值...
    # 這就像 Verilog 裡的: reg [3:0] trace [0:9];
    trace = [BitVec(f'cnt_{t}', 4) for t in range(steps)]
    
    # 3. 設定初始狀態 (Initial State, Reset)
    # t=0 時，計數器必須是 0
    s.add(trace[0] == 0)
    
    # 4. 設定狀態轉移 (State Transition / Next State Logic)
    # 這就是 Verilog 裡的 always @(posedge clk) cnt <= cnt + 1;
    for t in range(steps - 1): # 從 t=0 到 t=8
        # 下一秒的值 = 這一秒 + 1
        s.add(trace[t+1] == trace[t] + 1)
        
    # 5. 設定驗證目標 (Verification Goal / Assertion)
    # 我們想證明：「如果不歸零，第 5 秒絕對不可能跳到 10 (0xA)」
    # 在 Formal Verification 裡，我們通常用「歸謬法」：
    # 我們挑戰 Solver：「你能不能找出一種情況，讓第 5 秒變成 10？」
    print("正在挑戰 Solver：有沒有可能在 t=5 時變成 10？")
    s.add(trace[5] == 10)
    
    # 6. 求解
    if s.check() == sat:
        print("Bug Found! 居然發生了！(Satisfiable)")
        print("以下是反例波形 (Counter-example Waveform)：")
        m = s.model()
        for t in range(steps):
            # 印出每個時間點的數值
            val = m[trace[t]].as_long()
            print(f"Time {t}: {val}")
    else:
        print("Safe! (Unsatisfiable)")
        print("證明完畢：計數器正常運作，第 5 秒絕對不會是 10。")

if __name__ == "__main__":
    step2_temporal_unrolling()
