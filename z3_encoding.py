# 檔名: z3_encoding.py
from z3 import *

def step3_bitvector_encoding():
    print("--- Step 3: Bitvector Encoding (平行化檢查) ---")
    s = Solver()
    
    # 1. 模擬波形
    steps = 8
    alert_trace = [BitVec(f'alert_{t}', 1) for t in range(steps)]
    
    # 2. 設定訊號 (只有 t=5 是 1，其他是 0)
    for t in range(steps):
        if t == 5:
            s.add(alert_trace[t] == 1)
        else:
            s.add(alert_trace[t] == 0)
            
    # --- Bitvector Encoding ---
    print("正在建構 Bitvector Encoding...")
    
    # 初始化：拿第一個 bit
    packed_bv = alert_trace[0]
    
    # 拼接迴圈
    for t in range(1, steps):
        packed_bv = Concat(alert_trace[t], packed_bv) 
        
    # 3. 驗證：檢查是否大於 0
    s.add(packed_bv > 0)
    
    print("Solver 檢查中：有沒有任何警報發生？")
    
    # --- 注意這裡的縮排，必須跟上面的 print 對齊 ---
    if s.check() == sat:
        print("Verified: Eventually an alert happened! (Sat)")
        m = s.model()
        
        # --- 這裡包含了剛剛修復的 .eval() ---
        val = m.eval(packed_bv).as_long()
        
        print(f"Packed Bitvector Value: {val:08b}")
        print("看到了嗎？那個 '1' 就是在第 5 秒發生的警報！")
    else:
        print("Failed: No alert ever happened.")

if __name__ == "__main__":
    step3_bitvector_encoding()