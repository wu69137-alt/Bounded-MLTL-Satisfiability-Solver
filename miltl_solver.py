# -*- coding: utf-8 -*-
"""
miltl_solver.py

這個檔案示範一個「很小片」的 MLTL SAT solver，支援：
- 原子命題：p, q, r ...
- 布林運算：not, and, or, implies
- 時序運算：
    F[a,b] φ    （在未來 a~b 步內，φ 至少真一次）
    G[a,b] φ    （在未來 a~b 步內，φ 一直都真）
    φ U[a,b] ψ  （φ 一直真，直到 a~b 步內某時刻 ψ 真）

重點：全部東西都串在一個檔案裡，並且有非常多中文註解。
"""

# ------------------------------------------------------------
# 第 0 步：匯入需要的工具
# ------------------------------------------------------------

# dataclass：幫我們方便定義「資料結構」，有點像 struct
from dataclasses import dataclass
import csv
# z3：SMT solver。這裡只用到一些基本東西。
from z3 import (
    Solver,
    Bool,
    And as zAnd,       # 把 z3 的 And 重新命名為 zAnd，避免跟我們的 AndF 類別撞名
    Or as zOr,
    Not as zNot,
    Implies as zImplies,
    is_true,
    sat, unsat, unknown,
)

# ------------------------------------------------------------
# 第 1 步：定義「公式」的樹狀結構 (AST)
# ------------------------------------------------------------
#
# 想法：
#   把一個公式想成一棵樹，像這樣：
#
#     (p & q) | ~r
#
#         Or
#        /  \
#      And   Not
#     /  \    |
#    p    q   r
#
# 每一個節點就是一個「類別的物件」。
# 例如 Var / NotF / AndF / OrF / ImpliesF / F / G / U
#
# 所有類別都繼承 Formula，表示「這是一種公式」。


class Formula:
    """所有公式類別的共同父類別，自己不放資料。"""
    pass


# --------- 布林運算子 ----------

@dataclass
class Var(Formula):
    """原子命題，例如 p, q, r ..."""
    name: str  # 例如 "p"


@dataclass
class NotF(Formula):
    """布林的否定：not φ"""
    phi: Formula  # 裡面包一個子公式


@dataclass
class AndF(Formula):
    """布林的且：φ and ψ"""
    left: Formula   # 左子公式
    right: Formula  # 右子公式


@dataclass
class OrF(Formula):
    """布林的或：φ or ψ"""
    left: Formula
    right: Formula


@dataclass
class ImpliesF(Formula):
    """布林的蘊含：φ -> ψ"""
    left: Formula
    right: Formula


# --------- 時序運算子 ----------

@dataclass
class F_op(Formula):
    """F[a,b] φ：在未來 a~b 步內，φ 至少真一次"""
    a: int
    b: int
    phi: Formula


@dataclass
class G_op(Formula):
    """G[a,b] φ：在未來 a~b 步內，φ 一直都真"""
    a: int
    b: int
    phi: Formula


@dataclass
class U_op(Formula):
    """φ U[a,b] ψ：φ 一直真，直到 a~b 步內某時刻 ψ 真"""
    a: int
    b: int
    phi: Formula  # 左邊：φ
    psi: Formula  # 右邊：ψ


# ------------------------------------------------------------
# 第 2 步：漂亮印出這棵樹，幫你「看見」 AST 長什麼樣子
# 一直 睡覺 直到0到8小時內 鬧鐘響且天亮 才 起床
# 跳到U 印U[0,8]
# 空兩個phi 
# 空四格印睡覺
# 空四格印And
# 空六格印鬧鐘響
# 空六格印天亮
# To make it more like a Tree!
# ------------------------------------------------------------

def pretty_print(formula: Formula, indent: int = 0) -> None:
    """把公式的樹狀結構印出來。indent 表示縮排層數。"""
    space = "  " * indent  # 每一層多兩個空白

    if isinstance(formula, Var):
        print(f"{space}Var({formula.name})")

    elif isinstance(formula, NotF):
        print(f"{space}Not")
        pretty_print(formula.phi, indent + 1)

    elif isinstance(formula, AndF):
        print(f"{space}And")
        pretty_print(formula.left, indent + 1)
        pretty_print(formula.right, indent + 1)

    elif isinstance(formula, OrF):
        print(f"{space}Or")
        pretty_print(formula.left, indent + 1)
        pretty_print(formula.right, indent + 1)

    elif isinstance(formula, ImpliesF):
        print(f"{space}Implies")
        pretty_print(formula.left, indent + 1)
        pretty_print(formula.right, indent + 1)

    elif isinstance(formula, F_op):
        print(f"{space}F[{formula.a},{formula.b}]")
        pretty_print(formula.phi, indent + 1)

    elif isinstance(formula, G_op):
        print(f"{space}G[{formula.a},{formula.b}]") #G一直 gloabal 不斷的(先印G[時間區間])再做下一件事(phi)然後呼叫 換行了
        pretty_print(formula.phi, indent + 1)

    elif isinstance(formula, U_op):
        print(f"{space}U[{formula.a},{formula.b}]")
        print(f"{space}  (phi)")
        pretty_print(formula.phi, indent + 2)
        print(f"{space}  (psi)")
        pretty_print(formula.psi, indent + 2)

    else:
        print(f"{space}Unknown node: {formula}")


# ------------------------------------------------------------
# 第 3 步：收集所有「原子命題名字」，例如 p, q, r ...
# ------------------------------------------------------------

def collect_vars(formula: Formula, acc=None):
    """
    走過整棵 AST，把所有 Var(name) 的 name 收集到一個 set 裡。
    例如公式裡出現 Var("p"), Var("q")，就會拿到 {"p", "q"}。
    """
    if acc is None:
        acc = set()

    if isinstance(formula, Var):
        acc.add(formula.name)

    elif isinstance(formula, NotF):
        collect_vars(formula.phi, acc)

    elif isinstance(formula, (AndF, OrF, ImpliesF)):
        collect_vars(formula.left, acc)
        collect_vars(formula.right, acc)

    elif isinstance(formula, (F_op, G_op)):
        collect_vars(formula.phi, acc)

    elif isinstance(formula, U_op):
        collect_vars(formula.phi, acc)
        collect_vars(formula.psi, acc)

    return acc


# ------------------------------------------------------------
# 第 4 步：為每個原子命題產生「一條時間軸上的 Bool 變數」
# ------------------------------------------------------------
#
# 想像：trace 長度為 N。
# 每個原子命題 name（例如 "p"），在每個時間點 i 都有一個 Bool：
#   p_0, p_1, ..., p_{N-1}
#
# 這就像 Verilog 裡面：
#   wire [N-1:0] p;
#
# 不同的是，在 SMT 裡，我們不知道這些 bit 是 0 還是 1，
# 要靠 solver 去找一組 assignment 讓公式成立。


def mk_atom_bitvectors(var_names, N):
    """
    var_names: 一個 set，例如 {"p","q"}
    N        : trace 長度
    回傳    : dict，key 是名字，value 是 list of Bool
              例如 {"p": [p_0, p_1, ...], "q": [q_0, q_1, ...]}
    """
    bv = {}
    for name in var_names:
        bits = []
        for i in range(N):
            bits.append(Bool(f"{name}_{i}"))
        bv[name] = bits
    return bv
##def mk_atom_bitvectors(var_names, N):
# 直接用一行搞定：外層是字典生成式，內層是列表生成式
#return {name: [Bool(f"{name}_{i}") for i in range(N)] for name in var_names}



# ------------------------------------------------------------
# 第 5 步：把「公式」變成「每個時間點上的 bitvector」＋ 語意約束
# ------------------------------------------------------------
#
# encode(formula, N, atom_bv, solver, cache)
#
# - 對每個子公式 φ，產生一條長度 N 的 Bool 列表 bits_phi[i]
# - 根據 MLTL 的語意，對 solver.add(...) 加上約束，
#   讓 bits_phi[i] 必須跟 atom_bv、其他子公式的 bits 的關係一致。
#
# 例子：布林 and
#   如果 φ = (ψ and χ)
#   那麼對每個時間 i：
#     bits_phi[i] == bits_ψ[i] AND bits_χ[i]
#
# 例子：F[a,b] φ
#   bits_F[i] == OR_{k=a..b 且 i+k<N} bits_phi[i+k]
#
# 我們用 cache 避免重複編碼同一個子公式（同一個物件）。


def encode(formula: Formula, N: int, atom_bv, solver: Solver, cache=None):
    """
    回傳：對應這個 formula 的 bit 列表 bits[0..N-1]
    """
    if cache is None:
        cache = {}

    # 用 id(formula) 當 key，而不是 formula 本身
    key = id(formula)

    # 如果這個公式已經編碼過，就直接回傳舊的（避免重複）
    if key in cache:
        return cache[key]

    # Var 特例：直接用 atom_bv 裡的那條線
    if isinstance(formula, Var):
        bits = atom_bv[formula.name]
        cache[key] = bits
        return bits

    # 對非 Var，我們先建立一條新的 bitvector
    bits = [Bool(f"f_{id(formula)}_{i}") for i in range(N)]
    #id(formula) it means the object's number, everyone differs.
    #e.g. f_9487_0, f_9487_1, f_9487_N
    #""引號裡面 {} 包起來的東西，當成程式碼執行，不要當成普通文字
    #f_ What is this? It's the name you can choose by yourself.
    #Bool() We tell Z3 some logical varients, which can be added constraints. 

    # ---------------- 布林運算 ----------------

    if isinstance(formula, NotF):### NOT(Sleep)
        phi_bits = encode(formula.phi, N, atom_bv, solver, cache)##We get sleep[0], sleep[1], sleep[2],...  
        for i in range(N):
            # bits[i] = not phi_bits[i]
            solver.add(bits[i] == zNot(phi_bits[i])) #Not(sleep)_[0]=~sleep[0],....Not(sleep)_[N]=~sleep[N]

    elif isinstance(formula, AndF):
        l_bits = encode(formula.left, N, atom_bv, solver, cache)
        r_bits = encode(formula.right, N, atom_bv, solver, cache)
        for i in range(N):
            # bits[i] = l_bits[i] & r_bits[i]
            solver.add(bits[i] == zAnd(l_bits[i], r_bits[i]))

    elif isinstance(formula, OrF):
        l_bits = encode(formula.left, N, atom_bv, solver, cache)
        r_bits = encode(formula.right, N, atom_bv, solver, cache)
        for i in range(N):
            # bits[i] = l_bits[i] | r_bits[i]
            solver.add(bits[i] == zOr(l_bits[i], r_bits[i]))

    elif isinstance(formula, ImpliesF):
        l_bits = encode(formula.left, N, atom_bv, solver, cache)
        r_bits = encode(formula.right, N, atom_bv, solver, cache)
        for i in range(N):
            # bits[i] = (l -> r)
            solver.add(bits[i] == zImplies(l_bits[i], r_bits[i]))##If l_bits[0] is true then r_[0] is true ->  A Implies B_[1] is true Or you can say that ~A or B = A implies B

    # ---------------- 時序運算：F[a,b] ----------------
    # F means I have to find at least one True for the folllowing phi. 
    elif isinstance(formula, F_op):
        phi_bits = encode(formula.phi, N, atom_bv, solver, cache)
        a, b = formula.a, formula.b

        for i in range(N):#From time 0 begin
            # 收集所有合法的 phi[i+k] 
            #k means that we check the time period start from [i+a to i+b] i+b should<N
            #e.g. When it's time[2] and [a,b]==[1,3], we check p[3] to p[5] -> ors[p[3],p[4],p[5]]
            ors = []
            for k in range(a, b + 1):
                t = i + k
                if 0 <= t < N:
                    ors.append(phi_bits[t])

            if not ors:
                # 沒有任何合法時間 → F[i] = False
                solver.add(bits[i] == False)#ors[] nothing in this, I can't find any 1.
            else:
                # bits[i] = OR_{候選} phi_bits[t]
                solver.add(bits[i] == zOr(*ors))
                #solver.add(Result_0 == Or(p_0, p_1))檢查有沒有1

    # ---------------- 時序運算：G[a,b] ----------------
    # G means: I don't want to see any False for the following phi

    elif isinstance(formula, G_op):
        phi_bits = encode(formula.phi, N, atom_bv, solver, cache)
        a, b = formula.a, formula.b

        for i in range(N):
            ands = []
            for k in range(a, b + 1):
                t = i + k
                if 0 <= t < N:
                    ands.append(phi_bits[t])

            if not ands:
                # 完全沒有合法時間點 → vacuum true
                solver.add(bits[i] == True)## No one can say that's wrong, then it's true\.
            else:
                # bits[i] = AND_{候選} phi_bits[t]
                solver.add(bits[i] == zAnd(*ands))

    # ---------------- 時序運算：U[a,b] ----------------
    ## U[i] is True if:
    # There exists a time 't' in [i+a, i+b] where:
    # 1. psi is True at time 't' (達成目標)
    # 2. AND phi is True for all time steps from 'i' to 't-1' (過程堅持住)
    elif isinstance(formula, U_op):
        phi_bits = encode(formula.phi, N, atom_bv, solver, cache)
        psi_bits = encode(formula.psi, N, atom_bv, solver, cache)
        a, b = formula.a, formula.b

        for i in range(N):
            disjuncts = []  # 存所有 k 的候選條件

            # k 是 "ψ 第幾步發生"
            for k in range(a, b + 1):
                t = i + k
                if not (0 <= t < N):
                    continue  # 超出 trace 的就跳過

                psi_at_t = psi_bits[t]

                # φ 要從 i .. t-1 都成立
                conj = []
                for j in range(0, k):
                    tt = i + j
                    if 0 <= tt < N:
                        conj.append(phi_bits[tt])

                if conj:##A U (B) Before B==1 Ashould be true, k==0 B is true即可 因為根本沒之前的時間So we don't have to check A. 
                    # k > 0 的情況：phi[i..i+k-1] 全部要真
                    disjuncts.append(zAnd(psi_at_t, zAnd(*conj))) 
                else:
                    # k == 0：不需要前面 φ 成立，只要當下 ψ 成立即可
                    disjuncts.append(psi_at_t)

            if not disjuncts:
                solver.add(bits[i] == False)
            else:
                solver.add(bits[i] == zOr(*disjuncts)) ## In the time i U[i]裡頭的time i+a~i+b有任一一個是True U[i]就是True
                

    else:
        raise ValueError(f"不認識的公式類別：{formula}")

    # 記得這裡也要用 key
    cache[key] = bits
    return bits



# ------------------------------------------------------------
# 第 6 步：把所有東西串起來，建出 solver
# ------------------------------------------------------------

def mk_solver_for(formula: Formula, N: int):
    """
    給一個公式 + trace 長度 N，建立一個 Z3 Solver。

    流程：
      1. 收集所有原子命題名字
      2. 為每個原子命題做一條長度 N 的 bitvector
      3. encode 整棵公式 → 加進 solver 的約束
      4. 要求「root 公式在時間 0 為真」
    """
    s = Solver()

    # 1. 收集原子命題
    atoms = collect_vars(formula)

    # 2. 為每個原子命題建立 bitvector
    atom_bv = mk_atom_bitvectors(atoms, N)

    # 3. 把公式編碼成 bit + 約束
    cache = {}
    root_bits = encode(formula, N, atom_bv, s, cache)

    # 4. 要求「整個公式在時間 0 為真」
    s.add(root_bits[0] == True)

    return s, atom_bv, root_bits


# ------------------------------------------------------------
# 第 7 步：如果 SAT，就把一條滿足公式的 trace 印出來
# ------------------------------------------------------------

def show_trace(model, atom_bv, N):
    """
    給一個 Z3 的 model，列出每個原子命題在每一個時間點是 0/1。
    很像看 waveform，只是用文字版本。
    """
    atoms = sorted(atom_bv.keys())  # 排序一下，輸出比較固定
    print("time\t" + "\t".join(atoms))
    for i in range(N):
        row = [str(i)]
        for name in atoms:
            val = model.eval(atom_bv[name][i], model_completion=True)
            row.append("1" if is_true(val) else "0")
        print("\t".join(row))
def save_trace_to_csv(model, atom_bv, N, filename="trace.csv"):
    """
    把一條 SAT 的 trace 寫到 CSV 檔裡。

    檔案格式：
        time,atom1,atom2,...
        0,0,1,...
        1,1,1,...
        ...
    """
    atoms = sorted(atom_bv.keys())  # 固定欄位順序

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # 寫 header
        writer.writerow(["time"] + atoms)

        # 寫每一個時間點的值
        for i in range(N):
            row = [i]
            for name in atoms:
                val = model.eval(atom_bv[name][i], model_completion=True)
                row.append(1 if is_true(val) else 0)
            writer.writerow(row)

    print(f"→ trace 已輸出到 CSV 檔：{filename}")
# ------------------------------------------------------------
# 第 9 步：parser (字串 -> AST)
# ------------------------------------------------------------
#
# 我們先做兩件事：
#   1. 把字串切成 token（lexer / tokenizer）
#   2. 用遞迴下降 parser 把 token 變成 AST
#
# 支援的語法（你可以寫的公式長相）：
#   變數：
#     p, q, r1, flag_ok （英文字母開頭，可以有數字跟底線）
#   布林：
#     !phi       表示 not
#     phi & psi  and
#     phi | psi  or
#     phi -> psi implies（右結合）
#   時序：
#     F[a,b](phi)
#     G[a,b](phi)
#     U[a,b](phi, psi)
#   括號：
#     (phi)
# ------------------------------------------------------------

class Token:
    """一個最簡單的 token 結構：kind + value。"""
    def __init__(self, kind, value=None):
        # kind：token 類型，例如 "ID", "INT", "ARROW", "(", ")", "[", "]", ",", "!", "&", "|", "EOF"
        # value：對於 ID 就是字串名字，對於 INT 就是整數，其他多半是 None
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"Token({self.kind}, {self.value})"


def tokenize(s: str):
    """
    把輸入字串 s 切成一串 token。
    例： "G[0,3](p -> F[0,2](q))"
       -> [Token("ID","G"), Token("["), Token("INT",0), Token(","), ... , Token("EOF")]
    """
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]

        # 跳過空白
        if c.isspace():
            i += 1
            continue

        # 英文字母開頭 -> 識別字 (ID)
        if c.isalpha() or c == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            ident = s[i:j]
            tokens.append(Token("ID", ident))
            i = j
            continue

        # 數字 -> INT
        if c.isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            num = int(s[i:j])
            tokens.append(Token("INT", num))
            i = j
            continue

        # 箭頭運算子 "->"
        if c == '-' and i + 1 < len(s) and s[i+1] == '>':
            tokens.append(Token("ARROW"))
            i += 2
            continue

        # 單字元符號：括號、方括號、逗號、not、and、or
        if c in "()[]!,&|,":
            tokens.append(Token(c))
            i += 1
            continue

        # 其他看不懂的字元，就報錯
        raise ValueError(f"無法辨識的字元: {c}")

    # 結尾加上一個 EOF token
    tokens.append(Token("EOF"))
    return tokens
class Parser:
    """
    最簡版遞迴下降 parser。
    負責把 token 串變成我們的 AST（Var / AndF / OrF / ...）。
    """

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0  # 現在讀到第幾個 token

    # 取得目前 token
    def current(self):
        return self.tokens[self.pos]

    # 往前看 offset 個 token（預設 0 就是 current）
    def peek(self, offset=0):
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]  # 超出去就回 EOF

    # 吃掉一個 token，並檢查 kind 是否符合（如果有指定的話）
    def consume(self, kind=None):
        tok = self.current()
        if kind is not None and tok.kind != kind:
            raise ValueError(f"預期 {kind}, 但看到 {tok}")
        self.pos += 1
        return tok

    def parse(self) -> Formula:
        """
        入口：從最高優先層級（implies）開始 parse。
        """
        node = self.parse_implies()
        if self.current().kind != "EOF":
            raise ValueError(f"多出不預期的 token: {self.current()}")
        return node

    # ----------------- 語法：implies 層 -----------------
    #
    # implies 右結合：a -> b -> c = a -> (b -> c)
    def parse_implies(self) -> Formula:
        left = self.parse_or()
        while self.current().kind == "ARROW":
            self.consume("ARROW")
            right = self.parse_or()
            left = ImpliesF(left, right)
        return left

    # ----------------- 語法：or 層 -----------------

    def parse_or(self) -> Formula:
        left = self.parse_and()
        while self.current().kind == "|":
            self.consume("|")
            right = self.parse_and()
            left = OrF(left, right)
        return left

    # ----------------- 語法：and 層 -----------------

    def parse_and(self) -> Formula:
        left = self.parse_unary()
        while self.current().kind == "&":
            self.consume("&")
            right = self.parse_unary()
            left = AndF(left, right)
        return left

    # ----------------- 語法：unary 層 -----------------
    #
    # 處理：
    #   !phi
    #   F[a,b](phi)
    #   G[a,b](phi)
    #   U[a,b](phi, psi)
    #   變數
    #   (phi)
    def parse_unary(self) -> Formula:
        tok = self.current()

        # 1) not
        if tok.kind == "!":
            self.consume("!")
            phi = self.parse_unary()
            return NotF(phi)

        # 2) ID 開頭：
        #   - 可能是變數：p, q, r1
        #   - 也可能是 F/G/U（如果後面跟 [ 開頭）
        if tok.kind == "ID":
            # 看下一個 token，如果是 '[' 而且名字是 F/G/U -> 時序運算子
            if tok.value in ("F", "G", "U") and self.peek(1).kind == "[":
                name = tok.value
                self.consume("ID")   # 吃掉 F / G / U

                # 讀 [a,b]
                self.consume("[")
                a_tok = self.consume("INT")
                self.consume(",")
                b_tok = self.consume("INT")
                self.consume("]")

                a = a_tok.value
                b = b_tok.value

                # 讀 () 裡面的公式
                self.consume("(")
                if name in ("F", "G"):
                    inner = self.parse_implies()
                    self.consume(")")
                    if name == "F":
                        return F_op(a, b, inner)
                    else:
                        return G_op(a, b, inner)
                else:
                    # U[a,b](phi, psi)
                    phi = self.parse_implies()
                    self.consume(",")
                    psi = self.parse_implies()
                    self.consume(")")
                    return U_op(a, b, phi, psi)

            else:
                # 否則就是普通變數
                self.consume("ID")
                return Var(tok.value)

        # 3) 括號起頭：(phi)
        if tok.kind == "(":
            self.consume("(")
            node = self.parse_implies()
            self.consume(")")
            return node

        raise ValueError(f"parse_unary 遇到無法處理的 token: {tok}")


def parse_formula_from_string(s: str) -> Formula:
    """外部介面：從字串 s 解析出 AST。"""
    tokens = tokenize(s)
    parser = Parser(tokens)
    return parser.parse()
# ------------------------------------------------------------
# 第 10 步：從「字串公式」直接跑 SAT + trace
# ------------------------------------------------------------

def run_formula_from_string(formula_str: str, N: int, csv_filename: str | None = None):
    """
    便利函式：
      輸入：公式字串 + trace 長度 N
      動作：
        1. 解析成 AST
        2. 印出 AST
        3. 建 solver 做 SAT/UNSAT 檢查
        4. 如果 SAT，就印出一條 trace，且（如有指定）輸出 CSV
    """
    print("輸入公式字串:", formula_str)
    print("trace 長度 N =", N)

    print("\n[1] 解析字串成 AST ...")
    formula = parse_formula_from_string(formula_str)
    pretty_print(formula)

    print("\n[2] 建立 solver 並檢查 SAT/UNSAT ...")
    solver, atom_bv, root_bits = mk_solver_for(formula, N)
    res = solver.check()
    print("solver result:", res)

    if res == unsat:
        print("→ UNSAT：在長度 N 的 trace 上，這個公式不可能被滿足。")
    elif res == unknown:
        print("→ UNKNOWN：Z3 說算不出來（很少見）。")
    elif res == sat:
        print("→ SAT：找到一條 trace 讓公式在時間 0 為真。")
        m = solver.model()
        print("\n[3] 一條滿足公式的 trace：")
        show_trace(m, atom_bv, N)

        if csv_filename is not None:
            save_trace_to_csv(m, atom_bv, N, csv_filename)


# ------------------------------------------------------------
# 第 8 步：一些測試例子（主程式）
# ------------------------------------------------------------

def example_unsat():
    """
    例子 1：G[0,3] p  AND  F[0,3] not p
    在長度 N=4 的 trace 上應該 UNSAT
    """
    phi = AndF(
        G_op(0, 3, Var("p")),
        F_op(0, 3, NotF(Var("p")))
    )

    N = 4
    print("=== Example 1: G[0,3] p  AND  F[0,3] ¬p, N=4 ===")
    pretty_print(phi)

    solver, atom_bv, root_bits = mk_solver_for(phi, N)
    res = solver.check()
    print("solver result:", res)

    if res == unknown:
        print("→ Z3 回傳 unknown（通常代表需要更多設定或太難）")
    elif res == unsat:
        print("→ 公式在這個 N 下是 UNSAT（預期如此）")
    elif res == sat:
        print("→ SAT，這就奇怪了，可以印 trace 看看：")
        m = solver.model()
        show_trace(m, atom_bv, N)







def example_until():
    """
    例子 3：φ U[0,2] ψ
    簡單測試 until 的行為。
    這裡我們假設 φ = p，ψ = q：
      p U[0,2] q

    直覺上：
      - 在時間 0，看 0~2 步內是否存在 q=1
      - 且在那之前 p 一直是 1
    """
    phi = U_op(0, 2, Var("p"), Var("q"))

    N = 5
    print("\n=== Example 3: p U[0,2] q, N=5 ===")
    pretty_print(phi)

    solver, atom_bv, root_bits = mk_solver_for(phi, N)
    res = solver.check()
    print("solver result:", res)

    if res == unknown:
        print("→ Z3 回傳 unknown")
    elif res == unsat:
        print("→ 居然 UNSAT，這就怪了")
    elif res == sat:
        print("→ SAT，印出一條滿足的 trace：")
        m = solver.model()
        show_trace(m, atom_bv, N)


if __name__ == "__main__":
    print("=== MLTL 小型 SAT Solver 互動模式 ===")
    print("語法範例：")
    print("  G[0,3](p) & F[0,3](!p)")
    print("  G[0,3](p -> F[0,2](q))")
    print("  U[0,2](p, q)")
    print("輸入空行可離開。")
    print()

    while True:
        formula_str = input("請輸入公式：").strip()
        if not formula_str:
            print("結束。")
            break

        N_str = input("請輸入 trace 長度 N（例如 4）：").strip()
        try:
            N = int(N_str)
        except ValueError:
            print("N 必須是整數，請重試。\n")
            continue
        if N <= 0:
            print("N 必須 >= 1，因為 trace 至少要有一個時間點（index 0）。\n")
            continue
            

        csv_name = input("若要輸出 CSV 檔名（例如 trace.csv，空白=不要輸出）：").strip()
        if csv_name == "":
            csv_name = None

        print()
        run_formula_from_string(formula_str, N, csv_name)
        print("\n" + "-" * 60 + "\n")

