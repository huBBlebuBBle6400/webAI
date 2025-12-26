# local_math_physics_ai.py
from dotenv import load_dotenv
import os
import openai

load_dotenv()  # loads .env file
openai.api_key = os.getenv("xxx")
import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import re
import warnings
import sys
warnings.filterwarnings("ignore")

# ========== CORE FUNCTIONS ==========

def solve_expression(expr_str, symbol='x'):
    x = sp.Symbol(symbol)
    if '=' in expr_str:
        lhs, rhs = expr_str.split('=', 1)
        eq = sp.Eq(parse(lhs), parse(rhs))
        sol = sp.solve(eq, x)
        return eq, sol
    else:
        expr = parse(expr_str)
        return expr, sp.simplify(expr)

def parse(s: str):
    return sp.sympify(s, evaluate=True)

def factorial(n):
    return sp.factorial(n)

def combinations(n, k):
    return sp.binomial(n, k)

def gcd(a, b):
    return sp.gcd(a, b)

def is_prime(n):
    return sp.isprime(n)

def matrix_operations(expr):
    try:
        mat = sp.Matrix(eval(expr))
        return mat, {
            "determinant": mat.det(),
            "inverse": mat.inv() if mat.det() != 0 else "Not invertible",
            "eigenvalues": mat.eigenvals(),
            "rank": mat.rank(),
            "reduced_row_echelon": mat.rref()[0]
        }
    except Exception as e:
        return None, str(e)

# ========== EXPLANATION ENGINE ==========

def explain_solution(user_input, expr=None, sol=None, operation="solve"):
    if operation == "solve":
        if sol is None:
            return f"We attempted to solve: {user_input}. No solutions found."
        return f"We solved the equation {expr} = 0.\nThe solution(s) are: {sol}."
    elif operation == "simplify":
        return f"The expression {expr} was simplified to: {sol}."
    elif operation == "physics":
        return f"Physics result for {user_input}: {sol}"
    elif operation == "matrix":
        return f"Matrix result for {user_input}:\n{sol}"
    else:
        return f"Computed result: {sol}"

# ========== PHYSICS FUNCTIONS ==========

def solve_kinematics(u=None, a=None, t=None, s=None, v=None):
    t_sym, u_sym, a_sym, s_sym, v_sym = sp.symbols('t u a s v')
    if s is None and u is not None and t is not None and a is not None:
        expr = sp.Eq(s_sym, u_sym * t_sym + (1/2) * a_sym * t_sym**2)
        sol = sp.solve(expr.subs({u_sym: u, a_sym: a, t_sym: t}), s_sym)
        return expr, sol
    if v is None and u is not None and a is not None and t is not None:
        expr = sp.Eq(v_sym, u_sym + a_sym * t_sym)
        sol = sp.solve(expr.subs({u_sym: u, a_sym: a, t_sym: t}), v_sym)
        return expr, sol
    return None, "Invalid or incomplete kinematics input."

def solve_newtons_second_law(m=None, a=None, f=None):
    m_sym, a_sym, f_sym = sp.symbols('m a F')
    if f is None and m is not None and a is not None:
        expr = sp.Eq(f_sym, m_sym * a_sym)
        sol = sp.solve(expr.subs({m_sym: m, a_sym: a}), f_sym)
        return expr, sol
    return None, "Invalid or incomplete Newton's law input."

def solve_ohms_law(v=None, i=None, r=None):
    v_sym, i_sym, r_sym = sp.symbols('V I R')
    if v is None and i is not None and r is not None:
        expr = sp.Eq(v_sym, i_sym * r_sym)
        sol = sp.solve(expr.subs({i_sym: i, r_sym: r}), v_sym)
        return expr, sol
    return None, "Invalid or incomplete Ohm's law input."

# ========== PLOTTING ==========

def plot(expr_str, var='x', domain=(-10, 10), points=300):
    expr = parse(expr_str)
    sym = sp.Symbol(var)
    f = sp.lambdify(sym, expr, modules=['numpy'])

    xs = np.linspace(domain[0], domain[1], points)
    ys = f(xs)

    plt.figure(figsize=(8, 4))
    plt.plot(xs, ys, label=f"${sp.latex(expr)}$")
    plt.xlabel(var); plt.ylabel("f(" + var + ")")
    plt.title("Function Plot")
    plt.grid(True)
    plt.legend()
    plt.show()

# ========== MAIN INTERFACE ==========

def main():
    print("\n📘 Local Math & Physics Assistant")
    print("Type 'exit' to quit.")
    sys.stdout.flush()

    while True:
        user = input("\nEnter a math/physics problem: ").strip()
        if user.lower() == 'exit':
            print("Goodbye!")
            break

        try:
            if user.startswith("plot "):
                expr = user[5:].strip()
                try:
                    plot(expr)
                except Exception as e:
                    print("Plotting error:", e)

            elif '!' in user:
                num = int(user.replace('!', '').strip())
                sol = factorial(num)
                print(explain_solution(user, sol=sol, operation="simplify"))

            elif user.startswith("C("):
                match = re.match(r"C\((\d+),(\d+)\)", user)
                if match:
                    n, k = map(int, match.groups())
                    sol = combinations(n, k)
                    print(explain_solution(user, sol=sol, operation="simplify"))

            elif "gcd" in user:
                a, b = map(int, re.findall(r'\d+', user))
                sol = gcd(a, b)
                print(explain_solution(user, sol=sol, operation="simplify"))

            elif "prime" in user:
                n = int(re.search(r'\d+', user).group())
                sol = is_prime(n)
                print(explain_solution(user, sol=sol, operation="simplify"))

            elif user.startswith("matrix"):
                expr = user[len("matrix"):].strip()
                mat, result = matrix_operations(expr)
                print(explain_solution(user, sol=result, operation="matrix"))

            elif "kinematics" in user:
                values = dict(re.findall(r'(\b[a-z])=(\d+(?:\.\d+)?)', user))
                values = {k: float(v) for k, v in values.items()}
                expr, sol = solve_kinematics(**values)
                print(explain_solution(user, expr=expr, sol=sol, operation="physics"))

            elif "newton" in user:
                values = dict(re.findall(r'(\b[a-z])=(\d+(?:\.\d+)?)', user))
                values = {k: float(v) for k, v in values.items()}
                expr, sol = solve_newtons_second_law(**values)
                print(explain_solution(user, expr=expr, sol=sol, operation="physics"))

            elif "ohm" in user:
                values = dict(re.findall(r'(\b[a-z])=(\d+(?:\.\d+)?)', user))
                values = {k: float(v) for k, v in values.items()}
                expr, sol = solve_ohms_law(**values)
                print(explain_solution(user, expr=expr, sol=sol, operation="physics"))

            else:
                expr, sol = solve_expression(user)
                print("🧮 Parsed:", expr)
                print("✅", explain_solution(user, expr=expr, sol=sol))

        except Exception as e:
            print("❌ Error:", e)

if __name__ == '__main__':
    from IPython import get_ipython
    if get_ipython():
        main()

# --- [6] GPT-3.5 Natural Language Processing ---
# --- [1] Imports ---
import streamlit as st
import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("xxx")

# --- [3] Streamlit UI ---
st.title("🧠 Math & Physics AI Assistant")
st.image("logo.png", caption="Powered by GPT-3.5 + SymPy", use_column_width=True)

# Chat input
user_input = st.chat_input("Ask a math or physics question in plain English:")

import openai

def ask_gpt(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful math and physics assistant that explains answers clearly."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ GPT Error: {e}"

# --- [5] Display result ---
if user_input:
    st.info(f"📝 You asked: {user_input}")
    gpt_reply = ask_gpt(user_input)
    st.success("✅ GPT-3.5 says:")
    st.markdown(gpt_reply)
