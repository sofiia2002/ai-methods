import sympy as sp

def sigmoid_act():
    x = sp.Symbol('x')
    f = 1 / (1 + sp.exp(-x))
    return f

def tanh_act():
    x = sp.Symbol('x')
    f = (sp.exp(x) - sp.exp(-x)) / (sp.exp(x) + sp.exp(-x))
    return f

def relu_act():
    x = sp.Symbol('x')
    f = sp.Max(0, x)
    return f
