import numpy as np
import sympy as sp

def get_derivative(func):
  gradient = []
  variables = sorted(func.free_symbols, key = lambda symbol: symbol.name)
  for variable in variables:
    gradient.append(sp.diff(func, variable))
  return gradient

def calc_derivative(func, value):
  func_gradient = get_derivative(func())
  variables = sorted(func().free_symbols, key = lambda symbol: symbol.name)
  gradient_value = func_gradient[0].evalf(subs = dict(zip(variables,np.array([value]))))
  return gradient_value