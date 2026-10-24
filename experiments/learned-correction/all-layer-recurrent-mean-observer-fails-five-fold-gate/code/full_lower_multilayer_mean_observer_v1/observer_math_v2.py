"""Stable evaluation of the same normalized source columns before transport."""
from observer_math import raw_design,local_design

def normalized_design(raw,sigma,gate,weights,mean,scale,transport):
    normalized=(raw-mean)/scale
    return raw_design(normalized,sigma,gate,weights) if transport else local_design(normalized,sigma)
