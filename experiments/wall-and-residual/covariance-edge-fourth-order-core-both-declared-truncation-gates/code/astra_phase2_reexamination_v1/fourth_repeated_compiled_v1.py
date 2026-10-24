"""Canonical elementwise compilation of non-third AUG211 repeated terms.

This is exact algebra, no rank selection, targets, or fitted constants.
Production can bake compile_terms()'s finite recipe; JSON enumeration is
an offline compiler input, never an estimator-side external dependency.
"""
from collections import defaultdict
from itertools import combinations,product
from pathlib import Path
import json
import numpy as np


def compile_terms():
    terms=json.loads(Path(__file__).with_name('thin_fourth_primary_terms_v1.json').read_text())['terms']
    polynomial=defaultdict(float)
    def expansion(block):
        axes=tuple(i for i,x in enumerate(block) if x)
        vec=tuple(x for x in block if x)
        if len(axes)<=2:
            if len(axes)==1:
                return [(1.,[(axes,{1:'mean',2:'v',3:'k3',4:'k4'}[sum(vec)])])]
            label={(1,1):'C',(2,1):'K21',(1,2):'K21T',(2,2):'K22',(3,1):'K31',(1,3):'K31T'}[vec]
            return [(1.,[(axes,label)])]
        legs=tuple(i for i,x in enumerate(block) for _ in range(x)); answer=[]
        assert sum(block)==4
        for chosen in combinations(range(4),2):
            other=tuple(i for i in range(4) if i not in chosen)
            answer.append((1/6,[((legs[pair[0]],legs[pair[1]]),label) for pair,label in ((chosen,'A'),(other,'B'))]))
        return answer
    diagonal={'C':'v','K21':'k3','K21T':'k3','K22':'k4','K31':'k4','K31T':'k4','A':'ad','B':'bd'}
    transpose={'K21':'K21T','K21T':'K21','K31':'K31T','K31T':'K31'}
    for term in terms:
        for summands in product(*(expansion(b) for b in term['blocks'])):
            coefficient=term['coefficient']; factors=[]
            for c,f in summands:coefficient*=c;factors+=f
            for collapse in ((0,0,1),(1,0,1)):
                vectors=[[],[]]; matrices=[]
                for axis,label in enumerate(('h'+str(term['ks'][0]),'w'+str(term['ks'][1]),'w'+str(term['ks'][2]))):
                    vectors[collapse[axis]].append(label)
                for axes,label in factors:
                    mapped=tuple(collapse[a] for a in axes)
                    if len(mapped)==1:vectors[mapped[0]].append(label)
                    elif mapped[0]==mapped[1]:vectors[mapped[0]].append(diagonal[label])
                    else:matrices.append(transpose.get(label,label) if mapped==(1,0) else label)
                key=(tuple(sorted(vectors[0])),tuple(sorted(vectors[1])),tuple(sorted(matrices)))
                polynomial[key]+=coefficient
    grouped=defaultdict(list)
    for (left,right,matrices),coefficient in sorted(polynomial.items()):
        grouped[matrices].append((coefficient,left,right))
    return dict(grouped)


RECIPE=compile_terms()


def repeated(C,K21,K22,K31,A,B,h,w):
    values=dict(C=C,K21=K21,K21T=K21.T,K22=K22,K31=K31,K31T=K31.T,A=A,B=B,
                v=np.diag(C),k3=np.diag(K21),k4=np.diag(K22),ad=np.diag(A),bd=np.diag(B))
    values.update({'h'+str(i):x for i,x in enumerate(h)})
    values.update({'w'+str(i):x for i,x in enumerate(w)})
    cache={}
    def multiply(labels):
        if labels not in cache:
            value=values[labels[0]]
            for label in labels[1:]:value=value*values[label]
            cache[labels]=value
        return cache[labels]
    answer=np.zeros_like(C)
    for matrices,terms in RECIPE.items():
        subtotal=np.zeros_like(C)
        for coefficient,left,right in terms:
            # Apply scalar to a vector rather than a full matrix.
            subtotal+=(coefficient*multiply(left))[:,None]*multiply(right)[None,:]
        answer+=subtotal*multiply(matrices) if matrices else subtotal
    np.fill_diagonal(answer,0)
    return answer
