## Automata Learning

### Sources
These notes capture information from [Borja Balle's Lecture](https://www.youtube.com/watch?v=g-5PPYDiL2k) and [Angluin's L* Paper](https://people.eecs.berkeley.edu/~dawnsong/teaching/s10/papers/angluin87.pdf).

### Automata Learning: A Timeline

* 1967: Gold: Regular languages are learnable in the limit
* 1987: Angluin:  Regular languages are learnable from queries
* 1993: Pitt and Warmuth: PAC-Learning DFA is NP-hard
* 90's, 00's: Combinatorial methods meet statistics and linear algebra
* 2009: Spectral Learning

### Method Categories

* **Exact Learning**
  * Hankel Trick for Deterministic Automata
  * Angluin's L* Algorithm 
* **PAC Learning**
  * Hankel Trick for Weighted Automata
  * Spectral Learning 
* **Statistical Learning**
  *  Hankel Matrix Completion 

```python
# imports
from automata.fa.dfa import DFA
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='retina'
```

## From DFA to Hankel Matrix

```python
# let's make an example automata to learn
dfa1 = DFA(
    states={'q0', 'q1', 'q2'},
    input_symbols={'a', 'b'},
    transitions={
        'q0': {'b': 'q0', 'a': 'q1'},
        'q1': {'a': 'q1', 'b': 'q2'},
        'q2': {'a': 'q0', 'b': 'q1'}
    },
    initial_state='q0',
    final_states={'q1'}
)
# plot it
dfa1.show_diagram()
```

```python
# select prefixes and suffixes for the Hankel Block Matrix
prefixes = ['', 'a', 'b', 'aa', 'ab', 'ba', 'bb']
suffixes = prefixes

# build a Hankel Block Matrix
H = np.zeros((len(prefixes), len(suffixes)), dtype=bool)
for pi, p in enumerate(prefixes):
    for si, s in enumerate(suffixes):
        H[pi, si] = dfa1.accepts_input(p + s)
```

```python
# Create a color map based on unique rows
unique_rows, indices = np.unique(H, axis=0, return_inverse=True)
colors = plt.cm.get_cmap('Pastel1', len(unique_rows))

# Plot the matrix
fig, ax = plt.subplots()
for (i, j), val in np.ndenumerate(H):
    ax.text(j, len(H) - i - 1, int(val), ha='center', va='center', color='black')
    ax.add_patch(plt.Rectangle((j-0.5, (len(H) - i - 1)-0.5), 1, 1, fill=True, color=colors(indices[i])))

ax.set_xticks(np.arange(-0.5, len(H[0]), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(H), 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
ax.set_xticks(np.arange(len(H[0])))
ax.set_yticks(np.arange(len(H)))
ax.set_xticklabels(suffixes)
ax.set_yticklabels(prefixes[::-1])
ax.xaxis.set_ticks_position('top')
ax.set_title("Hankel Matrix (Unique Rows are Highlighted)")
plt.show()
```

## From Hankel Matrix to DFA


## Angluin's Algorithm (L*)

```python

```
