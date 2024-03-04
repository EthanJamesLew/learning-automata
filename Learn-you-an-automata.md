## Automata Learning

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
 
### Resources
These notes capture information from [Borja Balle's Lecture](https://www.youtube.com/watch?v=g-5PPYDiL2k) and [Angluin's L* Paper](https://people.eecs.berkeley.edu/~dawnsong/teaching/s10/papers/angluin87.pdf).


## Math Preliminaries

Let $\Sigma$ be an alphabet and we use $\Sigma^*$ to denote a set of all strings strings over $\Sigma$. We use $\epsilon$ to denote the empty string.

A *deterministic finite automaton (DFA)* is a tuple $\text{DFA}(A) = (\Sigma, Q, q_0, \tau, \phi)$
* $\Sigma$ is a set of input symbols
* $Q$ is a set of states
* $\tau: Q \times \Sigma \rightarrow Q$ is a transition function
* $q_0$ is the initial state
* $\phi$ is a set of accepting states

A *Hankel Matrix* $H_f \in \mathbb R^{\Sigma^* \times \Sigma^*}$ is a matrix defined by $H_f(p, s) = f(p \cdot s)$, where $f: \Sigma^* \rightarrow \mathbb R$. In this discussion, $H$ is shorthand for a Hankel matrix $H_m$ associated with some implied DFA or Teacher where $m$ is the membership query. 

**Myhill-Nerod Theorem:** The number of distinct rows of a binary Hankel matrix $H$ equals the minimal number of states of a DFA recognizing the language of $H$.

```python
# imports
from automata.fa.nfa import NFA
from automata.fa.dfa import DFA
from automata.base.automaton import Automaton

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format='retina'

from hankel import *
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
    final_states={'q1'},
    allow_partial=True
)
# plot it
dfa1.show_diagram()
```

```python
# select prefixes and suffixes for the Hankel Block Matrix
# This needs to be closed and consistent!
prefixes = ['', 'a', 'b', 'aa', 'ab', 'ba', 'bb', 'aba', 'abb']
suffixes = prefixes

# build a Hankel Block Matrix
H = np.zeros((len(prefixes), len(suffixes)), dtype=bool)
for pi, p in enumerate(prefixes):
    for si, s in enumerate(suffixes):
        H[pi, si] = dfa1.accepts_input(p + s)
```

```python
# Plot the matrix
fig, ax = plt.subplots()
plot_hankel(ax, H, prefixes, suffixes)
plt.show()
```

## From Hankel Matrix to DFA (Hankel Trick Automata Learning)

```python
# automata learning via the "Hankel Trick"
dfa_learned = hankel_to_dfa(H, prefixes, suffixes)
dfa_learned.show_diagram()
```

## Angluin's Algorithm (L*)


### Algorithm Description

1. Initialize prefixes and suffixes: $P = \{\epsilon\}, S=\{\epsilon\}$
2. Maintain a Hankel block $H$ for $P' = P \cup P \Sigma$ and $S$ using membership queries $f$.
3. Repeat
  * While $H$ is not closed and consistent
    *  if $H$ is not closed, add a new prefix $P \Sigma$ to $P$
    *  if $H$ is not consistent, add a distinguishing suffix to $S$
  * Construct a DFA $A$ from $H$ and ask the teacher an equivalence query
    * if yes, terminate
    * Otherwise, add all prefixes of counter-example $x$ to $P$
   
### Complexity

$\mathcal O (n)$ equivalence queries and $\mathcal O(|\Sigma| n^2 L)$ MQs.

Both EQs and MQs are needed for a polynomial time algorithm. Otherwise, exponential time is required.


```python
import abc

class Teacher(abc.ABC):
    """Angluin's Minimally Adequate Teacher"""
    @abc.abstractmethod
    def query_membership(self, s: str):
        """membership equivalence is done by running the DFA"""
        pass

    @abc.abstractmethod
    def query_equivalence(self, d: DFA):
        """query equivalence is done by enumerating words up to a sufficient length 
        and looking at the symmetric difference
        """
        pass


class DFATeacher(Teacher):
    """teacher that implements queries from a hidden DFA"""
    def __init__(self, dfa: DFA, k=4):
        self.dfa = dfa
        self.k = k
        self.words = set(self.dfa.words_of_length(k))

    def query_membership(self, s: str):
        """membership equivalence is done by running the DFA"""
        return self.dfa.accepts_input(s)

    def query_equivalence(self, d: DFA):
        """query equivalence is done by enumerating words up to a sufficient length 
        and looking at the symmetric difference
        """
        words = set(d.words_of_length(self.k))
        counterexamples = list(self.words.symmetric_difference(words))
        if len(counterexamples) == 0:
            return None
        return counterexamples[0]


class RegexTeacher(DFATeacher):
    """teacher that implements queries from a regex"""
    def __init__(self, regex: str, k=4):
        nfa = NFA.from_regex(regex)
        dfa = DFA.from_nfa(nfa)
        super().__init__(dfa, k=k)

    def query_membership(self, s: str):
        """membership equivalence is done by running the DFA"""
        return self.dfa.accepts_input(s)

    def query_equivalence(self, d: DFA):
        """query equivalence is done by enumerating words up to a sufficient length 
        and looking at the symmetric difference
        """
        words = set(d.words_of_length(self.k))
        counterexamples = list(self.words.symmetric_difference(words))
        if len(counterexamples) == 0:
            return None
        return counterexamples[0]
```

```python
# Lazy Man's L*

def angluins_algorithm(teacher: Teacher, max_iter=100):
    """Angluin's L* Algorithm"""
    def build_hankel(prefixes, suffixes, teacher):
        """build Hankel matrix via Membership queries"""
        H = np.zeros((len(prefixes), len(suffixes)), dtype=bool)
        for pi, p in enumerate(prefixes):
            for si, s in enumerate(suffixes):
                H[pi, si] = teacher.query_membership(p + s)
        return H

    # start with empty prefixes and suffixes
    prefixes = ['']
    suffixes = ['']

    # iteratively build the "observation table" (i.e., Hankel Block)
    for i in range(max_iter):
        # build Hankel requires membership queries
        H = build_hankel(prefixes, suffixes, teacher)
        d = hankel_to_dfa(H, prefixes, suffixes)

        # now, use equivalent query to get counterexample
        counterexample = teacher.query_equivalence(d)

        # an equivalent DFA has been learned
        if counterexample is None:
            break
    
        # Update prefixes for closedness
        new_prefixes = set(prefixes)
        for j in range(len(counterexample) + 1):
            prefix = counterexample[:j]
            if prefix not in prefixes:
                new_prefixes.add(prefix)

        # Update suffixes for consistency
        new_suffixes = set(suffixes)
        for j in range(1, len(counterexample) + 1):
            suffix = counterexample[-j:]
            if suffix not in suffixes:
                new_suffixes.add(suffix)

        prefixes = sorted(list(new_prefixes))
        suffixes = sorted(list(new_suffixes))
        
    return d, H, prefixes, suffixes

# run the algorithm
teacher = DFATeacher(dfa1)
d, H, p, s = angluins_algorithm(teacher)
d.show_diagram()
```

```python
# Plot the Hankel matrix used in the L* observation table
fig, ax = plt.subplots()
plot_hankel(ax, H, p, s)
ax.set_title("L* Observation Table")
plt.show()
```

### Learning DFA of Regex

```python
rteacher = RegexTeacher('(a|b)*abb', k=10)
d, _, _, _ = angluins_algorithm(rteacher)
d.show_diagram()
```

```python
rteacher = RegexTeacher('((0|1|2)+(a|b)*a)+', k=10)
d, _, _, _ = angluins_algorithm(rteacher)
d.show_diagram()
```

```python

```
