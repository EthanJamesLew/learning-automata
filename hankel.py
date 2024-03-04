from automata.fa.dfa import DFA

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_hankel(ax, H: np.ndarray, prefixes: List[str], suffixes: List[str]):
    unique_rows, indices = np.unique(H, axis=0, return_inverse=True)
    colors = plt.cm.get_cmap('Pastel1', len(unique_rows))

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


def hankel_to_dfa(H:np.ndarray, prefixes: List[str], suffixes: List[str]) -> DFA:
    unique_rows, indices = np.unique(H, axis=0, return_inverse=True)
    states = set(indices)
    input_symbols = set(''.join(prefixes+suffixes))
    initial_state = indices[0]
    final_states = set((np.where(unique_rows[:, 0]==True)[0]).tolist())

    # collect transitions
    transitions = []
    for pi, p in enumerate(prefixes):
        if len(p)==1:
            transitions.append((indices[0], p, indices[pi]))
        elif len(p) > 1:
            pre = p[:-1]
            idx = prefixes.index(pre)
            transitions.append((indices[idx], p[-1], indices[pi]))
        else:
            pass

    # reformat
    transition_dict = {k: {} for k in states}
    for current_state, symbol, next_state in transitions:
        current_state_key = current_state
        next_state_key = next_state

        if current_state_key not in transition_dict:
            transition_dict[current_state_key] = {}

        transition_dict[current_state_key][symbol] = next_state_key

    return DFA(
        states=states,
        input_symbols=input_symbols,
        transitions=transition_dict,
        initial_state=initial_state,
        final_states=final_states,
        allow_partial=True
    )