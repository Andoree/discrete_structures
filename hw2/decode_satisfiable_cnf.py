import codecs
import re
from itertools import product

import graphviz
import numpy as np

VALUE_VECTOR_STR_TO_BASIS_FUNCTION = {
    "1_1_0_1": "->",
    "1_0_0_0": "|",
}


def load_variable2id(path: str):
    variable2id = {}
    with codecs.open(path, 'r', encoding="utf-8") as inp_file:
        for line in inp_file:
            attrs = line.strip().split('\t')
            variable2id[attrs[0]] = int(attrs[1])
    return variable2id


def load_cnf_solution(path: str):
    with codecs.open(path, 'r', encoding="utf-8") as inp_file:
        inp_file.readline()
        variable_values = [int(x) for x in inp_file.readline().strip().split()][:-1]
    return variable_values


def create_nodes(dot_digraph: graphviz.Digraph, num_gates_n, output_size_m, T_matrix):
    num_gates_N = T_matrix.shape[0]
    # Создание вершин переменных
    for i in range(num_gates_n):
        dot_digraph.node(f"x_{i + 1}")
    # Создание вершин гейтов
    for i in range(num_gates_N):
        t_00 = T_matrix[i][0][0]
        t_01 = T_matrix[i][0][1]
        t_10 = T_matrix[i][1][0]
        t_11 = T_matrix[i][1][1]
        value_vector_str = f"{t_00}_{t_01}_{t_10}_{t_11}"
        node_function_str = VALUE_VECTOR_STR_TO_BASIS_FUNCTION[value_vector_str]
        dot_digraph.node(f"T_{i + 1}", node_function_str)
    # Создание вершин выходных гейтов
    for i in range(output_size_m):
        dot_digraph.node(f"y_{i + 1}")


def create_c_ikj_edges(dot_digraph: graphviz.Digraph, C_matrix):
    num_gates_N = C_matrix.shape[0]
    num_gates_n_plus_N = C_matrix.shape[2]
    num_gates_n = num_gates_n_plus_N - num_gates_N

    for (i, k, j) in product(range(num_gates_n, num_gates_n_plus_N), range(2), range(num_gates_n_plus_N)):
        c_ikj_value = C_matrix[i - num_gates_n][k][j]
        if c_ikj_value == 1:
            target_gate_id = f"T_{i - num_gates_n + 1}"
            if j < num_gates_n:
                source_gate_id = f"x_{j + 1}"
            else:
                source_gate_id = f"T_{j - num_gates_n + 1}"
            dot_digraph.edge(source_gate_id, target_gate_id)
        elif c_ikj_value == 0:
            pass
        else:
            raise ValueError(f"Invalid c_{i}_{k}_{j}: {c_ikj_value}")


def create_o_ij_edges(dot_digraph: graphviz.Digraph, num_gates_n, O_matrix):
    num_gates_N = O_matrix.shape[0]
    num_gates_n_plus_N = num_gates_n + num_gates_N
    output_size_m = O_matrix.shape[1]

    for (i, j) in product(range(num_gates_n, num_gates_n_plus_N), range(output_size_m)):
        o_ij_value = O_matrix[i - num_gates_n][j]
        if o_ij_value == 1:
            target_gate_id = f"y_{j + 1}"
            source_gate_id = f"T_{i - num_gates_n + 1}"
            dot_digraph.edge(source_gate_id, target_gate_id)


def create_graphviz_graph(C_matrix, O_matrix, T_matrix, num_gates_n, output_size_m):
    dot_digraph = graphviz.Digraph('circuit_graph', comment='Circuit graph')
    create_nodes(dot_digraph=dot_digraph, T_matrix=T_matrix, num_gates_n=num_gates_n, output_size_m=output_size_m)
    create_c_ikj_edges(dot_digraph=dot_digraph, C_matrix=C_matrix)
    create_o_ij_edges(dot_digraph=dot_digraph, num_gates_n=num_gates_n, O_matrix=O_matrix)
    dot_digraph.render(directory='graph', view=True)


def main():
    variable2id_path = "cnf/variable2id.tsv"
    input_solved_cnf = "cnf/minisat_output"
    num_gates_n = 2
    num_gates_N = 1
    output_size_m = 2
    t_ib1b2_pattern = "t_(?P<i>[0-9]+)_(?P<b1>[0-9]+)_(?P<b2>[0-9]+)"
    c_ikj_pattern = "c_(?P<i>[0-9]+)_(?P<k>[0-9]+)_(?P<j>[0-9]+)"
    o_ij_pattern = "o_(?P<i>[0-9]+)_(?P<j>[0-9]+)"
    v_it_pattern = "v_(?P<i>[0-9]+)_(?P<t>[0-9]+)"

    T_matrix = np.zeros(shape=(num_gates_N, 2, 2), dtype=int)  # 16
    C_matrix = np.zeros(shape=(num_gates_N, 2, num_gates_n + num_gates_N), dtype=int)  # 4 * 2 * 8 = 64
    V_matrix = np.zeros(shape=(num_gates_n + num_gates_N, 2 ** num_gates_n), dtype=int)  # 16 * 16 = 256
    O_matrix = np.zeros(shape=(num_gates_N, output_size_m), dtype=int)  # 8

    variable2id = load_variable2id(path=variable2id_path)
    id2variable = {idx: v for v, idx in variable2id.items()}
    variable_values = load_cnf_solution(path=input_solved_cnf)
    for val in variable_values:
        sign = 1 if val > 0 else -1
        val_id = abs(val)
        variable_str = id2variable[val_id]

        if variable_str.startswith('c'):
            m = re.fullmatch(c_ikj_pattern, variable_str)
            i = int(m.group("i")) - num_gates_n
            k = int(m.group("k"))
            j = int(m.group("j"))
            if sign == 1:
                C_matrix[i][k][j] = 1
        elif variable_str.startswith('t'):
            m = re.fullmatch(t_ib1b2_pattern, variable_str)
            i = int(m.group("i")) - num_gates_n
            b1 = int(m.group("b1"))
            b2 = int(m.group("b2"))
            if sign == 1:
                T_matrix[i][b1][b2] = 1
        elif variable_str.startswith('o'):
            m = re.fullmatch(o_ij_pattern, variable_str)
            i = int(m.group("i")) - num_gates_n
            j = int(m.group("j"))
            if sign == 1:
                O_matrix[i][j] = 1
        elif variable_str.startswith('v'):
            m = re.fullmatch(v_it_pattern, variable_str)
            i = int(m.group("i"))
            t = int(m.group("t"))
            if sign == 1:
                V_matrix[i][t] = 1
    create_graphviz_graph(C_matrix=C_matrix, O_matrix=O_matrix, T_matrix=T_matrix, num_gates_n=num_gates_n,
                          output_size_m=output_size_m)


if __name__ == '__main__':
    main()
