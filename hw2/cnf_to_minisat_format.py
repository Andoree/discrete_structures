# TODO

# c_i_k_j (n <= i < n + N − 1, 0 <= k < 2, 0 <= j < n + N)
import codecs
import itertools
from itertools import product
from typing import Dict, List, Tuple


# def initialize_variables(num_gates_n: int, num_gates_N: int, output_size_m):
#     c_ikj_variables_dict = {}
#     o_ij_variables_dict = {}
#     t_ib1b2_variables_dict = {}
#     v_it_variables_dict = {}
#
#     num_gates_N_range = range(num_gates_N)
#     num_gates_n_range = range(num_gates_n)
#     two_exp_n_range = range(2 ** num_gates_n)
#     bit_range = range(2)
#     num_gates_n_plus_N_range = range(num_gates_n + num_gates_N)
#     m_range = range(output_size_m)
#     # todo: ТЭШКИ
#     for (i, k, j) in product(num_gates_N_range, bit_range, num_gates_n_plus_N_range):
#         c_ikj_variables_dict[f"c_{i}_{k}_{j}"] = Variable(f"c_{i}_{k}_{j}")
#     for (i, j) in product(num_gates_n_range, m_range):
#         o_ij_variables_dict[f"o_{i}_{j}"] = Variable(f"o_{i}_{j}")
#     for (i, t) in product(num_gates_N_range, two_exp_n_range):
#         v_it_variables_dict[f"v_{i}_{t}"] = Variable(f"v_{i}_{t}")


def create_t_ib1b2_clauses(num_gates_n: int, num_gates_N: int, disjunctions_list):
    i_range = range(num_gates_n, num_gates_N + num_gates_n)
    for i in i_range:
        # clauses = (f"t_{i}_0_0", f"-t_{i}_1_0", f"-t_{i}_0_1 | -t_{i}_1_1", f"t_{i}_0_1 | t_{i}_1_1")
        # TODO: Тут проверить скобки, почему здесь по два отрицания?
        clauses = ((f"t_{i}_0_0", ), (f"-t_{i}_1_0", ),
                   (f"-t_{i}_0_1", f"t_{i}_1_1"), (f"t_{i}_0_1", f"-t_{i}_1_1"))
        disjunctions_list.extend(clauses)


def create_c_ikj_clauses(num_gates_n: int, num_gates_N: int, disjunctions_list):
    # TODO: Тут возможно вычесть единицу
    i_range = range(num_gates_n, num_gates_n + num_gates_N)
    k_range = range(2)
    j_range = range(num_gates_N + num_gates_n)
    for (i, k) in product(i_range, k_range):
        # existence_cond_variables = ' | '.join((f"c_{i}_{k}_{j}" for j in j_range))
        existence_cond_variables = tuple(f"c_{i}_{k}_{j}" for j in j_range)
        disjunctions_list.append(existence_cond_variables)
        for j_1 in j_range:
            for j_2 in range(j_1 + 1, num_gates_N + num_gates_n):
                if j_1 < j_2:
                    # disjunction_clause = f"-c_{i}_{k}_{j_1} | -c_{i}_{k}_{j_2}"
                    disjunction_clause = (f"-c_{i}_{k}_{j_1}", f"-c_{i}_{k}_{j_2}")
                    disjunctions_list.append(disjunction_clause)


def create_o_ij_clauses(num_gates_n: int, num_gates_N: int, output_size_m: int, disjunctions_list):
    i_range = range(num_gates_n, num_gates_n + num_gates_N)
    j_range = range(output_size_m)
    for j in j_range:
        # existence_cond = " | ".join((f"o_{i}_{j}" for i in i_range))
        existence_cond = tuple(f"o_{i}_{j}" for i in i_range)
        disjunctions_list.append(existence_cond)
        for i_1 in i_range:
            for i_2 in range(i_1 + 1, num_gates_n + num_gates_N):
                if i_1 < i_2:
                    # disjunction_clause = f"-o_{i_1}_{j} | -o_{i_2}_{j}"
                    disjunction_clause = (f"-o_{i_1}_{j}", f"-o_{i_2}_{j}")
                    disjunctions_list.append(disjunction_clause)


def create_v_it_input_clauses(num_gates_n: int, input_sets, disjunctions_list):
    # TODO: везде проверить рейнджы. Где-то диапазоны не от нуля, а от эн малого до эн большого
    i_range = range(num_gates_n)
    t_range = range(2 ** num_gates_n)
    for (i, t) in product(i_range, t_range):
        # TODO: Запомнить, что инвертировал i t на t i
        input_value = input_sets[t][i]
        sign = '' if input_value == 1 else '-'
        clause = (f"{sign}v_{i}_{t}", )
        disjunctions_list.append(clause)


def create_six_clauses(num_gates_n: int, num_gates_N: int, output_size_m: int, values, disjunctions_list):
    i_range = range(num_gates_n, num_gates_N + num_gates_n)
    t_range = range(2 ** num_gates_n)
    bit_range = range(output_size_m)
    for (i, r, i_0, i_1) in product(i_range, t_range, bit_range, bit_range):
        for j_0 in range(num_gates_n, i):
            for j_1 in range(j_0 + 1, i):
                i_0_sign = '-' if i_0 == 1 else ''
                i_1_sign = '-' if i_1 == 1 else ''

                clause_1 = (f"-c_{i}_{0}_{j_0}", f"-c_{i}_{1}_{j_1}", f"{i_0_sign}v_{j_0}_{r}",
                            f"{i_1_sign}v_{j_1}_{r}", f"v_{i}_{r}", f"-t_{i}_{i_0}_{i_1}")
                clause_2 = (f"-c_{i}_{0}_{j_0}", f"-c_{i}_{1}_{j_1}", f"{i_0_sign}v_{j_0}_{r}",
                            f"{i_1_sign}v_{j_1}_{r}", f"-v_{i}_{r}", f"t_{i}_{i_0}_{i_1}")
                # f"(v_{i}{r} >> t_{i}_{i_0}_{i_1}) & (t_{i}_{i_0}_{i_1} >> v_{i}{r})")
                # f"(-v_{i}{r} | t_{i}_{i_0}_{i_1}) & (v_{i}{r} | -t_{i}_{i_0}_{i_1})"
                disjunctions_list.append(clause_1)
                disjunctions_list.append(clause_2)

                # TODO: условие не дописано


def create_output_check_clauses(num_gates_n: int, num_gates_N: int, output_size_m: int, values, disjunctions_list):
    # TODO: везде проверить рейнджы. Где-то диапазоны не от нуля, а от эн малого до эн большого
    i_range = range(num_gates_n, num_gates_N + num_gates_n)
    r_range = range(2 ** num_gates_n)
    k_range = range(output_size_m)
    for (i, r, k) in product(i_range, r_range, k_range):
        # TODO: Запомнить, что тут инвертировал индексы
        value = values[r][k]
        sign = '' if value == 1 else '-'
        clause = (f"-o_{i}_{k}", f"{sign}v_{i}_{r}")
        disjunctions_list.append(clause)


def create_clauses(num_gates_n, num_gates_N, output_size_m, output_values):
    input_sets = list(itertools.product((0, 1), repeat=num_gates_n))

    clauses_list = []
    create_t_ib1b2_clauses(num_gates_n=num_gates_n, num_gates_N=num_gates_N, disjunctions_list=clauses_list)
    create_c_ikj_clauses(num_gates_n=num_gates_n, num_gates_N=num_gates_N, disjunctions_list=clauses_list)
    create_o_ij_clauses(num_gates_n=num_gates_n, num_gates_N=num_gates_N, output_size_m=output_size_m,
                        disjunctions_list=clauses_list)
    create_v_it_input_clauses(num_gates_n=num_gates_n, input_sets=input_sets, disjunctions_list=clauses_list)
    create_six_clauses(num_gates_n=num_gates_n, num_gates_N=num_gates_N, output_size_m=output_size_m,
                       disjunctions_list=clauses_list, values=output_values)
    create_output_check_clauses(num_gates_n=num_gates_n, num_gates_N=num_gates_N, output_size_m=output_size_m,
                                disjunctions_list=clauses_list, values=output_values)
    print(len(clauses_list))
    return clauses_list


def create_variables_to_id_dict(clauses: List[Tuple[str]]):
    variables = set()
    for cl in clauses:
        for term in cl:
            variables.add(term.strip('-'))
    variable2id = {var: i + 1 for i, var in enumerate(variables)}
    return variable2id


def create_dimacs_cnf(clauses, variable2id, output_path: str):
    num_clauses = len(clauses)
    num_variables = len(variable2id.keys())
    with codecs.open(output_path, 'w+', encoding="utf-8") as out_file:
        out_file.write(f"p cnf {num_variables} {num_clauses}\n")
        for cl in clauses:
            terms_ids_list = []
            for term in cl:
                sign = '-' if term.startswith('-') else ''
                var = term.strip('-')
                terms_ids_list.append(f"{sign}{variable2id[var]}")
            out_file.write(' '.join(terms_ids_list))
            out_file.write(' 0\n')


def main():
    num_gates_n = 4
    num_gates_N = 4
    output_size_m = 2
    output_values = [(0, 1) for i in range(2 ** num_gates_n)]
    var2id_output_path = "cnf/variable2id.tsv"
    dimacs_cnf_output_path = "cnf/dimacs.cnf"
    clauses = create_clauses(num_gates_n, num_gates_N, output_size_m, output_values)
    for cl in clauses:
        print(cl)
    variable2id = create_variables_to_id_dict(clauses)
    for k, v in variable2id.items():
        print(k, v)
    create_dimacs_cnf(clauses=clauses, variable2id=variable2id, output_path=dimacs_cnf_output_path)
    with codecs.open(var2id_output_path, 'w+', encoding="utf-8") as out_file:
        for var, idx in variable2id.items():
            out_file.write(f"{var}\t{idx}\n")


if __name__ == '__main__':
    main()
