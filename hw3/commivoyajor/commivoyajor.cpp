#include <stdio.h>
#include <omp.h>
#include <iostream>
#include <time.h>
#include <random>
#include <ctime>
#include <set>
#include <list>

using namespace std;

struct Result { int subtree_cost; list<int> subtree_nodes; };

void modify_struct(Result &r, int cost, list<int>nodes) {
    r.subtree_cost = cost;
    r.subtree_nodes = nodes;
}
int calculate_path_cost_estimation(int num_nodes, int* adj_matrix, set<int> not_visited_nodes, int source_id, bool parallelize)
{   
    int estimated_cost = 0;
    if (not_visited_nodes.size() == 0) {
        return estimated_cost;
    }

    int* adjacency_matrix = (int*)malloc(num_nodes * num_nodes * sizeof(int));
    int k;
    if (parallelize) {
        #pragma omp parallel for private(k) num_threads(2)
        for (k = 0; k < num_nodes; k++) {
            for (int j = 0; j < num_nodes; j++) {
                adjacency_matrix[k * num_nodes + j] = adj_matrix[k * num_nodes + j];
            }
        }
    }
    else {
        for (k = 0; k < num_nodes; k++) {
            for (int j = 0; j < num_nodes; j++) {
                adjacency_matrix[k * num_nodes + j] = adj_matrix[k * num_nodes + j];
            }
        }
    }

    int* row_wise_minimums = new int[num_nodes];
    int* column_wise_minimums = new int[num_nodes];
    bool* row_wise_minimums_initialized = new bool[num_nodes];
    bool* column_wise_minimums_initialized = new bool[num_nodes];
    for (int i = 0; i < num_nodes; i++) {
        row_wise_minimums_initialized[i] = false;
        column_wise_minimums_initialized[i] = false;
    }

    // Считаем минимумы по строкам
    for (const int i : not_visited_nodes) {
        for (const int j : not_visited_nodes) {
            if (i != j) {
                // Учитываем, что из нулевой вершины выходить не нужно, она - конец цепочки
                if (!row_wise_minimums_initialized[i] && i != 0) {
                    row_wise_minimums_initialized[i] = true;
                    row_wise_minimums[i] = adjacency_matrix[i * num_nodes + j];
                }
                else {
                    if (i != 0 && adjacency_matrix[i * num_nodes + j] < row_wise_minimums[i]) {
                        row_wise_minimums[i] = adjacency_matrix[i * num_nodes + j];
                    }
                }
            }
        }
    }
    
    // Вычитаем минимумы из строк
    for (int i = 0; i < num_nodes; i++) {
        for (int j = 0; j < num_nodes; j++) {
            adjacency_matrix[i * num_nodes + j] -= row_wise_minimums[i];
        }
    }
    // Считаем минимумы по столбцам
    for (const int i : not_visited_nodes) {
        for (const int j : not_visited_nodes) {
            if (i != j) {
                // Учитываем, что в вершину, в которой сейчас находимся, повторно заходить не надо
                if (j != source_id && !column_wise_minimums_initialized[j]) {
                    column_wise_minimums_initialized[j] = true;
                    column_wise_minimums[j] = adjacency_matrix[i * num_nodes + j];
                }
                else {
                    
                    if (j != source_id && adjacency_matrix[i * num_nodes + j] < column_wise_minimums[j]) {
                        column_wise_minimums[j] = adjacency_matrix[i * num_nodes + j];
                    }
                }
            }
        }
    }

    
    for (int i = 0; i < num_nodes; i++) {
        if (row_wise_minimums_initialized[i]) {
            // Учитываем, что из нулевой вершины выходить не нужно, она - конец цепочки
            if(i != 0)
            estimated_cost += row_wise_minimums[i];
        }
        if (column_wise_minimums_initialized[i]) {
            // Учитываем, что в вершину, в которой сейчас находимся, повторно заходить не надо
            if (i != source_id)
            estimated_cost += column_wise_minimums[i];
        }

    }
    return estimated_cost;
}


void process_subpath(int prefix_cost, list<int> visited_nodes_list, int node_id_under_processing,
    Result &baseline, int num_nodes,  int* adjacency_matrix, set<int> not_visited_nodes, bool parallel_flag)
{
    int next_step_prefix_cost;
    int estimated_subgraph_cost;
    int best_child_cost;
    Result candidate;
    Result result;
    result.subtree_cost = baseline.subtree_cost;
    result.subtree_nodes = std::list<int>(baseline.subtree_nodes);
    std::set<int> not_visited_nodes_copy;
    std::list<int>visited_nodes_list_copy;
    if (prefix_cost >= baseline.subtree_cost) {
        return;
    }
    if (not_visited_nodes.size() == 0) {
        if (parallel_flag) {
            #pragma omp critical
            {
                if (prefix_cost + adjacency_matrix[node_id_under_processing * num_nodes] < baseline.subtree_cost) {
                    std::list<int> visited_nodes_list_copy(visited_nodes_list);
                    visited_nodes_list_copy.push_back(0);
                    modify_struct(baseline, prefix_cost + adjacency_matrix[node_id_under_processing * num_nodes], visited_nodes_list_copy);
                }
            }

        }
        else {
            if (prefix_cost + adjacency_matrix[node_id_under_processing * num_nodes] < baseline.subtree_cost) {
                std::list<int> visited_nodes_list_copy(visited_nodes_list);
                visited_nodes_list_copy.push_back(0);
                modify_struct(baseline, prefix_cost + adjacency_matrix[node_id_under_processing * num_nodes], visited_nodes_list_copy);
            }
        }
    }
    for (const int next_node_id : not_visited_nodes) {
        next_step_prefix_cost = prefix_cost + adjacency_matrix[node_id_under_processing * num_nodes + next_node_id];
        std::set<int> not_visited_nodes_copy(not_visited_nodes);
        std::list<int> visited_nodes_list_copy(visited_nodes_list);
        // При оценке надо добавить нулевую вершину, потому что необходимо учесть, что хотя и не можем из ней выйти, но войти должны
        // Также нужно учесть, что из вершины next_node_id нужно выйти, но в неё не заходить
        not_visited_nodes_copy.insert(0);
        estimated_subgraph_cost = calculate_path_cost_estimation(num_nodes, adjacency_matrix, not_visited_nodes_copy, next_node_id, parallel_flag);
        not_visited_nodes_copy.erase(0);
        not_visited_nodes_copy.erase(next_node_id);
        visited_nodes_list_copy.push_back(next_node_id);
        if (estimated_subgraph_cost + next_step_prefix_cost <= baseline.subtree_cost) {
            process_subpath(next_step_prefix_cost, visited_nodes_list_copy,
                next_node_id, baseline, num_nodes, adjacency_matrix, not_visited_nodes_copy, parallel_flag);
       }
    }
}


int main() {
    /*
    В среднем многопоточность чаще даёт прирост (иногда время выполнения уменьшается на ~40%, как правило, чуть меньше),
    но иногда однопоточному коду "везёт", и он на ранних этапах достаточно хорошо улучшает бэйзлайн, чтобы
    избежать обработки огромного числа вершин.
    */
    int num_threads = 2;
    omp_set_num_threads(num_threads);
    // Включаю вложенный параллелизм, чтобы дополнительно параллелить операции над матрицы при подсчёте оценки стоимости пути
    omp_set_nested(true);
    
    srand(time(NULL));
    double time_parallel, time_consecutive;

    const int NUM_NODES = 20;
    int* adjacency_matrix = (int*)malloc(NUM_NODES * NUM_NODES * sizeof(int));

    for (int i = 0; i < NUM_NODES; i++) {
        for (int j = 0; j < NUM_NODES; j++) {
            if(i != j)
                adjacency_matrix[i * NUM_NODES + j] = rand();
            else
                adjacency_matrix[i * NUM_NODES + j] = -1;
        }
    }

    printf("Adjacency matrix:\n");
    for (int i = 0; i < NUM_NODES; i++) {
        for (int j = 0; j < NUM_NODES; j++) {
            printf("%d ", adjacency_matrix[i * NUM_NODES + j]);
        }
    printf("\n");    
    }


    // Базовое решение
    int baseline_cost = adjacency_matrix[(NUM_NODES - 1) * NUM_NODES ];
    for (int i = 0; i < NUM_NODES - 1; i++) {
        baseline_cost += adjacency_matrix[i * NUM_NODES + i + 1];
    }

    // Создаём множество непосещённых вершин
    set<int> not_visited_nodes; 
    for (int i = 1; i < NUM_NODES; i++) {
        not_visited_nodes.insert(i);
    }
    
    // Создаём список посещённых вершин чтобы запоминать уже пройденный путь
    std::list<int> visited_nodes_list = {0,};
    int prefix_path_cost;
    Result best;
    best.subtree_cost = baseline_cost;
    best.subtree_nodes = std::list<int>();
    for (int i = 0; i <= NUM_NODES; i++) {
        best.subtree_nodes.push_back(i % NUM_NODES);
    }


    double time = omp_get_wtime();
    printf("Baseline: %d\n", baseline_cost);
    for (int i = 1; i < NUM_NODES; i++) {
        std::set<int> not_visited_nodes_copy(not_visited_nodes);
        std::list<int> visited_nodes_list_copy(visited_nodes_list);
        
        not_visited_nodes_copy.erase(i);
        visited_nodes_list_copy.push_back(i);
        process_subpath(adjacency_matrix[0 * NUM_NODES + i], visited_nodes_list_copy,  i, best, NUM_NODES, adjacency_matrix, not_visited_nodes_copy, false);
    }
    time_consecutive = omp_get_wtime() - time;
    printf("Non-parallel execution time: %f\n", time_consecutive);

    int checksum = 0;
    int start_pointer = 0;
    printf("Non-parallel Best path, cost %d:\n", best.subtree_cost);
    for (const int i : best.subtree_nodes) {
        printf("%d ", i);
        if (i != start_pointer) {
            checksum += adjacency_matrix[start_pointer * NUM_NODES + i];
            start_pointer = i;
        }
        
    }
    printf("\nchecksum: %d\n---\n", checksum);

    best.subtree_cost = baseline_cost;
    best.subtree_nodes = std::list<int>();
    for (int i = 0; i < NUM_NODES; i++) {
        best.subtree_nodes.push_back(i);
    }
    best.subtree_nodes.push_back(0);
    int k;
    time = omp_get_wtime();
    #pragma omp parallel shared(adjacency_matrix, best)  private(k)
    {

            #pragma omp for
            for (k = 1; k < NUM_NODES; k++) {
                int id = omp_get_thread_num();
                printf("Thread number: %d, Processing node: %d\n", id, k);
                set<int> not_visited_nodes_parallel = set<int>();
                std::list<int> visited_nodes_list_parallel = {0};
                for (int i = 1; i < NUM_NODES; i++) {
                    if (i != k)
                    not_visited_nodes_parallel.insert(i);
                }
                visited_nodes_list_parallel.push_back(k);
                process_subpath(adjacency_matrix[0 * NUM_NODES + k], visited_nodes_list_parallel, k,
                    best, NUM_NODES, adjacency_matrix, not_visited_nodes_parallel, true);

            }
        
    }


    time_parallel = omp_get_wtime() - time;
    printf("Parallel execution time (including printfs): %f\n", time_parallel);
    checksum = 0;
    start_pointer = 0;

    printf("Baseline: %d\n", baseline_cost);
    printf("Parallel best path, cost %d:\n", best.subtree_cost);
    for (const int j : best.subtree_nodes) {
        printf("%d ", j);
        if (j != start_pointer) {
            checksum += adjacency_matrix[start_pointer * NUM_NODES + j];
            start_pointer = j;
        }

    }
    printf("\nChecksum: %d\n", checksum);
    
    
}

