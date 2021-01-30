#include "common/include/semblance/result/StatisticResult.hpp"

using namespace std;

unordered_map<StatisticResult, string> STATISTIC_NAME_MAP = {
    { StatisticResult::EFFICIENCY, "efficiency" },
    { StatisticResult::INTR_PER_SEC, "interpolations_per_sec"},
    { StatisticResult::SELECTED_TRACES, "selected_traces"},
    { StatisticResult::TOTAL_SELECTION_KERNEL_EXECUTION_TIME, "total_selection_kernel_exec_time"},
    { StatisticResult::TOTAL_KERNEL_EXECUTION_TIME, "total_kernel_exec_time"},
    { StatisticResult::TOTAL_STATISTIC_COMPUTE_TIME, "total_statistic_compute_time"},
    { StatisticResult::TOTAL_COPY_TIME, "total_statistic_copy_time"},
};
