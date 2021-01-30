#pragma once

#include <string>
#include <unordered_map>

using namespace std;

enum class StatisticResult {
    EFFICIENCY,
    INTR_PER_SEC,
    SELECTED_TRACES,
    TOTAL_SELECTION_KERNEL_EXECUTION_TIME,
    TOTAL_KERNEL_EXECUTION_TIME,
    TOTAL_STATISTIC_COMPUTE_TIME,
    TOTAL_COPY_TIME,
    CNT
};

extern unordered_map<StatisticResult, string> STATISTIC_NAME_MAP;
