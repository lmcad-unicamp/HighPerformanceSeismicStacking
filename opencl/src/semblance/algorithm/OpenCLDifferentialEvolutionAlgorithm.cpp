#include "opencl/include/execution/OpenCLUtils.hpp"
#include "opencl/include/semblance/algorithm/OpenCLDifferentialEvolutionAlgorithm.hpp"

OpenCLDifferentialEvolutionAlgorithm::OpenCLDifferentialEvolutionAlgorithm(
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder,
    unsigned int gen,
    unsigned int ind
) : DifferentialEvolutionAlgorithm(model, context, dataBuilder, gen, ind) {
}

void OpenCLDifferentialEvolutionAlgorithm::compileKernels(const string& deviceKernelSourcePath) {

}

void OpenCLDifferentialEvolutionAlgorithm::computeSemblanceAtGpuForMidpoint(float m0) {
    
}

void OpenCLDifferentialEvolutionAlgorithm::selectTracesToBeUsedForMidpoint(float m0) {
    
}

void OpenCLDifferentialEvolutionAlgorithm::setupRandomSeedArray() {
    
}

void OpenCLDifferentialEvolutionAlgorithm::startAllPopulations() {
    
}

void OpenCLDifferentialEvolutionAlgorithm::mutateAllPopulations() {
    
}

void OpenCLDifferentialEvolutionAlgorithm::crossoverPopulationIndividuals() {
    
}

void OpenCLDifferentialEvolutionAlgorithm::advanceGeneration() {
    
}

void OpenCLDifferentialEvolutionAlgorithm::selectBestIndividuals(vector<float>& resultArrays) {
    
}
