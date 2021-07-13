// File "pppm_impl.cpp"
#ifdef LMP_USER_INTEL
#include "pppm_intel.cpp"

using namespace LAMMPS_NS;

template void PPPMIntel::particle_map<float,double>(IntelBuffers<float,double> *buffers); 
template void PPPMIntel::particle_map<double,double>(IntelBuffers<double,double> *buffers);
template void PPPMIntel::particle_map<float,float>(IntelBuffers<float,float> *buffers);
template void PPPMIntel::make_rho<float,double,0>(IntelBuffers<float,double> *buffers); 
template void PPPMIntel::make_rho<double,double,0>(IntelBuffers<double,double> *buffers);
template void PPPMIntel::make_rho<float,float,0>(IntelBuffers<float,float> *buffers);
template void PPPMIntel::make_rho<float,double,1>(IntelBuffers<float,double> *buffers); 
template void PPPMIntel::make_rho<double,double,1>(IntelBuffers<double,double> *buffers);
template void PPPMIntel::make_rho<float,float,1>(IntelBuffers<float,float> *buffers);
#endif
