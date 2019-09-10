#include <f/optimization/nonlinear/steepest_descent.hpp>
#include <f/pattern/pattern.hpp>
#include <f/cuda_pattern/cuda_pattern.hpp>
#include <f/date/date_to_string.hpp>

#include <vector>
#include <iostream>
#include <complex>
#include <string>

int main()
{
    double const thickness_ = 1;
    //int gpu_id = 1;
    int gpu_id = 0;
    double const init_thickness = thickness_;
    std::complex<double> const thickness{ 0.0, init_thickness  };
    //auto pt = f::make_simulated_pattern("/testpool/ops/samfairman/larbed-4-working-folder/icecap1/testdata/new_txt", thickness);

#if 1
    auto pt = f::make_pattern("/testpool/ops/samfairman/larbed-4-working-folder/icecap1/testdata/new_txt", thickness);
    {   //update fake ug
        f::matrix<double> ug;
        ug.load( "/testpool/ops/samfairman/larbed-4-working-folder/icecap1/testdata/new_txt/ug.txt" );
        pt.update_ug( ug );
    }
    //pt.simulate_intensity();
#endif

for(int index=0; index<80; index++)
std::cout<<"intensity_x_"<<index<<"\n"<< pt.intensity[index]<<std::endl;

    return 0;
}

