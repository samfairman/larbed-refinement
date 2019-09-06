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
    int const thickness_ = 500.0;
    //int gpu_id = 1;
    int gpu_id = 0;
    double const init_thickness = thickness_;
    std::complex<double> const thickness{ 0.0, init_thickness  };
    //auto pt = f::make_simulated_pattern("/testpool/ops/samfairman/larbed-4-working-folder/icecap2/testdata/new_txt", thickness);

//if 1
    auto pt = f::make_pattern("/testpool/ops/samfairman/larbed-4-working-folder/icecap2/testdata/new_txt", thickness);
    {   //update fake ug
        f::matrix<double> ug;
        ug.load( "/testpool/ops/samfairman/larbed-4-working-folder/icecap2/testdata/new_txt/ug.txt" );
        pt.update_ug( ug );
    }
    pt.simulate_intensity();
//endif

    f::cuda_pattern cpt{ pt, gpu_id };
    float r1 = 0.8 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(1-0.8)));
    float r2 = 0.005 + static_cast <float> (rand()) /( static_cast <float> (RAND_MAX/(0.015-0.005)));
    std::vector<double> ug_initial;
    ug_initial.resize( pt.ug_size*2 + 1 );
    std::fill( ug_initial.begin(), ug_initial.end(), r2 );
    ug_initial[0] = r1;
    ug_initial[1] = 0.0;
    ug_initial[pt.ug_size*2] = init_thickness;

    auto const& merit_function = cpt.make_merit_function();

    f::simple_steepest_descent<double> sd( merit_function, pt.ug_size * 2 + 1 );
    sd.config_initial_guess( ug_initial.begin() );
    sd.config_total_steps( 100 );
    sd.config_eps( 1.0e-10 );

    std::string tk = std::to_string(thickness_);
    std::string file_name = f::date_to_string() + std::string{"-"} + tk + std::string{"_direct_cuda_pattern.dat"};
    std::string abs_file_name = f::date_to_string() + std::string{"-"} + tk +  std::string{"_direct_cuda_pattern_abs.dat"};

    const unsigned long unknowns = pt.ug_size * 2 + 1;

    auto const& abs_function = cpt.make_abs_function();

    auto on_iteration_over = [ &file_name, &abs_file_name, unknowns, &abs_function ]( double residual, double* current_solution )
    {
        std::ofstream ofs1( file_name.c_str(), std::fstream::app );
        ofs1 << residual << "\t";
        std::copy( current_solution, current_solution+unknowns, std::ostream_iterator<double>( ofs1, "\t" ) );
        ofs1 << "\n";
        ofs1.close();

        std::ofstream ofs2( abs_file_name.c_str(), std::fstream::app );
        double const abs_residual =  abs_function( current_solution );
        ofs2 << abs_residual << "\t";
        std::copy( current_solution, current_solution+unknowns, std::ostream_iterator<double>( ofs2, "\t" ) );
        ofs2 << "\n";
        ofs2.close();
    };

    on_iteration_over( merit_function( ug_initial.data() ), ug_initial.data() );

    sd.config_iteration_function( on_iteration_over );

    sd( ug_initial.begin() );

    f::matrix<std::complex<double> > rug{ ug_initial.size() / 2, 1 };
    std::copy( ug_initial.begin(), ug_initial.end(), reinterpret_cast<double*>(rug.data()) );
    std::cout << "\nsolution:\n" << rug << "\n";

//****************************************************************************************************************
//****************************************************************************************************************
//This section is added for repetition of the fitting algorithm but using the weighted solution as the first guess
//and of an unweighted as defined in folder new_txt2. The idea is that the initial weighting avoids many local minima and 
//the final solution is more accurate when all beams are considered but are already in a global minimum.
//****************************************************************************************************************
//****************************************************************************************************************

 //if 1
	pt = f::make_pattern("/testpool/ops/samfairman/larbed-4-working-folder/icecap2/testdata/new_txt2", thickness); // redefine the pattern pt based on the unweighted model.
    {   //update fake ug
	f::matrix<double> ug;
        ug.load( "/testpool/ops/samfairman/larbed-4-working-folder/icecap2/testdata/new_txt2/ug.txt" );
        pt.update_ug( ug );
    }
    pt.simulate_intensity();
//endif

    ug_initial.resize( pt.ug_size*2 + 1 );

    sd.config_initial_guess( ug_initial.begin() );
    sd.config_total_steps( 100 );
    sd.config_eps( 1.0e-10 );

    on_iteration_over( merit_function( ug_initial.data() ), ug_initial.data() );

    sd.config_iteration_function( on_iteration_over );   


    sd( ug_initial.begin() );

    std::copy( ug_initial.begin(), ug_initial.end(), reinterpret_cast<double*>(rug.data()) );
    std::cout << "\nsolution:\n" << rug << "\n";

    return 0;
}

