#include <f/optimization/nonlinear/steepest_descent.hpp>
#include <f/pattern/pattern.hpp>
#include <f/cuda_pattern/cuda_pattern.hpp>
#include <f/date/date_to_string.hpp>

#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <typeinfo>

int main()
{
    int const thickness_ = 10.0; //in nm
    //int gpu_id = 1;
    int gpu_id = 0;
    double const init_thickness = thickness_;
    std::complex<double> const thickness{ 0.0, init_thickness  };
    auto pt = f::make_simulated_pattern("/testpool/ops/samfairman/larbed-4-working-folder/icecap2/testdata/new_txt", thickness);

//if 1
    //auto pt = f::make_pattern("/testpool/ops/samfairman/larbed-4-working-folder/icecap2/testdata/new_txt", thickness);
    {   //update fake ug
        f::matrix<double> ug;
        ug.load( "/testpool/ops/samfairman/larbed-4-working-folder/icecap2/testdata/new_txt/ug.txt" );
        //pt.update_ug( ug );
    }
    //pt.simulate_intensity();
//endif

for (int scalingFactor=100; scalingFactor<(1000); scalingFactor=(scalingFactor*10+1){
for (int loop=9; loop<10; loop++){

    f::cuda_pattern cpt{ pt, gpu_id };
 
    std::vector<double> ug_initial;
    ug_initial.resize( pt.ug_size*2+1);
    std::fill( ug_initial.begin(), ug_initial.end(), 0.01 );
    int ugSize = pt.ug_size;
    double randMax = 0.00000001*scalingFactor;
    double randMin = -randMax;
    for( int i=1; i < ugSize; i++ ){	              	
        double r = randMin + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(randMax-randMin)));
	//double r = randMin +  ( static_cast <double> (rand()) / randMax ) * (randMax - randMin); 
	std::cout << "r : " << r << std::endl;
	
	ug_initial[2*i] = double(real(pt.ug[i][0]));//-r;
        ug_initial[2*i+1] = double(imag(pt.ug[i][0]));//-r;
        ug_initial[0] = double(real(pt.ug[0][0]));//-r;
        ug_initial[1] = double(imag(pt.ug[0][0]));//-r;
	}  


    
    ug_initial[pt.ug_size*2] = init_thickness; 
    
    for( int i=0; i < ugSize*2; i=i+2 ){
    std::cout <<"\n"<< i << "    "<< ug_initial[i] << "   " << ug_initial[i+1] << std::endl; 
    }
     

    auto const& merit_function = cpt.make_merit_function();

    f::simple_steepest_descent<double> sd( merit_function, pt.ug_size * 2 + 1 );
    sd.config_initial_guess( ug_initial.begin() );
    sd.config_total_steps( 10 );
    sd.config_eps( 1.0e-10 );

    std::string tk = std::to_string(thickness_);
    std::string file_name = f::date_to_string() + std::string{"-"} + std::string{"loop"} + std::to_string(loop) + std::string{"sf"} + std::to_string(scalingFactor) + std::string{"-"} + tk + std::string{"_direct_cuda_pattern.dat"};
    std::string abs_file_name = f::date_to_string() + std::string{"-"} + std::string{"loop"} + std::to_string(loop) + std::string{"sf"} + std::to_string(scalingFactor) + std::string{"-"} + tk +  std::string{"_direct_cuda_pattern_abs.dat"};

    const unsigned long unknowns = pt.ug_size * 2 + 1;

    auto const& abs_function = cpt.make_abs_function(); //abs function better for thick specimens, merit function better for thin

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

}
}
    return 0;
}

