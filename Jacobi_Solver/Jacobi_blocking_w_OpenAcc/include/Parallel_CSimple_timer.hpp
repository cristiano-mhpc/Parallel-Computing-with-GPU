#pragma once

#include <chrono>
#include <map>
#include <string>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <vector>

/**
 *  USE THIS AS
 *  {CSimple_timer t("WHATEVER YOU ARE TIMING");
 *        THE CODE YOU ARE TIMING
 *    }
 *
 *  THEN AT THE END OF MAIN PUT
 *  CSimple_timer::print_timing_results();
 */
struct Data{

  long long int timings;
  int num_calls;

};

std::map<std::string, struct Data> table;

class CSimple_timer {

public:
  
  using time_units = std::chrono::microseconds;

  // DECLARE VARIABLES FOR WHAT WE ARE TIMING, FOR CLOCK START AND END
  std::chrono::time_point<std::chrono::steady_clock> t_start;
  std::chrono::time_point<std::chrono::steady_clock> t_end;
 
  struct Data TimerData; 
  
  long long int duration;

  std::string timewhat;  

  //static void print_timing_results(); // IMPLEMENT THIS

  int rank;
  int comm_sz;
  MPI_Comm comm;

  // constructior
  CSimple_timer(const std::string &timewhat0, int rank1, int size, MPI_Comm comm1):rank(rank1), comm_sz(size), comm(comm1){
    // SET WHAT WE ARE TIMING FROM THE PASSED PARAMETER
    timewhat = timewhat0;
    // LOOK FOR THE STRING OCCURENCE AND EITHER AUGMENT THE CALLS OR
    if(table.find(timewhat) != table.end()){
      table[timewhat].num_calls++;  
      TimerData = table[timewhat];

    } else {
    // INSERT A NEW THING INTO THE TABLE
      TimerData.num_calls = 1;
      TimerData.timings = 0;
      table.insert(std::make_pair(timewhat, TimerData));
    }
    // START THE CLOCKS
    t_start = std::chrono::steady_clock::now();
  };

  // destructor
  ~CSimple_timer() {
    // STOP THE CLOCK
    t_end = std::chrono::steady_clock::now();
    // CALCULATE DURATION
    duration =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start)
            .count();
    // INSERT THAT INTO YOUR TABLE
      table[timewhat].timings += duration; 
  } // destructor

  // here comes important new info: static class functions - they can be called
  // without the object of the class

  static void print_timing_results(int some_rank, int comm_size, MPI_Comm comm1) {
    //the number of unique scopes to time
    int num_funcs = table.size();
    int i;
    //declare vectors for the statistics
    std::vector<long long int> min;
    std::vector<long long int> max;
    std::vector<long long int> sum;
    //array containing the averages for each scope, since each scope can be timed more than once.
    std::vector<long long int> av_timing(num_funcs);
    //vector containing the names of the every scope timed
    std::vector<std::string> scope_names(num_funcs);

    if(some_rank == 0){
      //allocate vectors for the statistics
      min.resize(num_funcs);
      max.resize(num_funcs);
      sum.resize(num_funcs);
      i = 0;
      for (const auto &pair:table){
        scope_names[i] = pair.first;
        i++;
      }
    }

    /**
     * access timings in struct Data, located in the table map.
     * the same scope can be timed more than once. So we get here the average of the timings.
    */
    i = 0;
    for (const auto &pair:table){
      av_timing[i] = (pair.second).timings/(pair.second).num_calls;
      i++;
    }

    //get the reduced variables for each timed scope
    MPI_Reduce(av_timing.data(), max.data(), num_funcs, MPI_LONG_LONG, MPI_MAX, 0, comm1);
    MPI_Reduce(av_timing.data(), min.data(), num_funcs, MPI_LONG_LONG, MPI_MIN, 0, comm1);
    MPI_Reduce(av_timing.data(), sum.data(), num_funcs, MPI_LONG_LONG, MPI_SUM, 0, comm1);

    if (some_rank == 0){
      /*
      std::cout << "The minimum execution time across all process for each scope are:" << std::endl;
      for(i = 0; i < num_funcs; i++){
          std::cout << scope_names[i] << ": " << min[i] << std::endl;
      }
      */
     
      //std::cout << "The maximum execution time across all process for each scope are:" << std::endl;
      std::cout << std::fixed << std::setprecision(15);
      //std::cout << scope_names[0] << " " << max[0]*table["comm"].num_calls<< std::endl;
      std::cout << max[0]*table["comm"].num_calls << " " << max[1]*table["comp"].num_calls << std::endl;
      //std::cout << scope_names[0] << " " << scope_names[1] << std::endl;
      /*
      std::cout << "The sum of execution times across all processes for each scope are:" << std::endl;
      for(i = 0; i < num_funcs; i++){
          std::cout << scope_names[i] << ": " << sum[i] << std::endl;
      }
      */
      /*
      std::cout << "The average execution time across all processes for each scope are:" << std::endl;
      for(i = 0; i < num_funcs; i++){
        sum[i] /= comm_size;
        std::cout << scope_names[i] << ": "<< sum[i] << std::endl;
      }
      */

    }

  }
  
}; // of class



