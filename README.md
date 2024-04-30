There are multiple relevant files to run for this project, involving dataset generation, training, and testing.

Dataset Generation:
- to create channel matrices, in MATLAB, run 'Dataset Generation/QuadRiGa_test.m'
- to use the channel matrices to create csv files for LBCellularEnv.py to parse, run 'Dataset Generation/c_vals_saver_test.m'

Training:
- to train the standard DQN, run 'FB_LoadBalancing/Train.py'
- to train the DQN using the independent agent paradigm, run 'Train_partitioned.py'
- Note that paths will need to be changed in order to pull dataset files from the location saved to by c_vals_saver_test.m
- Different reward shaping functions can be implemented in LBCellularEnv.py _get_reward() function by applying desired function to the per_user_utility variable

Testing:
- to generate test results files, run 'FB_LoadBalancing/Test.py'
- Note that paths may need to be changed in order to use the files saved to a folder called 'runs/' that Train.py generates
- to generate plots without variance bounds, run 'FB_LoadBalancing/merge_plot_results.py'
- to generate plots with variance bounds, run 'FB_LoadBalancing/merge_plot_results_mult_ep.py'
