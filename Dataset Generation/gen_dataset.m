clear
clc

fignum = 0;

%% Parameters
num_ue = 120; % number of UEs
num_bs = 2; % number of BSs

Nt=32; % number of tx antenna elements
Nr=4; % number of rx antenna elements

v_mean = 15; % mean UE velocity
v_var = 9; % variance of UE velocity

ue_height = 1.5;
bs_height = 25;

center_freqs = [2500e6, 700e6]; % 2500 MHz, 700 MHz

BW = [60e6, 10e6]; % 60 MHz for 2500 MHz band, 10 MHz for 700 MHz band

num_subcarrier = 1;                  % Narrowband setting
% num_subcarrier = 667; % 512        % 667: subcarrier spacing 30kHz for 20MHz BW

%% Channel model setup and coefficient generation

s = qd_simulation_parameters; % Set up simulation parameters
s.show_progress_bars = 1; % Show progress bar 0: disabled
s.center_frequency = center_freqs; % Set center frequency (2500 MHz, 700 MHz)

% s.samples_per_meter = 1; 360/(40*pi); 1; % 1 sample per meter
% s.use_absolute_delays = 1; % Include delay of the LOS path

%%  Antenna arrays (Tx and Rx)

% Function: qd_arrayant( Ant elements in elevation / azimuth /  center_freq / vertical pol. /
% degrees of downtilt /  xx*lambda spacing)

% Set Dipole TX antenna
tx_array1 = [qd_arrayant( '3gpp-3d',  1, 64, s.center_frequency(1))];
tx_array2 = [qd_arrayant( '3gpp-3d',  1, 64, s.center_frequency(2))];

% tx_array = qd_arrayant( '3gpp-3d',  8, 4, s.center_frequency(1), 1, 12, 0.5);
% tx_array = qd_arrayant( '3gpp-3d',  2, 8, s.center_frequency(1), 2, 12, 0.5);
% tx_array = qd_arrayant( '3gpp-3d',  8, 4, s.center_frequency(1), 2 );

% Set Dipole RX antenna
rx_array1 = qd_arrayant( '3gpp-3d',  1, 1, s.center_frequency(1));
rx_array2 = qd_arrayant( '3gpp-3d',  1, 1, s.center_frequency(2));

% rx_array = qd_arrayant( '3gpp-3d',  2, 1, s.center_frequency(1), 2, [], 0.5);
% rx_array = qd_arrayant( '3gpp-3d',  4, 1, s.center_frequency(1));
% Another antenna config
% l.tx_array = qd_arrayant('omni'); % Omni-directional BS antenna
% l.rx_array = a; % Omni-directional MT antenna

%% Generate tracks
% gen_sq_track([start_x; start_y; start_z], height, width)
t1 = gen_sq_track([-300; 10; ue_height], 100, 600);
t2 = gen_sq_track([-300; -110; ue_height], 210, 200);
t3 = gen_sq_track([-100; 120; ue_height], 100, 200);
t4 = gen_sq_track([-300; -110; ue_height], 100, 600);
t5 = gen_sq_track([-100; -220; ue_height], 100, 200);
t6 = gen_sq_track([-100; 10; ue_height], 100, 200);
t7 = gen_sq_track([-100; -110; ue_height], 100, 200);

track_opts = [t1, t2, t3, t4, t5, t6, t7];
track_lens = arrayfun(@(x) x.get_length(), track_opts);

%% Layout and Channel Generation

l = qd_layout(s); % Create new QuaDRiGa layout

% Set number of tx, rx
l.no_rx = num_ue;
l.no_tx = num_bs;

% initialize arrays
l.tx_array = repmat([tx_array1; tx_array2], 1, num_bs);
l.rx_array = repmat([rx_array1; rx_array2], 1, num_ue);

% initialize positions
l.tx_position = [-100 0 bs_height; 100 0 bs_height]';  % (100,0), (-100, 0) x, y position

% Randomly distribute UEs
for ue=1:num_ue
    % Assign ue to random track

    % sample a track with equal likelihood
    % ue_track_idx = randi(numel(track_opts), 1); % sample num_ue tracks from available % even sample
    % ue_track = copy(track_opts(ue_track_idx));

    % alternatively sample track weighted by length
    % ue_track = copy(randsample(track_opts, 1, true, track_lens)); % sample track directly
    ue_track_idx = randsample(length(track_opts), 1, true, track_lens);
    ue_track = copy(track_opts(ue_track_idx));
    ue_track.name = ['track', num2str(ue_track_idx), 'ue', num2str(ue)]; % need unique name

    % Randomize ue position on track
    ue_track.positions = circshift(ue_track.positions, randi(size(ue_track.positions, 2)), 2); % random start point
    ue_track.positions = ue_track.positions + ue_track.initial_position; % uncancel old initial position
    ue_track.initial_position = ue_track.positions(:, 1); % zero out new initial position
    ue_track.positions = ue_track.positions - ue_track.initial_position; % add offset for new initial position

    % Randomly reverse direction
    if randi([0, 1])
        ue_track = rev_track(ue_track);
    end

    % Set Random Speed drawn from gaussian
    ue_track.set_speed(normrnd(v_mean, sqrt(v_var)))

    l.rx_track(1, ue) = ue_track;
end

fignum = fignum+1;
l.visualize([], [], 0);

% Scenarios
BerUMaL = 'BERLIN_UMa_LOS';
BerUMaN = 'BERLIN_UMa_NLOS';
StdUMa= '3GPP_38.901_UMa';
StdUMi= '3GPP_38.901_UMi';
temp = '3GPP_38.901_UMi_LOS';

l.set_scenario(StdUMa); % Select scenario

% Comment: qd_builder.supported_scenarios   % Scenario list


%% Build
% p = l.init_builder;

% p.scenpar.NumClusters = 20;     % Reduce paths (for faster processing)
% p.plpar=[];     % Disable pathloss model

% p.gen_parameters();          % Generate small-scaling fading parameters

% ch_coeffs = p.get_channels;    % Generate channel coefficients
ch_coeffs = l.get_channels;


tic; % start timer

%% Generation start

% initialize channel matrix - extra dim for each ue
H_set = zeros(num_tx, num_ue, l.rx_array(1).no_elements, l.tx_array(1).no_elements);

for ue_idx = 1 : num_ue
    
    % Channel generation
    h_t1 = ch_coeffs(ue_idx).fr( BW(1), num_subcarrier); %cn.fr( 3855000, 257 );   B    W 20 MHz     % Freq.-domain channel
    h_t2 = ch_coeffs(ue_idx).fr( BW(1), num_subcarrier);


    H_set(ue_idx, :, :) = h_t; 

    % Keep track of timing
    toc_temp = toc;
    for disp_idx = 1 : 10
        if ue_idx/num_ue == disp_idx*0.1
            disp(['     Processing : ', num2str(disp_idx*10), '% completed,  Elapsed time is ', num2str( toc_temp )])
        end
    end

end

%% Collect Receiver Positions
% Assuming each row of rx_positions contains the [x, y, z] coordinates of a receiver
rx_positions = zeros(num_ue, 3);
for ue_idx = 1:num_ue
    rx_positions(ue_idx, :) = c(ue_idx).rx_position;
end

%% Storage path
storage_path = "./Data_folder/";

if not(isfolder(storage_path))
    mkdir(storage_path);
end

layout_info = strcat("NumUEs", string(num_ue), "_");
ant_info = strcat("Nt", string(l.tx_array(1).no_elements), "_Nr", string(l.rx_array(1).no_elements), "_");
date_info = strcat("date", string(datetime('now', 'Format', 'yyyyMMdd_HHmmss')), "_");
description_info = "RLdata";

% For multi-user
filename = strcat(layout_info, ant_info, date_info, description_info,".mat");

% save(filename, "H_set"); 
save(filename, "H_set", "rx_positions") %, '-v7.3'); %'-v7.3'
