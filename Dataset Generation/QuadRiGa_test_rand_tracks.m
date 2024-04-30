clear; clc; close all;

chan_realization_num = 20;

output_folder_name = 'chan_realizations';
output_folder_path = fullfile(pwd, output_folder_name);  % pwd is the current directory
if ~exist(output_folder_path, 'dir')
    mkdir(output_folder_path);  % Make the directory if it doesn't exist
end

for chan_realization_ind = 1 : 1 : chan_realization_num
    output_folder_name = strcat('chan_realization_', num2str(chan_realization_ind));
    channel_folder_path = fullfile(output_folder_path, output_folder_name);  % pwd is the current directory
    if ~exist(channel_folder_path, 'dir')
        mkdir(channel_folder_path);  % Make the directory if it doesn't exist
    end
    matFileName = strcat('c_vals_', num2str(chan_realization_ind), '.mat');
    fullFilePath = fullfile(channel_folder_path, matFileName);

    %% Parameters
    ep_len = 900;

    num_ue = 120; % number of UEs
    num_bs = 2; % number of BSs

    Nt=10; % number of tx antenna elements
    Nr=1; % number of rx antenna elements

    v_mean = 15; % mean UE velocity
    v_var = 9; % variance of UE velocity

    ue_height = 1.5;
    bs_height = 25;

    center_freqs = [2500e6, 700e6]; % 2500 MHz, 700 MHz

    BW = [60e6, 10e6]; % 60 MHz for 2500 MHz band, 10 MHz for 700 MHz band
    % Just a note, I believe the BW should not be affecting the channel
    % behavior.

    num_subcarrier = 1;                  % Narrowband setting
    % num_subcarrier = 667; % 512        % 667: subcarrier spacing 30kHz for 20MHz BW

    %%

    s = qd_simulation_parameters;                           % Set up simulation parameters
    s.show_progress_bars = 1;                               % Disable progress bars
    s.center_frequency = [2.5e9, 0.7e9];                    % Two center frequencies of interest

    % Antenna configuration 1 (UMa and UMi)
    % 10 elements in elevation, 1 element in azimuth, vertical pol., 12 deg downtilt, 0.5 lambda spacing
    a = qd_arrayant( '3gpp-3d', Nt, 1, [], 4, 12, 0.5 ); % 10x1 array
    % a = qd_arrayant( '3gpp-3d', 2, 2, [], 4, 12, 0.5); % 2x2 array
    a.element_position(1,:) = 0.5;             % Distance from pole in [m]
    l = qd_layout.generate( 'regular', 1, 0, a);
    l.simpar = s;
    l.no_tx = num_bs;                                            % 2 BSs, 2 freqs, 3 sectors

    % BS SETUP
    l.tx_position(:,1) = [-100 ; 0 ; 25] ;                  % BS 1
    l.tx_position(:,2) = [100 ; 0; 25];                     % BS 2

    % UE SETUP
    l.rx_array = qd_arrayant('omni');                	    % Omni-Antenna, vertically polarized
    l.no_rx = num_ue;                                          % Number of UEs

    %% Generate tracks
    % gen_sq_track([start_x; start_y; start_z], height, width)
    t1 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t2 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t3 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t4 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t5 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t6 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t7 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t8 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t9 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t10 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t11 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t12 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    t13 = gen_sq_track_from_center([400*rand()-200; 200*rand()-100; ue_height], 400*rand()+50, 500*rand()+50);
    track_opts = [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13];
    track_lens = arrayfun(@(x) x.get_length(), track_opts);


    % Randomly distribute UEs
    for ue=1:num_ue
        % Assign ue to random track

        % sample a track with equal likelihood
        % ue_track_idx = randi(numel(track_opts), 1); % sample num_ue tracks from available % even sample
        % ue_track = copy(track_opts(ue_track_idx));

        % alternatively sample track weighted by length
        % ue_track = copy(randsample(track_opts, 1, true, track_lens)); % sample track directly
        ue_track_idx = randsample(length(track_opts), 1, true, track_lens);
        ue_track = track_opts(ue_track_idx).copy;

        % [~, ue_track] = interpolate(ue_track.copy,'distance', 1/s.samples_per_meter );
        ue_track.interpolate_positions(s.samples_per_meter);

        ue_track.name = ['track', num2str(ue_track_idx), 'ue', num2str(ue)]; % need unique name

        ue_track.scenario = '3GPP_38.901_UMa_LOS';

        % Randomize ue position on track
        ue_track.positions = circshift(ue_track.positions, randi(size(ue_track.positions, 2)), 2); % random start point
        ue_track.positions = ue_track.positions + ue_track.initial_position; % uncancel old initial position
        ue_track.initial_position = ue_track.positions(:, 1); % zero out new initial position
        ue_track.positions = ue_track.positions - ue_track.initial_position; % add offset for new initial position
        ue_track.positions = [ue_track.positions, ue_track.positions, ue_track.positions, ue_track.positions, ue_track.positions, ue_track.positions];

        % Set Random Speed drawn from gaussian
        % ue_track.set_speed(normrnd(v_mean, sqrt(v_var)));
        v = normrnd(v_mean, sqrt(v_var));
        ue_track.movement_profile = [0 (ep_len-1)*0.1; ...
            0 v*(ep_len-1)*0.1];

        min_track_len = v * (ep_len-1)*0.1;
        num_loops = ceil(min_track_len / ue_track.get_length);

        % Randomize ue position on track
        ue_track.positions = circshift(ue_track.positions, randi(size(ue_track.positions, 2)), 2); % random start point
        ue_track.positions = ue_track.positions + ue_track.initial_position; % uncancel old initial position
        ue_track.initial_position = ue_track.positions(:, 1); % zero out new initial position
        ue_track.positions = ue_track.positions - ue_track.initial_position; % add offset for new initial position

        ue_track.positions = repmat(ue_track.positions, 1, num_loops);


        % Randomly reverse direction
        if randi([0, 1])
            ue_track = rev_track(ue_track);
        end


        l.rx_track(1, ue) = ue_track;
    end

    % l.visualize([], [], 0);

    % [~,l.rx_track] = interpolate( t.copy,'distance',0.1 );  % Interpolate and assign track to layout
    %     l.visualize([],[],0);                                   % Plot
    % axis equal
    % title('Track layout');                                  % Set plot title

    % t.scenario = '3GPP_38.901_UMa_LOS';
    % t.movement_profile = [ 0, 556/15;... % Time points in seconds vs.
    %     0, 556];                  %    distance in meters
    % l.rx_track(1,1) = t;                                         % Assign terminal track for the receiver
    % t2 = t.copy();
    % t2.name = 'Rx2';
    % l.rx_track(1,1).name = 'Rx1';                           % Set the MT1 name
    % l.rx_track(1,2) = t2;                                         % Assign terminal track for the receiver




    % set(0,'DefaultFigurePaperSize',[14.5 7.8])              % Adjust paper size for plot
    % [map,x_coords,y_coords]=l.power_map('5G-ALLSTAR_Urban_LOS','quick',1,-300,300,-300,300);
    % P = 10*log10( map{1}(:,:,1) ) + 50;                     % RX copolar power @ 50 dBm TX power
    % l.visualize([],[],0);                                   % Plot layout
    % axis([-300,300,-300,300]);                              % Axis
    % hold on
    % imagesc( x_coords, y_coords, P );                       % Plot the received power
    % hold off

    l.update_rate = 0.1;                                   % Set channel update rate to 10 Hz
    % The duration of each slot is 100 ms, so changing this to 10 Hz.

    c = l.get_channels;                                     % Generate channels
    save(fullFilePath, 'c');
    fprintf("\nChannel realization: %d\n", chan_realization_ind);
    clear('c', 's', 'a', 'l', 'ue_track');

end
% pow_1  = 10*log10( reshape( sum(abs(c(1).coeff(:,1,:,:)).^2,3) ,1,[] ) );    % Calculate the power
% pow_2  = 10*log10( reshape( sum(abs(c(1).coeff(:,2,:,:)).^2,3) ,1,[] ) );    % Calculate the power
% pow_3  = 10*log10( reshape( sum(abs(c(1).coeff(:,3,:,:)).^2,3) ,1,[] ) );    % Calculate the power
%
% time = (0:c(1).no_snap-1)*0.1;                            % Vector with time samples
% figure(2);
% plot(time,pow_1'+50);
% hold on;
% plot(time,pow_2'+50);
% plot(time,pow_3'+50);
% hold off;

% /////////////////////////////////////////////////////
% CHECK THE c.name!
% I believe that the different channels are named similarly in Manan's
% code.
% /////////////////////////////////////////////////////