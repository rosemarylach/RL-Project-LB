clear; clc; close all;

s = qd_simulation_parameters;                           % Set up simulation parameters
s.show_progress_bars = 1;                               % Disable progress bars
s.center_frequency = [2.5e9, 0.7e9];                    % Two center frequencies of interest

% Antenna configuration 1 (UMa and UMi)
% 10 elements in elevation, 1 element in azimuth, vertical pol., 12 deg downtilt, 0.5 lambda spacing
a = qd_arrayant( '3gpp-3d', 10, 1, [], 4, 12, 0.5 );
a.element_position(1,:) = 0.5;             % Distance from pole in [m]
l = qd_layout.generate( 'regular', 1, 0, a);
l.simpar = s;
l.no_tx = 2;                                            % 2 BSs, 2 freqs, 3 sectors 

% BS SETUP
l.tx_position(:,1) = [-100 ; 0 ; 25] ;                  % BS 1
l.tx_position(:,2) = [100 ; 0; 25];                     % BS 2

% UE SETUP
l.rx_array = qd_arrayant('omni');                	    % Omni-Antenna, vertically polarized
l.no_rx = 2;                                          % Number of UEs
% Trajectories for UEs
t = qd_track('linear',200,0);
t.initial_position = [-100, 10, 1.5]';                          % Set the Rx height to 1.5 meters
c = 78*exp( 1i*pi/2 );                                  % 78 m segment, direction N
t.positions = [t.positions,...                          % Append segment to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];
c = 200*exp( 1i*pi );                                   % 200 m segment, direction W
t.positions = [t.positions,...                          % Append segment to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];
c = 78*exp( -1i*pi/2 );                                  % 78 m segment, direction S
t.positions = [t.positions,...                          % Append segment to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];
t.calc_orientation;                                     % Calculate the receiver orientation

% /////////////////////////////////////////////////////
% UPDATE HERE.
% We need to find a way to assign UEs randomly to these trajectories by
% shifting the initial positions of the trajectories.
% /////////////////////////////////////////////////////



% [~,l.rx_track] = interpolate( t.copy,'distance',0.1 );  % Interpolate and assign track to layout
% l.visualize([],[],0);                                   % Plot
% axis equal
% title('Track layout');                                  % Set plot title

t.scenario = '3GPP_38.901_UMa_LOS';
t.movement_profile = [ 0, 556/15;... % Time points in seconds vs.
    0, 556];                  %    distance in meters
l.rx_track(1,1) = t;                                         % Assign terminal track for the receiver
t2 = t.copy();
t2.name = 'Rx2';
l.rx_track(1,1).name = 'Rx1';                           % Set the MT1 name
l.rx_track(1,2) = t2;                                         % Assign terminal track for the receiver




% set(0,'DefaultFigurePaperSize',[14.5 7.8])              % Adjust paper size for plot
% [map,x_coords,y_coords]=l.power_map('5G-ALLSTAR_Urban_LOS','quick',1,-300,300,-300,300);
% P = 10*log10( map{1}(:,:,1) ) + 50;                     % RX copolar power @ 50 dBm TX power
% l.visualize([],[],0);                                   % Plot layout
% axis([-300,300,-300,300]);                              % Axis
% hold on
% imagesc( x_coords, y_coords, P );                       % Plot the received power
% hold off
l.update_rate = 0.01;                                   % Set channel update rate to 100 Hz
c = l.get_channels;                                     % Generate channels

pow_1  = 10*log10( reshape( sum(abs(c(1).coeff(:,1,:,:)).^2,3) ,1,[] ) );    % Calculate the power
pow_2  = 10*log10( reshape( sum(abs(c(1).coeff(:,2,:,:)).^2,3) ,1,[] ) );    % Calculate the power
pow_3  = 10*log10( reshape( sum(abs(c(1).coeff(:,3,:,:)).^2,3) ,1,[] ) );    % Calculate the power

time = (0:c(1).no_snap-1)*0.01;                            % Vector with time samples
figure(2);
plot(time,pow_1'+50);
hold on;
plot(time,pow_2'+50);
plot(time,pow_3'+50);
hold off;


