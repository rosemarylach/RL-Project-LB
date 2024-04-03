clear
clc

%%
Nt=32; Nr=4;
At=[];
Ar=[];
anglegrid_tx = zeros(Nt,1);
anglegrid_rx = zeros(Nr,1);

%% Making a grid on sin values 
for a = 1 : Nt
    anglegrid_tx(a) = 2/(Nt)*(a-1)-1;
end
for a = 1 : Nr
    anglegrid_rx(a) = 2/(Nr)*(a-1)-1;
end
asd = 1
%% Making DFT Matrix
for itx = 1 : Nt
    at = 1/sqrt(Nt).*exp(1j*pi.*(0:(Nt-1))'.*anglegrid_tx(itx));
    At = [At at];
end
for irx = 1 : Nr
    ar = 1/sqrt(Nr).*exp(1j*pi.*(0:(Nr-1))'.*anglegrid_rx(irx));
    Ar = [Ar ar];
end

%% Channel model setup and coefficient generation

s = qd_simulation_parameters; % Set up simulation parameters
s.show_progress_bars = 1; % Show progress bar 0: disabled
s.center_frequency = [2500e6, 700e6]; % Set center frequency (2500 MHz, 700 MHz)

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

%% Layout and Channel Generation
num_ue = 120; % number of UEs
num_bs = 2;

l = qd_layout(s); % Create new QuaDRiGa layout

% Set number of tx, rx
l.no_rx = num_ue;
l.no_tx = num_bs;

% initialize arrays
l.tx_array = [tx_array1, tx_array2];
l.rx_array = [rx_array1, rx_array2];

% initialize positions
l.tx_position = [-100 0 25; 100 0 25]';  % 25m Tx heigh and (100,0), (-100, 0) x, y position

l.randomize_rx_positions(100,1.2,2.0,0); % 200 m radius, 1.5 m Rx height, 0 tracklength -> "1 Rx snapshot".
% l.randomize_rx_positions(100,1.6,1.6,0); % 200 m radius, 1.5 m Rx height, 0 tracklength -> "1 Rx snapshot".

% Note: Default track length is 1m.
% randomize_rx_positions(max_dist, min_height, max_height, tracklength, rx_ind, min_dist, orientation)

% Scenarios
BerUMaL = 'BERLIN_UMa_LOS';
BerUMaN = 'BERLIN_UMa_NLOS';
StdUMa= '3GPP_38.901_UMa';
StdUMi= '3GPP_38.901_UMi';
temp = '3GPP_38.901_UMi_LOS';

l.set_scenario(StdUMa); % Select scenario
% Comment: qd_builder.supported_scenarios   % Scenario list

% Build
p = l.init_builder;
% p.scenpar.NumClusters = 20;     % Reduce paths (for faster processing)
% p.plpar=[];     % Disable pathloss model
p.gen_parameters;          % Generate small-scaling fading parameters

c = p.get_channels;    % Generate channel coefficients

% sqrt(10^(0.1*c(1).par.pg)) .... sqrt(10^(0.1*c(1000).par.pg))

tic; % start timer

% For test (Seeing the average)
% iii=1;
% h_t_mean = zeros(Nr,Nt);

%% Generation start

% num_RBs = 52; % Note: in LTE, 12 subcarriers -> 1 RB
% num_subcarrier_per_RB = 12;

% initialize channel matrix - extra dim for each ue
H_set = zeros(num_tx, num_ue, l.rx_array(1).no_elements, l.tx_array(1).no_elements);

% H_set = zeros(num_ue, num_RBs, l.rx_array(1).no_elements, l.tx_array(1).no_elements);
% H_ang_set = zeros(num_ue, num_RBs, l.rx_array(1).no_elements, l.tx_array(1).no_elements);
% UE_position_set = zeros(num_ue, 3);

for ue_idx = 1 : num_ue
    % num_subcarrier = 667; % 512        % 667: subcarrier spacing 30kHz for 20MHz BW
    num_subcarrier = 1;                  % Narrowband setting
    
    % Channel generation
    h_t = c(ue_idx).fr( 20e6, num_subcarrier); %cn.fr( 3855000, 257 );   B    W 20 MHz     % Freq.-domain channel

    % h_f = ifft2(h_t);
    % h_t = (1/sqrt(10^(0.1*c(ue_idx).par.pg))) * h_t; %XXX PL Exist : PL yes, Not exist : PL no.
    % 
    % for RB_idx = 1 : num_RBs
    %      h_t_reshaped = reshape(h_t(:,:,1:num_RBs*num_subcarrier_per_RB), [Nr, Nt, num_subcarrier_per_RB, num_RBs]);
    %     val_tmp = 0;
    %     for subcar_idx = 1 : num_subcarrier_per_RB
    %         val_tmp = val_tmp + (h_t(:,:,RB_idx+subcar_idx)/num_subcarrier_per_RB);
    %         H_set(ue_idx, RB_idx, :, :) = H_set(ue_idx, RB_idx, :, :) + (h_t(:,:,RB_idx+subcar_idx)/num_subcarrier_per_RB);
    %     end
    %     H_set(ue_idx, RB_idx, :, :) = val_tmp;
    % end
    H_set(ue_idx, :, :) = h_t; 
%     h_ang = Ar'*h_t*At; % For narrowband, h_t is Nr x Nt.


    % Permutate it if you want
    % H_set(ue_idx,:,:) = permute(h_t,[2,1]);
    % Hf_set(ue_idx,:,:) = permute(h_f,[2,1]);
    % 
    % H_set(ue_idx,:,:) = h_t;
    % H_ang_set(ue_idx,:,:) = h_ang;
    % UE_position_set(ue_idx,:,:) = c(ue_idx).rx_position;

    toc_temp = toc;
    for disp_idx = 1 : 10
        if ue_idx/num_ue == disp_idx*0.1
            disp(['     Processing : ', num2str(disp_idx*10), '% completed,  Elapsed time is ', num2str( toc_temp )])
        end
    end

    % Test - print image
    %     h_t_mean = h_t_mean + h_t;
    %     figure(iii)
    %     imagesc(abs(Ar'*h_t*At))
    % %     figure(2)
    % %     imagesc(abs(h_f_test))
    %     disp(c(ue_idx).rx_position)
    %     iii = iii+1;
end

% % First, merge the first two dimensions
% H_set = reshape(H_set, [num_ue*num_RBs, Nr, Nt]);
% 
% H_set_final = zeros(num_ue*num_RBs, Nt, Nr);
% % Transpose the last two dimensions for each matrix
% for i = 1:(num_ue*num_RBs)
%     H_set_final(i, :, :) = permute(squeeze(H_set(i, :, :)), [2 1]);
% end


% h_t_mean = h_t_mean/num_ue;
% figure(1)
%     imagesc(abs(Ar'*h_t_mean*At))
% figure(2)
%     plot(abs((1/8 * ones(1,4) * Ar'*h_t_mean*At)))

% Collect Receiver Positions
% Assuming each row of rx_positions contains the [x, y, z] coordinates of a receiver
rx_positions = zeros(num_ue, 3);
for ue_idx = 1:num_ue
    rx_positions(ue_idx, :) = c(ue_idx).rx_position;
end

%% Storage path
% basepath = "D:/Dropbox/Code/Nokia-UT/Quadriga_chmodel";
basepath = "./Data_folder";
storage_path = strcat(basepath, "/Data_Narrowband(CH+UEposition)_Nt", string(l.tx_array(1).no_elements), "_Nr", string(l.rx_array(1).no_elements));
if not(isfolder(storage_path))
    mkdir(storage_path);
end

% H_set_perm = permute(H_set, [3,1,2]);
% Hf_set_perm = permute(H_ang_set, [3,1,2]);

DIRNAME = datestr(now,'yyyymmdd');
% filename = strcat(storage_path,"/Samples", string(num_ue*num_RBs) , "_date", DIRNAME,datestr(now,'HHMMSS'),".mat");
% save(filename, "H_set_final"); 

% For multi-user
% filename = strcat(storage_path,"/NumUEs", string(num_ue), "_num_RBs", string(num_RBs), "_date", DIRNAME,datestr(now,'HHMMSS'),"multiuser_ULA_nopl.mat");
filename = strcat(storage_path,"/NumUEs", string(num_ue), "_date", DIRNAME,datestr(now,'HHMMSS'),"multiuser_ULA_nopl.mat");

% save(filename, "H_set"); 
save(filename, "H_set", "rx_positions") %, '-v7.3'); %'-v7.3'

% save(filename, "H_set", "H_ang_set", "UE_position_set"); 

% basepath = "./Data_folder";
% filename = strcat(basepath,"/DeepMIMO/O1_/ch_data_4_64_only1.h5");
% batch_ch = h5read(filename,batch_ch);
% batch_UE_loc = h5read(filename,batch_UE_loc);

% Test code for visualization -- Ignore it!
% chtmp = zeros(Nr, Nt);
% fronorm = 0 ;
% for i = 1 : size(data_complex,1)
% %     chtmp = chtmp + squeeze(batch_ch(i,:,:))/norm(squeeze(batch_ch(i,:,:)),'fro');
%     chtmp = chtmp + squeeze(data_complex(i,:,:));
%     fronorm = fronorm + (norm(squeeze(data_complex(i,:,:))))^2;
% end
% chtmp = chtmp/size(data_complex,1);
% fronorm = sqrt(fronorm/size(data_complex,1));
% figure(1)
%     mesh(abs(Ar'*chtmp*At))
% %     imagesc(abs(Ar'*chtmp*At))
% figure(2)
%     plot(abs((1/Nr * ones(1,Nr) * Ar'*chtmp*At)))
% 
% mesh(abs(Ar'*squeeze(H_set(2060,:,:))*At))
% 
