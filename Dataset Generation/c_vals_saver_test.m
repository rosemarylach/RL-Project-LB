% QuadRiGa channel object loaded into variable 'quadriga_channel_obj'
load('c_vals.mat');
quadriga_channel_obj = c;
% BS transmit power in dBm
BS_TX_POWER_DBM = 40;

% Preallocate array to store CSV data
num_ue = 120;  % Total number of UEs
num_bs = 2;    % Number of base stations
num_sectors = 3; % Number of sectors per BS
num_freqs = 2;  % Number of frequencies
num_timestamps = 900; % Number of timestamps
num_taps = 34;  % Number of taps

% Define the CSV data columns
column_names = {'UE_Index', 'BS_Index', 'Sector_Index', 'Frequency_Index', 'Timestamp', 'RSRP_dBm', 'X_Coordinate', 'Y_Coordinate'};
csv_data = cell(num_ue * num_bs * num_sectors * num_freqs * num_timestamps, length(column_names));
row_counter = 1;

% Convert BS transmit power to linear scale
bs_tx_power_linear = 10^(BS_TX_POWER_DBM / 10);

% Iterate over each UE, BS, sector, frequency, and timestamp
for ue_idx = 1:num_ue
    for bs_idx = 1:num_bs
        for sector_idx = 1:num_sectors
            for freq_idx = 1:num_freqs
                for ts = 1:num_timestamps
                    % Access the channel coefficients for current UE, BS, sector, and frequency
                    channel = quadriga_channel_obj(ue_idx, bs_idx, freq_idx);
                    coeff = channel.coeff(1,:,sector_idx,ts);
                    
                    % Calculate channel energy (sum of the absolute square of all taps)
                    channel_energy_linear = sum(abs(coeff).^2, 2);
                    
                    % Combine BS transmit power with channel energy
                    rsrp_linear = bs_tx_power_linear * channel_energy_linear;
                    
                    % Convert back to dBm
                    rsrp_dbm = 10 * log10(rsrp_linear);
                    
                    % Retrieve UE position at current timestamp
                    rx_position = channel.rx_position(:, ts);
                    x_coordinate = rx_position(1);
                    y_coordinate = rx_position(2);
                    
                    % Store in cell array for CSV
                    csv_data(row_counter, :) = {ue_idx, bs_idx, sector_idx, freq_idx, ts, rsrp_dbm, x_coordinate, y_coordinate};
                    row_counter = row_counter + 1;
                end
            end
        end
    end
end

% Convert cell array to table
csv_table = cell2table(csv_data, 'VariableNames', column_names);

% Export table to CSV file
csv_filename = 'quadriga_simulation_rsrp.csv';
writetable(csv_table, csv_filename);
disp(['CSV file saved as ', csv_filename]);
