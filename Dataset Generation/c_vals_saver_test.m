% QuadRiGa channel object loaded into variable 'quadriga_channel_obj'
file_path = pwd;
file_path = strcat(file_path, '/chan_realizations/chan_realization_');
num_realizations = 16;

% Start loading the channel realizations in order and shape as data frames.
for chan_realization_ind = 1 : 1 : num_realizations
    load(strcat(file_path,num2str(chan_realization_ind),'/','c_vals_', num2str(chan_realization_ind), '.mat') );

    output_folder_name = 'UE_CSV_Files';
    output_folder_path = fullfile(pwd, output_folder_name);  % pwd is the current directory
    if ~exist(output_folder_path, 'dir')
        mkdir(output_folder_path);  % Make the directory if it doesn't exist
    end


    quadriga_channel_obj = c;
    % BS transmit power in dBm
    BS_TX_POWER_DBM = 40;

    % Create subfolder for specific channel realization
    chan_realization_folder = strcat('chan_realization_', num2str(chan_realization_ind));
    chan_realization_path = fullfile(output_folder_path, chan_realization_folder);
    if ~exist(chan_realization_path, 'dir')
        mkdir(chan_realization_path);  % Create the subfolder if it doesn't exist
    end

    % Preallocate array to store CSV data

    [num_ue, num_bs, num_freqs] = size(c);  % Total number of UEs, BSs, freqs
    [~, num_sectors, num_taps, num_timestamps] = size(c(1,1,1).coeff); % Number of sectors per BS, taps in a channel, timestamps

    % Define the CSV data columns
    column_names = {'x_coord', 'y_coord', 'z_coord', 'Timestamp', 'serving pci 1', 'serving pci 2', 'serving pci 3', 'serving pci 4', 'serving pci 5', 'serving pci 6', 'RSRP1', 'RSRP2','RSRP3','RSRP4','RSRP5','RSRP6'};
    csv_data = cell(num_timestamps, length(column_names));

    % Convert BS transmit power to linear scale
    bs_tx_power_linear = 10^(BS_TX_POWER_DBM / 10);

    freqs_string = ["0", "1"];

    % Iterate over each UE, BS, sector, frequency, and timestamp
    for ue_idx = 1:num_ue
        for freq_idx = 1:num_freqs
            % Iterate for each UE and freq component and save it in a file.
            row_counter = 1;
            % Initialize the csv again for the next iteration.
            csv_data = cell(num_timestamps, length(column_names));
            for ts = 1:num_timestamps
                rsrp_6 = [];        % Holds the 6 RSRP values.
                for bs_idx = 1:num_bs
                    for sector_idx = 1:num_sectors
                        % Access the channel coefficients for current UE, BS, sector, and frequency
                        channel = quadriga_channel_obj(ue_idx, bs_idx, freq_idx);
                        coeff = channel.coeff(1,sector_idx,:,ts);

                        % Calculate channel energy (sum of the absolute square of all taps)
                        channel_energy_linear = sum(abs(coeff).^2);

                        % Combine BS transmit power with channel energy
                        rsrp_linear = bs_tx_power_linear * channel_energy_linear;

                        % Convert back to dBm
                        rsrp_dbm = 10 * log10(rsrp_linear);

                        rsrp_6 = [rsrp_6, rsrp_dbm];
                    end
                end
                % Retrieve UE position at current timestamp
                rx_position = channel.rx_position(:, ts);
                x_coord = rx_position(1);
                y_coord = rx_position(2);
                z_coord = rx_position(3);
                % Store in cell array for CSV
                csv_data(row_counter, :) = {x_coord, y_coord, z_coord, ts, 1,2,3,4,5,6, rsrp_6(1),rsrp_6(2),rsrp_6(3),rsrp_6(4),rsrp_6(5),rsrp_6(6) };
                row_counter = row_counter + 1;
            end
            % Convert cell array to table
            csv_table = cell2table(csv_data, 'VariableNames', column_names);

            % Full path for the CSV file
            csv_filename = strcat('exp', num2str(chan_realization_ind),'_ue', num2str(ue_idx), '_', freqs_string(freq_idx), '.csv');
            full_csv_path = fullfile(chan_realization_path, csv_filename);

            
            % Export table to CSV file
            writetable(csv_table, full_csv_path);
        end
    end

end

% disp(['CSV file saved as ', csv_filename]);
