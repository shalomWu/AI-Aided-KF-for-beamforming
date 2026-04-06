clear all;
% Number of configurations
numUsers = 1;       % Number of users
numAntennas = 16;    % Number of antennas


% Define the file name for this configuration
filename = sprintf('channel_magnitudes.mat');

% Initialize QuaDRiGa simulation parameters
simParams = qd_simulation_parameters;
simParams.center_frequency = 0.5e9;    % 0.5 GHz center frequency
simParams.sample_density = 40;%40;     % Samples per half wavelength

% Create layout object
layout = qd_layout(simParams);

%% Step 1: Define the Base Station (BS)
layout.tx_position = [0; 0; 10];        % Base station at origin with 10m height
layout.no_tx = 1;                       % Single base station

% Create the BS antenna array (MIMO)

bsArray = qd_arrayant('omni');          % Omnidirectional antennas
bsArray.center_frequency = simParams.center_frequency;
bsArray.no_elements = numAntennas;      % Antennas at BS
wavelength = simParams.wavelength;

bsArray.element_position(2, :) = ...
    (0:wavelength:(numAntennas-1)*wavelength) ...
    - 0.5*numAntennas*wavelength;       % 0.5 wavelength spacing
layout.tx_array = bsArray;              % Assign array to the BS

%% Step 2: Define the User Equipment (UEs)
layout.no_rx = numUsers;                % Number of receivers

% Set random positions for UEs within a 25m x 25m area
layout.randomize_rx_positions(50, 0.5, 2.5, 0); % was 50 in the beginging, 1.5, 1.5

% Create single omnidirectional antenna for each UE
ueArray = qd_arrayant('omni');          % Omnidirectional antenna
ueArray.no_elements = 1;                % Single antenna per user
% Create an array of qd_arrayant objects for each user
layout.rx_array = repmat(ueArray, 1, numUsers); % Directly replicate the object

%% Step 3: Define Receiver Tracks
% Define track length and direction for each user
trackLength = 20*19.8;                       % was 2*19.8 Track length in meters
directions = linspace(0, 2*pi, numUsers+1); % Unique directions for each user
directions = directions(1:end-1);
% Create an array of qd_track objects for each user
rxTracks = repmat(qd_track, 1, numUsers); % Initialize array of qd_track objects

% Define movement for each user
for i = 1:numUsers
    rxTracks(i) = qd_track('linear', trackLength, directions(i)); % Create linear track
    rxTracks(i).initial_position = layout.rx_position(:, i); % Set initial positions
    rxTracks(i).name = ['User-' num2str(i)];                      % Assign unique name
    rxTracks(i).interpolate_positions(simParams.samples_per_meter); % Interpolate positions
end

% Assign tracks to the layout
layout.rx_track = rxTracks; % Assign array of qd_track objects

% Plot Receiver and Tracks
plot_receiver_and_tracks(layout.tx_position(1:2), rxTracks);

%% Step 4: Set Propagation Scenario
% Set the 3GPP Urban Macro (UMa) scenario for all users
layout.set_scenario('3GPP_38.901_UMa');

%% Step 5: Generate Channel Coefficients
% Generate channel coefficients using the layout configuration
channels = layout.get_channels();

%% Step 6: Analyze and Visualize Results
% Initialize the result matrix
numSnapshots = size(channels(1,1).coeff, 4);                     % Extract the number of snapshots
H = zeros(numUsers, bsArray.no_elements, numSnapshots); % Size: n_users x n_antennas x n_snapshots

% Loop through all users
for userIndex = 1:numUsers
    % Get the channel object for the current user
    channel = channels(userIndex, :); % All antennas for this user

    % Extract the coefficients for the first path (LOS)
    coeff_first_path = squeeze(channel.coeff(1, :, 1, :)); % Size: nAntennas x nSnapshots

    % Store in the result matrix
    H(userIndex, :, :) = coeff_first_path; % This is the original channel values
   % H(userIndex, :, :) = coeff_first_path*sqrt(length(coeff_first_path)./sum(abs(coeff_first_path).^2)); %This is the normalized channel values

end

%% Normalizing H to have gain 1.
% Assuming H is of size [N, M, L]
% [N, M, L] = size(H);
% 
% % Preallocate normalized tensor
% H_norm = zeros(N, M, L);
% 
% % Loop through each channel and normalize
% for n = 1:N
%     for m = 1:M
%         h = squeeze(H(n, m, :));       % Extract Lx1 vector
%         %norm_val = norm(h, 2);         % Compute 2-norm
%         norm_val=1/sqrt(L/sum(abs(h).^2));
%         if norm_val ~= 0
%             H_norm(n, m, :) = h / norm_val;  % Normalize if norm is non-zero
%         end
%     end
% end
% %H=H_norm;
% clear h;
% Save the 'magnitudes' matrix to disk
save(filename, 'H');

% Plot Magnitudes and Phases
plot_magnitudes_and_phases(H);




% =================== AUXILIARY FUNCTIONS ===================

function plot_receiver_and_tracks(receiver_pos, user_tracks)
figure; hold on; grid on;
set(gcf,'Position',[100 100 700 500])
xlabel('X Position', 'FontSize', 16);
ylabel('Y Position', 'FontSize', 16);
sgtitle('Receiver and Transmitter Tracks');

% Plot receiver
plot(receiver_pos(1), receiver_pos(2), 'rp', 'MarkerSize', 12, ...
    'DisplayName', 'Receiver');

% Plot each user track
num_users = numel(user_tracks);
colors = lines(num_users);
for i = 1:num_users
    track = user_tracks(i).positions(1:2, :)' + ...
        + user_tracks(i).initial_position(1:2)';

    % Initial position
    plot(track(1,1), track(1,2), 'o', 'Color', colors(i, :), ...
        'MarkerSize', 8, 'HandleVisibility', 'off');

    % Path
    plot(track(:,1), track(:,2), '-', 'Color', colors(i, :), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('User %d', i));
end

legend('Location', 'bestoutside');
axis equal;
exportgraphics(gcf, 'receiver_tracks.pdf', 'ContentType', 'vector');
end

function plot_magnitudes_and_phases(H)
[num_users, ~, T] = size(H);
time = 1:T;
colors = lines(num_users);  % Consistent colors for antennas

figure;
set(gcf,'Position',[100 100 1200 500])

% Magnitude subplot
subplot(1, 2, 1);
hold on;
for u = 1:num_users
    h_ua = squeeze(H(u, 1, :));
    plot(time, abs(h_ua), 'Color', colors(u,:), 'LineWidth', 1.5, ...
        'DisplayName', sprintf('User %d', u));
end
xlabel('Time', 'FontSize', 16); ylabel('|h|', 'FontSize', 16);
legend('show');
grid on;

% Phase subplot (with unwrapping)
subplot(1, 2, 2);
hold on;
for u = 1:num_users
    h_ua = squeeze(H(u, 1, :));
    plot(time, unwrap(angle(h_ua)), 'Color', colors(u,:), ...
        'LineWidth', 1.5, 'DisplayName', sprintf('User %d', u));
end
xlabel('Time', 'FontSize', 16); ylabel('\angle h', 'FontSize', 16);
legend('show');
grid on;

sgtitle('Time Evolution of MIMO Channel Coefficients (Magnitude & Phase)');
exportgraphics(gcf, 'channel_characteristics.pdf', 'ContentType', 'vector');
end


