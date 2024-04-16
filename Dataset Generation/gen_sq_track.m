function t = gen_sq_track(init_pos, height, width, interp)
%GEN_SQ_TRACK Generate a square track from linear tracks
% Note: init_pos relative to bottom left corner

% if nargin < 4
%     cw = true;
% end

if nargin < 4
    interp = true;
end


t = qd_track('linear', width, 0); % initial horizontal leg
t.initial_position = init_pos;

new_len = height;
new_dir = pi/2; % north
c = new_len * exp(1j*new_dir);
t.positions = [t.positions,...                          % Append segment to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];

new_len = width;
new_dir = pi; % west
c = new_len * exp(1j*new_dir);
t.positions = [t.positions,...                          % Append segment to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];

new_len = height;
new_dir = -pi/2; % south
c = new_len * exp(1j*new_dir);
t.positions = [t.positions,...                          % Append segment to existing track
    [ t.positions(1,end) + real(c); t.positions(2,end) + imag(c); zeros( 1,numel(c) ) ]];

if interp
    [~, t] = interpolate(t.copy,'distance', 0.1 );
end

% if not(cw)
%     % reverse to travel ccw
%     t.positions = [t.positions(:, 1), t.positions(:, end:-1:2)];
% end


end

