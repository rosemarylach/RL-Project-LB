function rev = rev_track(track)
%REV_TRACK Reverses an input track to go the opposite direction
%   Detailed explanation goes here


rev = copy(track);
rev.positions = [rev.positions(:, 1), rev.positions(:, end:-1:2)];

end

