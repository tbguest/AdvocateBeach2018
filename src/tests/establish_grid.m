% establish_grid.m

% Tristan Guest
% 12 Oct 2018

% define grid for Advocate Beach surveying; rotate and translate to easting
% northing

clear 
close all


% select beach-frame coordinates to revert to pixel coordinates 
xshore = (-40:0.5:60)';
loncoord= (-70:0.5:30);

figure(1), clf, hold on

for ii = 1 : length(loncoord)

    % define series of longshore transect coordinates
    lonshore = loncoord(ii)*ones(1,length(xshore))';

    % flip axis for consistency
    lonshore = -lonshore;

    % compute angle for rotation into beach-oriented ref frame using stakes
    % aligned across shore
    stakes_east1 = 3.578993979000000e+05;
    stakes_east2 = 3.579084006000000e+05;
    stakes_north1 = 5.022709436700000e+06;
    stakes_north2 = 5.022717801400000e+06;
    beach_rot = pi + atan((stakes_east1 - stakes_east2)/(stakes_north2 - stakes_north1));

    % define rotation matrix for beach-oriented reference frame
    R = [cos(beach_rot) -sin(beach_rot); sin(beach_rot) cos(beach_rot)];
    % apply beach rotation matrix (back to world coordinates)
    Y = [xshore,lonshore]';
    Yp = R'*Y; % counter-rotate -- multiply by rotation matrix transpose
    LON = Yp(1,:)';
    LAT = Yp(2,:)';

    % define array origin
    originx = 3.579093296000000e+05;
    originy = 5.022719408400000e+06;
    
    easting = LON + originx;
    northing = LAT + originy;

    % center on origin
    LAT0 = 5.022771079200000e+06 - originy; % cam tower northing
    LON0 = 3.579174971000000e+05 - originx; % cam tower easting
    x_w = LON - LON0;
    z_w = LAT - LAT0;

    % define y==H
%     y_w = H*ones(size(x_w));
   
    figure(1)
        plot(easting, northing, '.')
end


% data point filenames
fn_stakes = 'ADVOCATE_stakepoints.txt';
fn_array1 = 'ADVOCATE_array1.txt';

% Hemisphere RTK directory
dnRTK = '~/Experiments/Advocate_2015/data/HemisphereGPS/';

% load in stake positions
fid = fopen([dnRTK fn_stakes]);
A = textscan(fid, '%f %f %f %f %s','Delimiter',',');
fclose(fid);

% load in PT sensor positions
fid = fopen([dnRTK fn_array1]);
B = textscan(fid, '%f %f %f %f %s','Delimiter',',');
fclose(fid);

% assign vector labels to columns
stakes_wp = A{1};
stakes_north = A{2};
stakes_east = A{3};
stakes_height = A{4};

% assign vector labels to columns
array1_wp = B{1};
array1_north = B{2};
array1_east = B{3};
array1_height = B{4};

figure(1)
    plot(stakes_east, stakes_north, '*')
    plot(array1_east, array1_north, '*')