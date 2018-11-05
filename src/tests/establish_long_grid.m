% establish_grid.m

% Tristan Guest
% 12 Oct 2018

% define grid for Advocate Beach surveying; rotate and translate to easting
% northing

clear 
close all


% select beach-frame coordinates to revert to pixel coordinates 
xshore = (-30:3:60)';
% xshore = ([-15, -10, -5, 0, 5])';
% loncoord= (-70:1:30);
loncoord= ([0]);

fileID = fopen('survey_grid_long.txt', 'w');
fclose(fileID); 

figure(1), clf, hold on

Ipt = 0;

for ii = 1 : length(loncoord)
    
%     count = count + 1;
    
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
        
    utm_coords = [(Ipt + (1:length(xshore))')'; northing'; easting'; zeros(size(easting'))];    
%     utm_coords = [northing'; easting'; zeros(size(easting'))];        
    
    fileID = fopen('survey_grid_long.txt', 'a');
%     fprintf(fileID,['\ncross-shore line ' num2str(ii) '\n\n']);
    fmt = '%d %10.10f %10.10f %10.10f\n';
%     fmt = '%10.10f %10.10f %10.10f\n';
    fprintf(fileID,fmt, utm_coords);
    fclose(fileID);    
    
    Ipt = Ipt + length(xshore);
    
end

% select beach-frame coordinates to revert to pixel coordinates 
% xshore = (-40:1:60)';
xcoord = ([-5]);
lonshore= (-50:3:25)';
% loncoord= ([-30,0,30]);

% flip axis for consistency
    lonshore = -lonshore;

for ii = 1 : length(xcoord)
    
    % define series of longshore transect coordinates
    xshore = xcoord(ii)*ones(1,length(lonshore))';

%     % flip axis for consistency
%     lonshore = -lonshore;

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
%         
% %     utm_coords = [1:length(lonshore); northing'; easting'; zeros(size(easting'))];    
%     utm_coords = [northing'; easting'; zeros(size(easting'))];        
%     
%     fileID = fopen('survey_grid.txt', 'a');
% %     fprintf(fileID,['\nlongshore line ' num2str(ii) '\n\n']);
% %     fmt = '%d %10.10f %10.10f %10.10f\n';
%     fmt = '%10.10f %10.10f %10.10f\n';
%     fprintf(fileID,fmt, utm_coords);
%     fclose(fileID);   
    
    
    
    %
        utm_coords = [(Ipt + (1:length(lonshore))')'; northing'; easting'; zeros(size(easting'))];    
%     utm_coords = [northing'; easting'; zeros(size(easting'))];        
    
    fileID = fopen('survey_grid_long.txt', 'a');
%     fprintf(fileID,['\nlongshore line ' num2str(ii) '\n\n']);
    fmt = '%d %10.10f %10.10f %10.10f\n';
%     fmt = '%10.10f %10.10f %10.10f\n';
    fprintf(fileID,fmt, utm_coords);
    fclose(fileID);    
    
    Ipt = Ipt + length(lonshore);
end

%%
% data point filenames
fn_stakes = 'ADVOCATE_stakepoints.txt';
fn_array1 = 'ADVOCATE_array1.txt';

% Hemisphere RTK directory
dnRTK = 'C:\Projects\AdvocateBeach2018\data\raw\past_experiments\RTK_GPS\';

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