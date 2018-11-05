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

Ipt = 0;

for ii = 1 : length(loncoord)
    
%     count = count + 1;
    
    % define series of longshore transect coordinates
    lonshore = loncoord(ii)*ones(1,length(xshore))';

    % flip axis for consistency
    lonshore = -lonshore;

    figure(1), hold on
        plot(lonshore, xshore, 'y.', 'Markersize', 12)    
    
end


% long grid 2
xcoord = ([-13 -5]);
lonshore= (-50:3:25)';

% flip axis for consistency
    lonshore = -lonshore;

for ii = 1 : length(xcoord)
    
    % define series of longshore transect coordinates
    xshore = xcoord(ii)*ones(1,length(lonshore))';

   
    figure(1)
        plot(lonshore, xshore, 'y.', 'Markersize', 12)

end

% dense_grid2:
xcoord = ([-17,-15,-13,-11,-9,-7]);
lonshore= (-24:-1)';

% flip axis for consistency
    lonshore = -lonshore;

for ii = 1 : length(xcoord)
    
    % define series of longshore transect coordinates
    xshore = xcoord(ii)*ones(1,length(lonshore))';
    
    figure(1)
        plot(lonshore, xshore, 'y.', 'Markersize', 12)


end

set(gca, 'Color', 'none'); % Sets axes background
set(gca, 'ydir', 'reverse')
export_fig(['C:\Projects\AdvocateBeach2018\reports\figures\grid.png'],figure(1),'-dpng','-r450','-transparent')