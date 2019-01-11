
% pix_to_length.m

% use ginput to relate pixels to length, based on framing square placed in
% images. Come up with deviation stats.

% 1st run (using 25 images from BeachSurveys\15_10_2018\PM\CrossShore\00_m)
% gave stats (in mm/pixel):
% meanxscaling = 0.162023948505943 +- 0.002834179797841 in
% range(0.014195876101333) 
% meanyscaling = 0.163014071440058 +- 0.002124408418811 in
% range(0.007503191012270) 

% 0.162023948505943*5152 = 834.74 mm ground width
% /2 =  417.37 mm (ground width used for grain sizing)

% 2nd run (10 images from C:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\18_10_2018\AM\dense_array1)
% meanxscaling = 0.124623464007929 +- 0.001555618134125 in
% range(0.005322471140083) 
% meanyscaling = 0.122488913818034 +- 0.001909701006224 in
% range(0.006409800921243) 

% 642.060 mm ground width. To get same as before, must do
% 417*x = 642
% x = 1.54 ~= ***1.5*** [new factor to use in this code]

% canon resolution : 5152 x 3864

clear
close all

% dn = 'C:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\15_10_2018\PM\CrossShore\00_m\';
dn = 'C:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\18_10_2018\AM\dense_array1\';
fn = 'IMG_0345.jpg';

% validation
dn = 'C:\Projects\AdvocateBeach2018\data\raw\images\LabValidation\';

imnames = dir([dn 'IMG*']);

count = 0;

xscl = zeros(1, floor(length(imnames)/2));
yscl = zeros(1, floor(length(imnames)/2));

for ii = 1:length(imnames)
    
    if mod(ii, 2) == 0;
        
        count = count + 1;
       
        img = imread([dn imnames(ii).name]);

        figure(1), clf
            image((img))

        [x, ~] = ginput(2);    
        xlen = abs(x(2)-x(1));

        [~, y] = ginput(2);
        ylen = abs(y(2)-y(1));

        x_mm = 600;
        y_mm = 400;

        xscl(count) = x_mm/xlen;
        yscl(count) = y_mm/ylen;

    end
end

meanxscaling = mean(xscl(1:10));
meanyscaling = mean(yscl(1:10));

stdxscaling = std(xscl(1:10));
stdyscaling = std(yscl(1:10));

range(xscl(1:10))
range(yscl(1:10))