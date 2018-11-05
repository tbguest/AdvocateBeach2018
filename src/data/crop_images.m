

clear
close all

% dn = 'P:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\15_10_2018\PM\CrossShore\00_m\';
% dn = 'P:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\15_10_2018\PM\Longshore\m05_m\';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FOR SURVEY IMAGES
date_str = '27_10_2018';
tide = 'PM';
imset = 'longshore1'; %'cross_shore'; % longshore; longshore1; longshore2, dense_array1; dense_array2

dn = ['C:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\' date_str '\' tide '\' imset '\'];
dnout = ['C:\Projects\AdvocateBeach2018\data\processed\images\cropped\beach_surveys\' date_str '\' tide '\' imset];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % FOR PICAM IMAGES
% dn = ['C:\Projects\AdvocateBeach2018\data\interim\images\PiCameras\23_10_2018\horn_growth\selected\'];
% dnout = ['C:\Projects\AdvocateBeach2018\data\interim\images\PiCameras\23_10_2018\horn_growth\selected\cropped'];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create the folder if it doesn't exist already.
if ~exist(dnout, 'dir')
  mkdir(dnout);
end

% 21-B
% dn = 'C:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\21_10_2018\PM\cross_shore\';
% dnout = 'C:\Projects\AdvocateBeach2018\data\processed\images\cropped\beach_surveys\21_10_2018\PM\cross_shore\';

% fn = 'IMG_1755.jpg';

% camera height was changed (at least once)
% use 1 for 1st setting, 2 for 2nd, etc
cameraHeight = 2;

imnames = dir([dn 'img*']);

counter = 0;

for ii = 1:length(imnames)
    
    counter = counter + 1;
    
    if mod(counter, 2) == 0;

        img = imread([dn imnames(ii).name]);
    %     img = imread([dn fn]);

    %     figure(100)
    %     image(img)

        if cameraHeight == 1

            % mask dims
            hght = floor(size(img, 1)/2);
            wdth = floor(size(img, 2)/2);

            % mask origin
            h0 = floor(size(img, 1)/4);
            w0 = floor(size(img, 2)/4);

        elseif cameraHeight == 2

            % mask dims
            hght = floor(size(img, 1)/1.5);
            wdth = floor(size(img, 2)/1.5);

            % mask origin
            h0 = floor(size(img, 1)/6);
            w0 = floor(size(img, 2)/6);

        end

        newimg = imcrop(img, [w0, h0, wdth, hght]);

        imwrite(newimg, [dnout '\' imnames(ii).name(1:end-4) '-cropped.jpg'])

%         figure(ii), clf
%             image(newimg)
    
    end

end



