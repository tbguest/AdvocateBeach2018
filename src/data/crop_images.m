

clear
close all

% dn = 'P:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\15_10_2018\PM\CrossShore\00_m\';
% dn = 'P:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\15_10_2018\PM\Longshore\m05_m\';

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % FOR SURVEY IMAGES
% notValidation = 1;
% camera height was changed (at least once)
% use 1 for 1st setting, 2 for 2nd, etc
% 3 for validation
% cameraHeight = 2;
% date_str0 = ['23_10_2018'; '23_10_2018'; '24_10_2018'; '24_10_2018'; '25_10_2018'; '25_10_2018';...
%     '26_10_2018'; '26_10_2018'; '27_10_2018'; '27_10_2018'];
% tide0 = ['AM'; 'PM'; 'AM'; 'PM'; 'AM'; 'PM'; 'AM'; 'PM'; 'AM'; 'PM'];

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % FOR PICAM IMAGES
% notValidation = 1;
% % camera height was changed (at least once)
% % use 1 for 1st setting, 2 for 2nd, etc
% % 3 for validation
% cameraHeight = 1;
% % date_str0 = ['23_10_2018'; '23_10_2018'; '24_10_2018'; '24_10_2018'; '25_10_2018'; '25_10_2018';...
% %     '26_10_2018'; '26_10_2018'; '27_10_2018'; '27_10_2018'];
% % tide0 = ['AM'; 'PM'; 'AM'; 'PM'; 'AM'; 'PM'; 'AM'; 'PM'; 'AM'; 'PM'];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FOR VALIDATION IMAGES
% camera height was changed (at least once)
% use 1 for 1st setting, 2 for 2nd, etc
% 3 for validation
cameraHeight = 3;
notValidation = 0;
date_str0 = ['Oct21'; 'Oct21'; 'Oct21'; 'Oct21'; 'Oct21'; 'Oct21'; ...
    'Oct25'; 'Oct25'; 'Oct25'; 'Oct25'];
tide0 = {'horn1'; 'horn2'; 'horn3'; 'bay1'; 'bay2'; 'bay3'; 'horn1'; 'horn2'; 'bay1'; 'bay2'};
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% date_str0 = ['22_10_2018'];
% tide0 = ['AM'];

for kk = 1:length(tide0)

date_str = date_str0(kk, :);
tide = tide0{kk, :};
% imset = 'longshore1'; %'cross_shore'; % longshore; longshore1; longshore2, dense_array1; dense_array2
imset = 'dense_array2'; %'cross_shore'; % longshore; longshore1; longshore2, dense_array1; dense_array2

% dn = ['C:\Projects\AdvocateBeach2018\data\raw\images\LabValidation\' date_str '_' tide '\' ];
dn = ['C:\Projects\AdvocateBeach2018\data\raw\images\OutdoorValidation\' date_str '_' tide '\' ];
dnout = dn;
% dn = ['C:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\' date_str '\' tide '\' imset '\'];
% dnout = ['C:\Projects\AdvocateBeach2018\data\processed\images\cropped\beach_surveys\' date_str '\' tide '\' imset];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % FOR PICAM IMAGES
% dn = ['C:\Projects\AdvocateBeach2018\data\interim\images\PiCameras\tide15\pi74\'];
% dnout = ['C:\Projects\AdvocateBeach2018\data\processed\images\cropped\pi_cameras\tide15\pi74'];
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% create the folder if it doesn't exist already.
if notValidation == 1
    if ~exist(dnout, 'dir')
      mkdir(dnout);
    end
end

% 21-B
% dn = 'C:\Projects\AdvocateBeach2018\data\raw\images\BeachSurveys\21_10_2018\PM\cross_shore\';
% dnout = 'C:\Projects\AdvocateBeach2018\data\processed\images\cropped\beach_surveys\21_10_2018\PM\cross_shore\';

% fn = 'IMG_1755.jpg';



imnames = dir([dn 'img*']);

counter = 0;

for ii = 1:length(imnames)
    
    counter = counter + 1;
    
    if mod(counter, 2) == 0;

        img = imread([dn imnames(ii).name]);
%         img = imread([dn 'img1540667045-106253.jpg']);
    %     img = imread([dn fn]);

%         figure(100)
%         image(img)

        if cameraHeight == 1

            % mask dims
            hght = floor(size(img, 1)/2);
            wdth = floor(size(img, 2)/2);
            
            if strcmp(dn,'C:\Projects\AdvocateBeach2018\data\interim\images\PiCameras\tide27\pi74\')
            % avoid large water droplet on lens cover    
                
                % mask origin
                h0 = floor(size(img, 1)/8*3);
                w0 = floor(size(img, 2)/4);
                
            else
                
                % mask origin
                h0 = floor(size(img, 1)/4);
                w0 = floor(size(img, 2)/4);
                
            end

        elseif cameraHeight == 2

            % mask dims
            hght = floor(size(img, 1)/1.5);
            wdth = floor(size(img, 2)/1.5);

            % mask origin
            h0 = floor(size(img, 1)/6);
            w0 = floor(size(img, 2)/6);
            
        elseif cameraHeight == 3 % validation

            % mask dims
            hght = floor(size(img, 1) - 2*1150);
            wdth = floor(size(img, 2) - 2*1350);

            % mask origin
            h0 = 1150;
            w0 = 1350;    

        end

        newimg = imcrop(img, [w0, h0, wdth, hght]);

        imwrite(newimg, [dnout '\' imnames(ii).name(1:end-4) '-cropped.jpg'])

%         figure(ii+1), clf
%             image(newimg)
    
    end % mod

end % imnames


end % kk
