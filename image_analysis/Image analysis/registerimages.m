%Set current directory
function [] = registerimages(path)
%get list of phase images and folders
cd(path)

folders = dir('Unregistered/PH/');
saveloc = 'Registered/PH/';

greenpath = 'Unregistered/GRU/';
savgreenpath = 'Registered/GRU/';

mgreenpath = 'Unregistered/GM/';
savmgreenpath = 'Registered/GM/';

mphasepath = 'Unregistered/PM/';
savmphasepath = 'Registered/PM/';


folders = folders(~ismember({folders.name},{'.','..'}));
nfolders = length(folders);
nfolders
k = 0;

for i = 1:nfolders
    
    imagefiles = dir(fullfile(folders(i).folder,folders(i).name,'*tif'));
    nfiles = length(imagefiles);
    
    for j = 1:nfiles
        j
        if(j==1)
           carryforward = imread(fullfile(imagefiles(j).folder,imagefiles(j).name));
           
           mphasename = imagefiles(j).name;
           mphase = imread(fullfile(mphasepath,folders(i).name,mphasename));
           
           greenname = ['GRU' imagefiles(j).name(3:end)];
           green = imread(fullfile(greenpath,folders(i).name,greenname));
           mgreen = imread(fullfile(mgreenpath,folders(i).name,greenname));
                      
           imwrite(carryforward,fullfile(saveloc,imagefiles(j).name));
           imwrite(mphase,fullfile(savmphasepath,mphasename));
           imwrite(green,fullfile(savgreenpath,greenname));
           imwrite(mgreen,fullfile(savmgreenpath,greenname));
           k=k+1
           
       else
          
           core = carryforward;
           regimage = imread(fullfile(imagefiles(j).folder,imagefiles(j).name));
           
           mphasename = imagefiles(j).name;
           mphase = imread(fullfile(mphasepath,folders(i).name,mphasename));
           
           greenname = ['GRU' imagefiles(j).name(3:end)];
           green = imread(fullfile(greenpath,folders(i).name,greenname));
           mgreen = imread(fullfile(mgreenpath,folders(i).name,greenname));
           
           [carryforward,greensav,mphasesav,mgreensav] = phaseonlyreg(core,regimage,green,mphase,mgreen);
           
           imwrite(carryforward,fullfile(saveloc,imagefiles(j).name));
           imwrite(mphasesav,fullfile(savmphasepath,mphasename));
           imwrite(greensav,fullfile(savgreenpath,greenname));
           imwrite(mgreensav,fullfile(savmgreenpath,greenname));
           
           k=k+1
           
       end
       
    end
        
end
end

function [regN,greenN,mphaseN,mgreenN] = phaseonlyreg(core,reg,green,mphase,mgreen)

%Register images using a binary mask approach


% Find transformation
ph=imregcorr(reg,core,'translation');

% Execute transformation
coreimageref = imref2d(size(core));
regimageref = imref2d(size(reg));
regN = imwarp(reg,regimageref,ph,'OutputView',coreimageref);
greenN = imwarp(green,regimageref,ph,'OutputView',coreimageref);
mphaseN = imwarp(mphase,regimageref,ph,'OutputView',coreimageref);
mgreenN = imwarp(mgreen,regimageref,ph,'OutputView',coreimageref);


end