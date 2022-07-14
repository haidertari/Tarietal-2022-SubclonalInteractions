%Set current directory
function [] = phaseregisterimages(path)
%get list of phase images and folders
cd(path)

folders = dir('Unregistered/PH/');
saveloc = 'Registered/PH/';

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
           
                
           imwrite(carryforward,fullfile(saveloc,imagefiles(j).name));
           imwrite(mphase,fullfile(savmphasepath,mphasename));
           k=k+1
           
       else
          
           core = carryforward;
           regimage = imread(fullfile(imagefiles(j).folder,imagefiles(j).name));
           
           mphasename = imagefiles(j).name;
           mphase = imread(fullfile(mphasepath,folders(i).name,mphasename));
           
           [carryforward,mphasesav] = phaseonlyreg(core,regimage,mphase);
           
           imwrite(carryforward,fullfile(saveloc,imagefiles(j).name));
           imwrite(mphasesav,fullfile(savmphasepath,mphasename));
           
           k=k+1
           
       end
       
    end
        
end
end

function [regN,mphaseN] = phaseonlyreg(core,reg,mphase)

%Register images using a binary mask approach


% Find transformation
ph=imregcorr(reg,core,'translation');

% Execute transformation
coreimageref = imref2d(size(core));
regimageref = imref2d(size(reg));
regN = imwarp(reg,regimageref,ph,'OutputView',coreimageref);
mphaseN = imwarp(mphase,regimageref,ph,'OutputView',coreimageref);


end