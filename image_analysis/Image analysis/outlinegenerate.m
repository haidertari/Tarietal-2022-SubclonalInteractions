function [] = outlinegenerate(path)
%Generate outlines to assist in mask analysis
cd (path)

pfolders = dir('Stage3/Filtered/PM/');
gfolders = dir('Stage3/Filtered/GM/');

rawpfolders = 'Stage3/Registered/PH/';
rawgfolders = 'Stage3/Registered/GRU/';

savploc = 'Stage3/Outline/PH/';
savgloc = 'Stage3/Outline/GRU/';

for i = 1:length(pfolders)
   
    imagefiles = dir(fullfile(pfolders(i).folder,pfolders(i).name,'*tif'));
    nfiles = length(imagefiles);
    
    for j = 1:nfiles
        
        mask = imread(fullfile(imagefiles(j).folder,imagefiles(j).name));
        img = imread(fullfile(rawpfolders,imagefiles(j).name));
        
        outline = imdilate(bwperim(mask),strel('sphere',2));
        output = imoverlay(img,outline);
        
        filename = fullfile(savploc,imagefiles(j).name);
        imwrite(output,filename)
        
    end
    
end

for i = 1:length(gfolders)
   
    imagefiles = dir(fullfile(gfolders(i).folder,gfolders(i).name,'*tif'));
    nfiles = length(imagefiles);
    
    for j = 1:nfiles
        
        mask = imread(fullfile(imagefiles(j).folder,imagefiles(j).name));
        img = imread(fullfile(rawgfolders,imagefiles(j).name));
        
        outline = imdilate(bwperim(mask),strel('sphere',2));
        output = imoverlay(img,outline);
        
        filename = fullfile(savgloc,imagefiles(j).name);
        imwrite(output,filename)
        
    end
    
end


end

