%% load images extract some keypoints and extract patch
nr_patches = 50000;
patch_size = 32;
rho = 180/pi;
patch_size2 = patch_size/2;
patches = uint8(zeros(patch_size,patch_size,nr_patches));
currImg = 0;
created_patches = 1;
nr_keypts = 100;

keyptExtractors = {'AKAZE','ORB','SURF'};
orbExtractor = cv.ORB('NLevels',1);
akazeExtractor = cv.AKAZE('NOctaves',1,'NOctaveLayers',1);
surfExtractor = cv.SURF('NOctaves',1,'NOctaveLayers',1);

datastr = 'JPEGImages/';
imageNames = dir(strcat(datastr,'*.jpg'));
for kp=1:size(keyptExtractors,2)
    currImg = 1;
    orientation =[];
    while (created_patches <= nr_patches)
        img_path = strcat(datastr,'/',imageNames(currImg).name);
        currImg = currImg+1;
        try
        I = imread(img_path);
        catch
            break;
        end
        if (length(size(I)) >=3)
            I = rgb2gray(I);
        end
        [~, Iw] = size(I);
        % resize image
        scale =1;
        if (Iw > 1000)
            scale = 1000/Iw;
        end
        I = imresize(I,scale);
        [Ih, Iw] = size(I);
        mask = zeros(Ih,Iw);
        mask(patch_size:Ih-patch_size, patch_size:Iw-patch_size) = ...
            ones(Ih-2*patch_size+1,Iw-2*patch_size+1);
        if strcmp(keyptExtractors{kp},'ORB')
            [keypoints,descriptors] = orbExtractor.detectAndCompute(I);
        elseif strcmp(keyptExtractors{kp},'AKAZE')
            [keypoints,desc] = akazeExtractor.detectAndCompute(I,'Mask',mask);
        elseif strcmp(keyptExtractors{kp},'SURF')
            [keypoints] = surfExtractor.detect(I,'Mask',mask);
        end

        % take the strongest keypoints
        [vec,idx] = sort([keypoints.response]','descend');
        keypoints = keypoints(idx);
        if size(keypoints,2) > nr_keypts
            keypoints = keypoints(1:nr_keypts);
        end

        nr_pts_in_img = size(keypoints,2);
        pointsI1xy = reshape([keypoints(1:nr_pts_in_img).pt],2,nr_pts_in_img);
        % this is a opencv bug and might change in future version
        if strcmp(keyptExtractors{kp},'AKAZE')
            orientation = [orientation [keypoints(1:nr_pts_in_img).angle]];
        else
            orientation = [orientation [keypoints(1:nr_pts_in_img).angle]./rho];
        end
%             figure(10)
%             imshow(I)
%             hold on
%             plot(pointsI1xy(1,:),pointsI1xy(2,:),'r+')
%             pause
        % extract gaussian filtered patch
        If = imgaussfilt(I,2);
        for p=1:nr_pts_in_img
            patches(:,:,created_patches) = If(...
                round(pointsI1xy(2,p))-patch_size2+1:round(pointsI1xy(2,p))+patch_size2,...
                round(pointsI1xy(1,p))-patch_size2+1:round(pointsI1xy(1,p))+patch_size2);
            created_patches = created_patches+1;
            if created_patches >= nr_patches
                orientation = orientation(1:nr_patches);
                break;
            end
        end
    end
    created_patches = 1;
    currImg = 1;
    savestr=sprintf('patches_50k_%s_voc',keyptExtractors{kp});
    save(savestr,'patches','orientation');
end
