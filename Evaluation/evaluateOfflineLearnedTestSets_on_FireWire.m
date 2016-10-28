close all
clear all
%% constants
desc_size = 512;
patch_size = 32;
patch_size2 = patch_size/2;
seed = 42;
% nr_keypts = end;
gauss = 0;
rho = 180/pi;
nr_bins = 30;

%% declare detectors and descriptors
detectorBRISK = cv.BRISK;
detectorAKAZE = cv.AKAZE;
detectorORB = cv.ORB;
%% point that to the data folder
base_string = '../../testInstallation/Dataset_Planar/FireWire_10percent/online_data/';
load('../Dataset/CalibDataFireWireOcam.mat')
load('../Dataset/gt_homography_Firewire.mat')

%% test
nrImages = 13;
descriptors = cell(1,nrImages);
keypoints = cell(1,nrImages);
pointsI1 = 1;
features = cell(1,nrImages);
points = cell(1,nrImages);
Images = cell(1,nrImages);
Imagesu = cell(1,nrImages);


nrTestSets = 6;
nrDetectors = 2*nrTestSets+3;

descriptor_masks = cell(1,nrImages);
nrLevels = 8;
imgPyramid = cell(nrImages,nrLevels);
imgPyramidSmooth = cell(nrImages,nrLevels);
scaleFactor = 1.2;
border = 50;
nrKeyPoints = 200;
for im = 1:nrImages
    descriptors{im}.Descs = cell(1,nrTestSets); % for each detector 3 descriptors
    descriptors{im}.Masks = cell(1,2*nrTestSets); % bold and dBRIEF
    keypoints{im}.kp = cell(1,nrDetectors); 

    points{im}.pts = cell(1,nrDetectors);
    points{im}.ptsUndist = cell(1,nrDetectors);
    s = strcat(base_string,sprintf('img%d.png',im));

    Images{im} = rgb2gray(imread(s));

    % calculate smoothed image pyramid, need that for BOLD
    scalePoints = ocam_model.width/scaleImage;
    for s=1:nrLevels
        Ir = imresize(Images{im},1/getScale(s-1,0,scaleFactor));
        imgPyramid{im,s} = Ir;
        imgPyramidSmooth{im,s} = cv.GaussianBlur(Ir,'KSize',[7,7],'SigmaX',2,'SigmaY',2);
    end

    [keypoints{im}.kp{1},descriptors{im}.Descs{1}] = detectorBRISK.detectAndCompute(Images{im});
    [keypoints{im}.kp{2},descriptors{im}.Descs{2}] = detectorAKAZE.detectAndCompute(Images{im});
    [keypoints{im}.kp{3},descriptors{im}.Descs{3}] = detectorORB.detectAndCompute(Images{im});
    % sort and remove 
    for k=1:3
        [vec,idx] = sort([keypoints{im}.kp{k}.response]','descend');
        keypoints{im}.kp{k} = keypoints{im}.kp{k}(idx);

        nr_keypts = nrKeyPoints;
        keypoints{im}.kp{k} = keypoints{im}.kp{k}(1:nr_keypts);
        descriptors{im}.Descs{k} = descriptors{im}.Descs{k}(idx,:);
        descriptors{im}.Descs{k} = descriptors{im}.Descs{k}(1:nr_keypts,:);
        pointsI1 = keypoints{im}.kp{k};
        pointsI1xy = reshape([pointsI1.pt],2,size(keypoints{im}.kp{k},2));
        pointsI1xy_u = undistort_ocam([pointsI1xy(1,:);pointsI1xy(2,:)], ocam_model,-scalePoints);   
        points{im}.pts{k} = pointsI1xy;
        points{im}.ptsUndist{k} = pointsI1xy_u;  
    end
    
    %% now extract out learned desctiptors
    nrPts = nrKeyPoints; learnMasks = 1; useAgast = 0; 
    fastType = 2; nLevels = 8; doOrientation = 1; desc_sizes = 256;
    testSet = 0;
    for te=4:2:(2*nrTestSets)+3
        % now you can select a test set. see mdBRIEFFireWire.cpp for details
        [keypoints{im}.kp{te},descriptors{im}.Descs{te},~] = cv.mdBRIEFFireWire(Images{im}, ...
            nrPts, nLevels, 0, fastType, useAgast,...
            desc_sizes, doOrientation, testSet);

        [keypoints{im}.kp{te+1},descriptors{im}.Descs{te+1},descriptors{im}.Masks{te-2}] = cv.mdBRIEFFireWire(Images{im}, ...
            nrPts, nLevels, learnMasks, fastType, useAgast,...
            desc_sizes, doOrientation, testSet);   
        testSet = testSet+1;
    end
    for k=4:(2*nrTestSets)+3
        pointsI1 = keypoints{im}.kp{k};
        pointsI1xy = reshape([pointsI1.pt],2,size(keypoints{im}.kp{k},2));
        pointsI1xy_u = undistort_ocam([pointsI1xy(1,:);pointsI1xy(2,:)], ocam_model,-scalePoints);   
        points{im}.pts{k} = pointsI1xy;
        points{im}.ptsUndist{k} = pointsI1xy_u;  
    end
end

maxComb = 5;
% nrDescriptors = 3*nrDetectors;
circle_thresh = 3;
pr_curves = cell(size(allcombs(1:maxComb,:),1),nrDetectors);
tested_pairs = 0;
for c=1:size(allcombs(1:maxComb,:),1)
    %% first step: project the points using the gt homography
    idx1 = allcombs(c,1);
    idx2 = allcombs(c,2);
    if ~gt_homography{idx1,idx2}.bad   
        tested_pairs = tested_pairs+1;
        for k=1:nrDetectors
            pts1u = points{idx1}.ptsUndist{k};
            pts2u = points{idx2}.ptsUndist{k};
            pts1 = points{idx1}.pts{k};
            pts2 = points{idx2}.pts{k};
            H = gt_homography{idx1,idx2}.H;
            nr_keypts1 = size(pts1u,2);
            p_projected = H*[pts1u;ones(1,nr_keypts1)];
            threshs = linspace(1,size(descriptors{idx1}.Descs{k},2)*8,50);
            
            % normalize
            for p=1:nr_keypts1
                p_projected(:,p) = p_projected(:,p)/p_projected(3,p);
            end
            p_projected = p_projected(1:2,:);
        
            %% compute geometric neighbors
            %  find the correct matches by geometry
            nn = findNearestNeighbor(p_projected,pts2u,circle_thresh);
            [idx2s] = nn~=-1;
            idx2s = logical(idx2s(:,1).*idx2s(:,2));
            %% get the indices of relevant features, i.e. existing matches,
            % dicard all other features
            idxs_feat1 = nn(idx2s,1);
            idxs_feat2 = nn(idx2s,2);
        
%             showMatchedFeatures(Images{idx1},Images{idx2},pts1(:,nn(idx2s,1))',pts2(:,nn(idx2s,2))')
            number_real_true_matches = sum(idx2s);

            pr_curves{c,k}.stats = [0,0,0,0,0,0];
            pr_curves{c,k}.curves = [zeros(length(threshs),1),zeros(length(threshs),1)];

            recall = zeros(length(threshs),1);
            prec1m = zeros(length(threshs),1);
            nrMatched = zeros(length(threshs),1);
            for tr=1:length(threshs)
                if k==1 || k==2 || k == 3 || mod(k,2)==0
                    desc1 = uint8(descriptors{idx1}.Descs{k});
                    desc2 = uint8(descriptors{idx2}.Descs{k});
                    matches12 = cv.matchHammingExhaustive(desc1,desc2,int32(threshs(tr)));
                elseif mod(k,2)==1
                    desc1 = uint8(descriptors{idx1}.Descs{k});
                    desc2 = uint8(descriptors{idx2}.Descs{k});
                    mask1 = uint8(descriptors{idx1}.Masks{k-3});
                    mask2 = uint8(descriptors{idx2}.Masks{k-3});
                    matches12 = cv.matchHammingMaskedExhaustive(...
                        desc1,desc2,mask1,mask2,int32(threshs(tr)));    
                end

                distances = [matches12.distance];

                indices_match1 = [matches12.queryIdx]+1;
                indices_match2 = [matches12.trainIdx]+1;
                M = indices_match2 ~= 0;

                matched1 = p_projected(:,indices_match1(M));
                matched2 = pts2u(:,indices_match2(M));

                dists = sqrt((matched2(1,:)-matched1(1,:)).^2+(matched2(2,:)-matched1(2,:)).^2);
                correct = dists < circle_thresh;
                sumM = sum(M); 
                if sumM == 0 % prevent division by 0
                    prec1m(tr,1) = 1;
                else
                    prec1m(tr,1) = sum(correct) / sumM;
                end
                recall(tr,1) = sum(correct) / number_real_true_matches;

                if isnan(recall(tr,1))
                   recall(tr,1) = 0;
                end
                nrMatched(tr,1) = sumM;
            end
            pr_curves{c,k}.curves = [prec1m,recall,nrMatched];
            pr_curves{c,k}.stats = [length(idx2s),number_real_true_matches,sum(correct),sum(false)];
        end
    end
    disp(strcat(num2str(c),'/',num2str(maxComb),' image pairs finished'))
end

%% pr kurven mitteln
prec_mean = cell(1,nrDetectors);
recall_mean = cell(1,nrDetectors);
% for j=1:3
    for k=1:nrDetectors
        prec_mean1 = zeros(size(pr_curves{1,1}.curves,1),1);
        recall_mean1 = zeros(size(pr_curves{1,1}.curves,1),1);
        for c=1:size(allcombs(1:maxComb,:),1)
            idx1 = allcombs(c,1);
            idx2 = allcombs(c,2);
            if ~gt_homography{idx1,idx2}.bad
                prec_mean1 = prec_mean1+(1-pr_curves{c,k}.curves(:,1));
                recall_mean1 = recall_mean1+pr_curves{c,k}.curves(:,2);
            end
        end
        prec_mean{k} = prec_mean1./tested_pairs;
        recall_mean{k} = recall_mean1./tested_pairs;
    end
% end

% not masked
figure(1)
title('comparison off different datasets and detector for offline learning')
hold on
plot(prec_mean{1},recall_mean{1},'r')
plot(prec_mean{2},recall_mean{2},'g')
plot(prec_mean{3},recall_mean{3},'b')

plot(prec_mean{4},recall_mean{4},'r--')
plot(prec_mean{6},recall_mean{6},'g--')
plot(prec_mean{8},recall_mean{8},'b--')
plot(prec_mean{10},recall_mean{10},'m--')
plot(prec_mean{12},recall_mean{12},'c--')
plot(prec_mean{14},recall_mean{14},'black--')
plot(0,0,'w')
plot(0,0,'w')
axis([0 1 0 1])
legend('BRISK-512bit','AKAZE-488bit','ORB-256bit','orb+p-256bit','orb+f-256bit',...
'akaze+p-256bit','akaze+f-256bit','bold-256bit','brief-256bit','f -> learned on fisheye images','p -> learned on VOC2012')
set(gcf,'color','w');

figure(2)
title('comparison off different datasets and detector for offline learning - with online mask learning')
hold on
plot(prec_mean{1},recall_mean{1},'r')
plot(prec_mean{2},recall_mean{2},'g')
plot(prec_mean{3},recall_mean{3},'b')

plot(prec_mean{5},recall_mean{5},'r--')
plot(prec_mean{7},recall_mean{7},'g--')
plot(prec_mean{9},recall_mean{9},'b--')
plot(prec_mean{11},recall_mean{11},'m--')
plot(prec_mean{13},recall_mean{13},'c--')
plot(prec_mean{15},recall_mean{15},'black--')
plot(0,0,'w')
plot(0,0,'w')
axis([0 1 0 1])
legend('BRISK-512bit','AKAZE-488bit','ORB-256bit','orb+p-256bit','orb+f-256bit',...
'akaze+p-256bit','akaze+f-256bit','bold-256bit','brief-256bit','f -> learned on fisheye images','p -> learned on VOC2012')
set(gcf,'color','w');