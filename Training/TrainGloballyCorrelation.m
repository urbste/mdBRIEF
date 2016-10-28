clear all
close all
patch_size = 32;
patch_size2 = 32/2;
desc_size = 512;
% correlation_thresh = 0.2;
max_nr_tests = 100000;
seed = 1;
max_nr_patches = 20000; % this should be smaller than the number of patches you extracted
% [alltests] = int32(CreateAllPossibleTests(patch_size));
[alltests] = int32(create_tests(patch_size, max_nr_tests,1));
% select a random number of tests, training over all tests takes very long
% alltests = alltests(:,randi(size(alltests,2),1,max_nr_tests));

% diplay tests
figure
mat = zeros(32,32);
imshow(mat)
hold on
line([alltests(1,:) alltests(3,:)], [alltests(2,:) alltests(4,:)])

min_mean = 0;
best_mean = 1;

corr_threshs = 0.2;
% choose detector dataset on which we want to train on 
keyptExtractors = 'ORB'; %'AKAZE', 'SURF'

% load patches
patchStr = sprintf('patches_50k_%s_voc.mat',keyptExtractors);
load(patchStr);

% reshape to opencv compatible format
patches_2_cv = uint8(zeros(max_nr_patches,patch_size*patch_size));
rotations_2_cv = single(zeros(max_nr_patches,1));
for j=1:max_nr_patches
    patches_2_cv(j,:) = reshape(patches(:,:,j),1,patch_size*patch_size);
    rotations_2_cv(j) = single(orientation(j));
end
dispStr = sprintf('learning tests for %s - features',keyptExtractors);
disp(dispStr)

disp('peforming tests on all patches');
means = cv.performTestOnAllPatches(patches_2_cv,alltests,rotations_2_cv);
means_sorted = zeros(2,size(means,2));
[means_sorted(1,:),idx] = sort(means(1,:),2);
means_sorted(2,:) = means(2,idx);
disp('finished testing and sorting...')
disp('starting to search for tests with low correlation. This takes some time...');
final_descriptor = ...
cv.findMinCorrelatedTests(patches_2_cv,rotations_2_cv,means_sorted(2,:),alltests,desc_size,corr_threshs);
disp('having our final descriptor');    

descSaveStr = sprintf('full_desc512_%s.mat',keyptExtractors);
save(descSaveStr,'final_descriptor')
% wirte a csv file
desc_csv_out = reshape(final_descriptor-int32(repmat(16,4,desc_size)),1,4*desc_size);
csvwrite(strcat('final_descriptor_randinit_',keyptExtractors,'_.csv'),desc_csv_out);

% diplay tests
% figure
% mat = zeros(32,32);
% imshow(mat)
% hold on
% line([final_descriptor(1,1:64) final_descriptor(3,1:64)], [final_descriptor(2,1:64) final_descriptor(4,1:64)])

mat = zeros(32,32);
figure(10)
imshow(mat)
hold on
line([bold(1:200,1) bold(1:200,3)]+16, [bold(1:200,2) bold(1:200,4)]+16,'Color','r')
hold off