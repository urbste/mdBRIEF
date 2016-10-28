function [ filtered_tests ] = CreateAllPossibleTests(patch_size)

dist_min = 4;
diago=sqrt(31^2+31^2);
dist_max = round(8*diago/10);
% patch_size2 = patch_size/2;

% we omit the border pixels
% idxs = -patch_size2+2:patch_size2-1;
idxs = 2:patch_size-1;
[x,y] = meshgrid(idxs,idxs);
x=x(:);
y=y(:);

% all combinations
comb1 = combnk(1:length(idxs)^2,2);
filtered_tests = [];
for i = 1:size(comb1,1)
    pt1 = [x(comb1(i,1)) y(comb1(i,1))];
    pt2 = [x(comb1(i,2)) y(comb1(i,2))];
    dist = sqrt((pt1(1)-pt2(1))^2+(pt1(2)-pt2(2))^2);
    
    if (dist > dist_min && dist < dist_max)
        filtered_tests = [filtered_tests [x(comb1(i,1)) y(comb1(i,1)) x(comb1(i,2)) y(comb1(i,2))]'];
    end

end


end

