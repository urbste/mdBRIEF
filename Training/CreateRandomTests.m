% create a fixed number of random tests and save to file
function [tests] = create_tests(patch_size, nr_tests, seed)

rng(seed,'twister')
S = 1;
tests = floor(S+(patch_size-2*S)*rand(4,nr_tests))+ones(4, nr_tests);


end

