% create one hot representation of passed integer based on passed size
function vec = oneHotFcn(int, max) 
%oneHot = bsxfun(@eq, 1:max(y),y);  %original code

%set to int+1 to account for first bit representing input 0
vec = bsxfun(@eq, 1:max, int+1);
vec = transpose(vec);   %turn into column vector

% source: https://stackoverflow.com/questions/23078287/create-a-zero-filled-2d-array-with-ones-at-positions-indexed-by-a-vector
