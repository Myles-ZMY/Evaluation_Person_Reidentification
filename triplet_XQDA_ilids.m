close all; clear; clc;
addpath ./lib

Varin =  importdata('/Users/Roll/Riceroll/CNN/prp/LOMO_XQDA/code/feature.mat');
feat_test = Varin.feat;
feat_p_test = Varin.feat_p;

% for i=1:150
%     for j=1:150
%         dist_mat(i,j)=norm((feat_test(i,:)-feat_p_test(j,:)),2);
%     end
% end
% cms = EvalCMC( -dist_mat, 1:150, 1:150, 50 );

sizeFeat = size(feat_test);
numClass = sizeFeat(1);
numFolds = 10;
numRanks = floor(numClass/2);
cms = zeros(numFolds, numRanks);
for nf = 1 : numFolds
    p = randperm(numClass);
    
    galFea1 = feat_test( p(1:numClass/2), : );
    probFea1 = feat_p_test( p(1:numClass/2), : );
    
    t0 = tic;
    [W, M] = XQDA(galFea1, probFea1, (1:numClass/2)', (1:numClass/2)');

    clear galFea1 probFea1
    trainTime = toc(t0);
    
    galFea2 = feat_test(p(numClass/2+1 : end), : );
    probFea2 = feat_p_test(p(numClass/2+1 : end), : );
    
    t0 = tic;
    dist = MahDist(M, galFea2 * W, probFea2 * W);
    clear galFea2 probFea2 M W
    matchTime = toc(t0);
    
    fprintf('Fold %d: ', nf);
    fprintf('Training time: %.3g seconds. ', trainTime);    
    fprintf('Matching time: %.3g seconds.\n', matchTime);
    
    cms(nf,:) = EvalCMC( -dist, 1 : numClass / 2, 1 : numClass / 2, numRanks );
    clear dist
    
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(nf,[1,5,10,15,20]) * 100);
end

meanCms = mean(cms);
plot(1 : numRanks, meanCms);
