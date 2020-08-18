function [R_pos_permutation, R_neg_permutation, R_both_permutation] = CPM_n_folds_permutation (nfolds, no_iterations, thresh_pos, thresh_neg, all_mats, all_behav, all_cov, corr_type, regress_method)
%% This function is to use n-fold to calculate the R value of CPM analysis
% Input:
%       nfolds:         n-folds (N, a number)
%       no_iterations:  the number of permutations (N, a number)
%       all_mats:       an X*Y*Z 3D matrix
%       all_behav:      a Z*1 vector
%       all_cov:        a Z*M vector contains different covariables (can be empty)
%       corr_type:      correlation type: 'Pearson', 'Kendall', 'Spearman'
%       regress_method: 1-polyfit, 2-regress

% Output:
%       R_pos_permutation:   r values of n-fold in N permutation (positive)
%       R_neg_permutation:   r values of n-fold in N permutation (negative)
%       R_both_permutation:  r values of n-fold in N permutation (combined)

%%%%% The script is adopted from Xinlin Shen (2017, Nature Protocols). 
%%%%% Written by Mengxia Gao, PhD candidate of Dept. of Psychology, the University of Hong Kong.
%%%%% mengxia.gao@gmail.com, 20200509

no_repeat = 1;
no_sub = size(all_mats,3);

R_pos_permutation = zeros(nfolds,no_iterations);
R_neg_permutation = zeros(nfolds,no_iterations);
R_both_permutation = zeros(nfolds,no_iterations);

for it = 1:no_iterations
    fprintf('\n Performing iteration %d out of %d', it, no_iterations);
    new_behav = all_behav (randperm(no_sub));
    
    [R_pos_repeat, R_neg_repeat, R_both_repeat] = CPM_n_folds (nfolds, no_repeat, thresh_pos, thresh_neg, all_mats, new_behav, all_cov, corr_type, regress_method);

    R_pos_permutation(:,it) = R_pos_repeat;
    R_neg_permutation(:,it) = R_neg_repeat;
    R_both_permutation(:,it) = R_both_repeat;

end
    
end