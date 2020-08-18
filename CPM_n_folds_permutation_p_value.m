function [pval_pos, pval_neg, pval_both, prediction_r] = CPM_n_folds_permutation_p_value (no_iterations, R_pos_repeat, R_neg_repeat, R_both_repeat, R_pos_permutation, R_neg_permutation, R_both_permutation)
%% This function is to calculate the permutation p value of n-fold CPM
% Input:
%       no_iterations:  the number of permutations (N, a number)
%       R_pos_repeat:   r values of n-fold in N repeating calculation (positive)
%       R_neg_repeat:   r values of n-fold in N repeating calculation (negative)
%       R_pos_permutation:   r values of n-fold in N permutation (positive)
%       R_neg_permutation:   r values of n-fold in N permutation (negative)

% Output:
%       pval_pos:     p_value of permutation (positive)
%       pval_neg:     p value of permutation (negative)
%       pval_both:    p value of permutation (combined network)
%       prediction_r: a list of all r values in the permutation

%%%%% The script is adopted from Xinlin Shen (2017, Nature Protocols). 
%%%%% Written by Mengxia Gao, PhD candidate of Dept. of Psychology, the University of Hong Kong.
%%%%% mengxia.gao@gmail.com, 20200509

%     [dim1,dime2] = size(R_pos_repeat);      
%     true_prediction_r_pos = mean(R_pos_repeat,[dim1,dime2]);  %(can only be used after Matlab2018b)
    true_prediction_r_pos = mean(mean(R_pos_repeat));
    true_prediction_r_neg = mean(mean(R_neg_repeat));
    true_prediction_r_both = mean(mean(R_both_repeat));

    prediction_r = zeros(no_iterations+1, 3);
    prediction_r(1,1) = true_prediction_r_pos;
    prediction_r(1,2) = true_prediction_r_neg;
    prediction_r(1,3) = true_prediction_r_both;
    
    prediction_r(2:end,1) = (mean (R_pos_permutation))';
    prediction_r(2:end,2) = (mean (R_neg_permutation))';
    prediction_r(2:end,3) = (mean (R_both_permutation))';
    
    sorted_prediction_r_pos = sort (prediction_r(:,1),'descend');
    position_pos            = find (sorted_prediction_r_pos == true_prediction_r_pos);
    pval_pos                = position_pos(1)/(no_iterations+1);

    sorted_prediction_r_neg = sort (prediction_r(:,2),'descend');
    position_neg            = find (sorted_prediction_r_neg == true_prediction_r_neg);
    pval_neg                = position_neg(1)/(no_iterations+1);
    
    sorted_prediction_r_both = sort (prediction_r(:,3),'descend');
    position_both            = find (sorted_prediction_r_both == true_prediction_r_both);
    pval_both                = position_both(1)/(no_iterations+1);

end