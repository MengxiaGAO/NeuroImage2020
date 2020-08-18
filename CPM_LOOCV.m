function [R_pos,R_neg,R_both,P_pos,P_neg,P_both] = CPM_LOOCV(thresh_pos, thresh_neg, all_mats, all_behav, all_cov, corr_type, regress_method)
%% This function is to use the optimal threshold and select the features in the positive and negative networks (behavioral scores normalized within the loop)
% Input:
%       thresh_pos:     the p threshold of positive network
%       thresh_neg:     the p threshold of negative network
%       all_mats:       an X*Y*Z 3D matrix
%       all_behav:      a Z*1 vector
%       all_cov:        a Z*M vector contains different covariables (can be empty)
%       corr_type:      correlation type: 'Pearson', 'Kendall', 'Spearman'
%       regress_method: 1-polyfit, 2-regress

% Output:
%       R_pos:  r value of the positive network
%       R_neg:  r value of the negative network
%       R_both: r value of the combined network (positive - negative)
%       P_pos:  p value of the positive network
%       P_neg:  p value of the negative network
%       P_both: p value of the combined network

%%%%% The script is adopted from Xinlin Shen (2017, Nature Protocols). 
%%%%% Written by Mengxia Gao, PhD candidate of Dept. of Psychology, the University of Hong Kong.
%%%%% mengxia.gao@gmail.com, 20200310
%%%%% updated on 20200508

no_sub = size(all_mats,3);
no_node1 = size(all_mats,1);
no_node2 = size(all_mats,2);

behav_pred_pos = zeros(no_sub,1);
behav_pred_neg = zeros(no_sub,1);
behav_pred_both = zeros(no_sub,1);

% all_train_sumpos = zeros(no_sub-1,no_sub);
% all_train_sumneg = zeros(no_sub-1,no_sub);
% 
% all_test_sumpos = zeros(1,no_sub);
% all_test_sumneg = zeros(1,no_sub);

for leftout = 1:no_sub;
    fprintf('\n Leaving out subj # %6.3f',leftout);
    
    %-------(Step 2) Divide data into training and testing sets
    % leave out subject from matrices and behavior
    
    train_mats = all_mats;
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3));

    train_behav = all_behav;
    train_behav(leftout) = [];
    train_behav = zscore(train_behav);
    %-------(Step 3) Relate connectivity to behavior
    % correlate all edges with behavior

    ind = isempty(all_cov);
    if ind == 0
        train_cov = all_cov;
        train_cov(leftout,:) = [];
        train_cov = zscore(train_cov);
        [r_mat, p_mat] = partialcorr(train_vcts', train_behav, train_cov, 'type', corr_type);
        
    elseif ind == 1
        [r_mat, p_mat] = corr(train_vcts', train_behav, 'type', corr_type);
        
    else
        warning('Error correlating the brain and the behavior, please check your data and correlation_method');
    end
    
    r_mat = reshape(r_mat, no_node1, no_node2);
    p_mat = reshape(p_mat, no_node1, no_node2);
    
    %-------(Step 4) Edge selection
    % set threshold and define masks
    
    pos_mask = zeros(no_node1,no_node2);
    neg_mask = zeros(no_node1,no_node2);
    
    pos_edges = find(r_mat > 0 & p_mat < thresh_pos);
    neg_edges = find(r_mat < 0 & p_mat < thresh_neg);
    
    pos_mask(pos_edges) = 1;
    neg_mask(neg_edges) = 1;
    
    %-------(Step 5) Calculate single-subject summary values
    % get sum of all edges in TRAIN subs (divide by 2 to control for the
    % fact that matrices are symmetric)
    
    train_sumpos = zeros(no_sub-1,1);
    train_sumneg = zeros(no_sub-1,1);
    
    for ss = 1:size(train_sumpos)
        pos_select = train_mats(:,:,ss).*pos_mask;
        pos_select(isnan(pos_select)) = 0;    % turn the NaN value into zero
        train_sumpos(ss) = sum(sum(pos_select))/2;  % testing divided by 2, the R value did not change
        
        neg_select = train_mats(:,:,ss).*neg_mask;
        neg_select(isnan(neg_select)) = 0;    % turn the NaN value into zero
        train_sumneg(ss) = sum(sum(neg_select))/2;  % testing divided by 2, the R value did not change
    end
    
%     all_train_sumpos(:,leftout) = train_sumpos;
%     all_train_sumneg(:,leftout) = train_sumneg;
    train_sum_all = train_sumpos - train_sumneg;

    %-------(Step 6) Model fitting
    % build model on TRAIN subs
    if regress_method == 1
            fit_pos = polyfit(train_sumpos, train_behav,1);
            fit_neg = polyfit(train_sumneg, train_behav,1);
            fit_both = polyfit(train_sum_all, train_behav,1); % combined features
        
    elseif regress_method == 2
            fit_pos = regress(train_behav, [train_sumpos, ones(no_sub-1,1)]);
            fit_neg = regress(train_behav, [train_sumneg, ones(no_sub-1,1)]);
            fit_both = regress(train_behav, [train_sum_all, ones(no_sub-1,1)]); % combined features
    else
        warning('Error building linear model, please select a regression method: 1-polyfit, 2-regress');
    end
      
    % run model on TEST sub
    
    test_mat = all_mats(:,:,leftout);
    test_pos_select = test_mat.*pos_mask;
    test_pos_select(isnan(test_pos_select)) = 0;    % turn the NaN value into zero
    test_sumpos = sum(sum(test_pos_select))/2;  % testing divided by 2, the R value did not change
    
    test_neg_select = test_mat.*neg_mask;
    test_neg_select(isnan(test_neg_select)) = 0;    % turn the NaN value into zero
    test_sumneg = sum(sum(test_neg_select))/2;  % testing divided by 2, the R value did not change
    
%     all_test_sumpos(1,leftout) = test_sumpos;
%     all_test_sumneg(1,leftout) = test_sumneg;
    test_sum_all = test_sumpos - test_sumneg;
    
    behav_pred_both(leftout) = fit_both(1)*test_sum_all(1) + fit_both(2);
    behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
    behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);
    
end

%------(Step 7) Prediction in novel subjects
% compare predicted and observed scores
[R_both, P_both] = corr(behav_pred_both,all_behav,'type', corr_type);
[R_pos, P_pos] = corr(behav_pred_pos,all_behav,'type', corr_type);
[R_neg, P_neg] = corr(behav_pred_neg,all_behav,'type', corr_type);

end