function [R_pos,R_neg,R_both,P_pos,P_neg,P_both] = CPM_LOOCV(thresh_pos, thresh_neg, all_mats, all_behav, all_cov, corr_type, regress_method)
%% This function is to use the optimal threshold and select the features in the positive and negative networks (no normalization)
% Input:
%       thresh_pos:     the optimal p threshold of positive network (e.g., thresh_pos(1,1))
%       thresh_neg:     the optimal p threshold of negative network (e.g., thresh_neg(1,1))
%       all_mats:       an X*Y*Z 3D matrix
%       all_behav:      a Z*1 vector
%       all_cov:        a Z*M vector contains different covariables (can be empty)
%       corr_type:      correlation type: 'Pearson', 'Kendall', 'Spearman'
%       regress_method: 1-polyfit, 2-regress, 3-robustfit (for 2&3, the first output is constant term)

% Output:
%       R_pos:  r value of the positive network
%       R_neg:  r value of the negative network
%       R_both: r value of the combined network
%       P_pos:  p value of the positive network
%       P_neg:  p value of the negative network
%       P_both: p value of the combined network

%%%%% The script is adopted from Xinlin Shen (2017, Nature Protocols). 
%%%%% Written by Mengxia Gao, PhD student of Dept. of Psychology, the University of Hong Kong.
%%%%% mengxia.gao@gmail.com, 20200310
%%%%% updated on 20201022

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

for leftout = 1:no_sub
    fprintf('\n Leaving out subj # %6.3f',leftout);
    
    %-------(Step 2) Divide data into training and testing sets
    % leave out subject from matrices and behavior
    
    train_mats = all_mats;
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3));

    train_behav = all_behav;
    train_behav(leftout) = [];
    
    %-------(Step 3) Relate connectivity to behavior
    % correlate all edges with behavior

    ind = isempty(all_cov);
    if ind == 0
        train_cov = all_cov;
        train_cov(leftout,:) = [];
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
    
    pos_mask(r_mat > 0 & p_mat < thresh_pos) = 1;
    neg_mask(r_mat < 0 & p_mat < thresh_neg) = 1;
    
    %-------(Step 5) Calculate single-subject summary values
    % get sum of all edges in TRAIN subs (divide by 2 to control for the
    % fact that matrices are symmetric)
    
    train_sumpos = zeros(no_sub-1,1);
    train_sumneg = zeros(no_sub-1,1);
    
    for ss = 1:no_sub-1
        train_sumpos(ss) = sum(sum(train_mats(:,:,ss).*pos_mask, 'omitnan'));
        train_sumneg(ss) = sum(sum(train_mats(:,:,ss).*neg_mask, 'omitnan'));
    end
    train_sum_all = train_sumpos - train_sumneg;
    
    %-------(Step 6) Model fitting
    % build model on TRAIN subs
    if regress_method == 1
            fit_pos = polyfit(train_sumpos, train_behav,1);
            fit_neg = polyfit(train_sumneg, train_behav,1);
            fit_both = polyfit(train_sum_all, train_behav,1); 
        
    elseif regress_method == 2
            fit_pos = regress(train_behav, [ones(no_sub-1,1), train_sumpos]);
            fit_neg = regress(train_behav, [ones(no_sub-1,1), train_sumneg]);
            fit_both = regress(train_behav, [ones(no_sub-1,1), train_sum_all]);
            
    elseif regress_method == 3
            fit_pos = robustfit(train_sumpos, train_behav);
            fit_neg = robustfit(train_sumneg, train_behav);
            fit_both = robustfit(train_sum_all, train_behav); 
    else
        warning('Error building linear model, please select a regression method: 1-polyfit, 2-regress, 3-robustfit');
    end
      
    % run model on TEST sub
    
    test_sumpos = sum(sum(all_mats(:,:,leftout).*pos_mask, 'omitnan'));
    test_sumneg = sum(sum(all_mats(:,:,leftout).*neg_mask, 'omitnan'));
    test_sum_all = test_sumpos - test_sumneg;
    
    if regress_method == 1
        behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
        behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);
        behav_pred_both(leftout) = fit_both(1)*test_sum_all(1) + fit_both(2);
        
    elseif regress_method == 2
        behav_pred_pos(leftout) = fit_pos(1) + fit_pos(2)*test_sumpos;
        behav_pred_neg(leftout) = fit_neg(1) + fit_neg(2)*test_sumneg;
        behav_pred_both(leftout) = fit_both(1)+ fit_both(2)*test_sum_all(1);
            
    elseif regress_method == 3
        behav_pred_pos(leftout) = fit_pos(1) + fit_pos(2)*test_sumpos;
        behav_pred_neg(leftout) = fit_neg(1) + fit_neg(2)*test_sumneg;
        behav_pred_both(leftout) = fit_both(1)+ fit_both(2)*test_sum_all(1);
    else
        warning('Error building linear model, please select a regression method: 1-polyfit, 2-regress, 3-robustfit');
    end
    
end

%------(Step 7) Prediction in novel subjects
% compare predicted and observed scores
[R_pos, P_pos] = corr(behav_pred_pos,all_behav,'type', corr_type);
[R_neg, P_neg] = corr(behav_pred_neg,all_behav,'type', corr_type);
[R_both, P_both] = corr(behav_pred_both,all_behav,'type', corr_type);

end