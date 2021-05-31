function [results, predict, feature, ModelShared, ModelAll, train_behav_all] = CPM_LOOCV_advance(thresh_pos, thresh_neg, all_mats, all_behav, all_cov, corr_type, regress_method)
%% This function is to use the optimal threshold and select the features in the positive and negative networks (no normalization on the behavioral variable)
% Input:
%       thresh_pos:     the optimal p threshold of positive network (e.g., thresh_pos(1,1))
%       thresh_neg:     the optimal p threshold of negative network (e.g., thresh_neg(1,1))
%       all_mats:       an X*Y*Z 3D matrix
%       all_behav:      a Z*1 vector
%       all_cov:        a Z*M vector contains different covariables (can be empty)
%       corr_type:      correlation type: 'Pearson', 'Kendall', 'Spearman'
%       regress_method: 1-polyfit, 2-regress, 3-robustfit (for 2&3, the first output is constant term)

% Output:
%       results:     R, Rsquared and P value (positive, negative, combined)
%       predict:     predicted values (positive, negative, combined)
%       feature:     summed values in each loop (positive, negative)
%       ModelShared: shared masks and beta values (positive, negative, combined)
%       ModelAll:    masks and beta values from all subjects (positive, negative, combined)

%%%%% The script is adopted from Xinlin Shen (2017, Nature Protocols). 
%%%%% Written by Mengxia Gao, PhD student of Dept. of Psychology, the University of Hong Kong.
%%%%% mengxia.gao@gmail.com, 20200418
%%%%% updated on 20201022

no_sub = size(all_mats,3);
no_node1 = size(all_mats,1);
no_node2 = size(all_mats,2);

predict.behav_pred_both = zeros(no_sub,1);
predict.behav_pred_pos = zeros(no_sub,1);
predict.behav_pred_neg = zeros(no_sub,1);

all_train_sumpos = zeros(no_sub-1,no_sub);
all_train_sumneg = zeros(no_sub-1,no_sub);

all_test_sumpos = zeros(1,no_sub);
all_test_sumneg = zeros(1,no_sub);

pos_edges_all = cell(1,no_sub);
neg_edges_all = cell(1,no_sub);

train_behav_all = zeros(no_sub-1,no_sub);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for leftout = 1:no_sub
    fprintf('\n Leaving out subj # %6.3f',leftout);
    
    %-------(Step 2) Divide data into training and testing sets
    % leave out subject from matrices and behavior
    
    train_mats = all_mats;
    train_mats(:,:,leftout) = [];
    train_vcts = reshape(train_mats,[],size(train_mats,3));

    train_behav = all_behav;
    train_behav(leftout) = [];
    %train_behav = zscore(train_behav);
    train_behav_all(:,leftout) = train_behav;
    %-------(Step 3) Relate connectivity to behavior
    % correlate all edges with behavior

    ind = isempty(all_cov);
    if ind == 0
        train_cov = all_cov;
        train_cov(leftout,:) = [];
        %train_cov = zscore(train_cov);
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
    
    pos_edges_all{leftout} = pos_edges;
    neg_edges_all{leftout} = neg_edges;
    
    pos_mask(pos_edges) = 1;
    neg_mask(neg_edges) = 1;
    
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
    
    all_train_sumpos(:,leftout) = train_sumpos;
    all_train_sumneg(:,leftout) = train_sumneg;
    
    %-------(Step 6) Model fitting
    % build model on TRAIN subs
%   fit_both = regress(train_behav, [train_sum_all, ones(no_sub-1,1)]);   % combined features

    if regress_method == 1
            fit_pos = polyfit(train_sumpos, train_behav,1);
            fit_neg = polyfit(train_sumneg, train_behav,1);
            fit_both = polyfit(train_sum_all, train_behav,1); % combined features
        
    elseif regress_method == 2
            fit_pos = regress(train_behav, [ones(no_sub-1,1), train_sumpos]);
            fit_neg = regress(train_behav, [ones(no_sub-1,1), train_sumneg]);
            fit_both = regress(train_behav, [ones(no_sub-1,1), train_sum_all]); % combined features
            
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
    
    all_test_sumpos(1,leftout) = test_sumpos;
    all_test_sumneg(1,leftout) = test_sumneg;

    if regress_method == 1
        predict.behav_pred_pos(leftout) = fit_pos(1)*test_sumpos + fit_pos(2);
        predict.behav_pred_neg(leftout) = fit_neg(1)*test_sumneg + fit_neg(2);
        predict.behav_pred_both(leftout) = fit_both(1)*test_sum_all(1) + fit_both(2);
        
    elseif regress_method == 2
        predict.behav_pred_pos(leftout) = fit_pos(1) + fit_pos(2)*test_sumpos;
        predict.behav_pred_neg(leftout) = fit_neg(1) + fit_neg(2)*test_sumneg;
        predict.behav_pred_both(leftout) = fit_both(1)+ fit_both(2)*test_sum_all(1);
            
    elseif regress_method == 3
        predict.behav_pred_pos(leftout) = fit_pos(1) + fit_pos(2)*test_sumpos;
        predict.behav_pred_neg(leftout) = fit_neg(1) + fit_neg(2)*test_sumneg;
        predict.behav_pred_both(leftout) = fit_both(1)+ fit_both(2)*test_sum_all(1);
    else
        warning('Error building linear model, please select a regression method: 1-polyfit, 2-regress, 3-robustfit');
    end
    
end

feature.all_train_sumpos = all_train_sumpos;
feature.all_train_sumneg = all_train_sumneg;
feature.all_test_sumpos = all_test_sumpos;
feature.all_test_sumneg = all_test_sumneg;
%------(Step 7) Prediction in novel subjects
% compare predicted and observed scores

[results.R_pos, results.P_pos] = corr(predict.behav_pred_pos,all_behav,'type', corr_type);
[results.R_neg, results.P_neg] = corr(predict.behav_pred_neg,all_behav,'type', corr_type);
[results.R_both, results.P_both] = corr(predict.behav_pred_both,all_behav,'type', corr_type);

% calculate Rsquared (only one predictor, use the ordinary R squared)
model_pos = fitlm(predict.behav_pred_pos,all_behav);
results.Rsquared_pos = model_pos.Rsquared.Ordinary;

model_neg = fitlm(predict.behav_pred_neg,all_behav);
results.Rsquared_neg = model_neg.Rsquared.Ordinary;

model_both = fitlm(predict.behav_pred_both,all_behav);
results.Rsquared_both = model_both.Rsquared.Ordinary;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% shared edges
pos_edges_shared = intersect(pos_edges_all{1},pos_edges_all{2});    % returns the data common to both A and B
for k = 2:size(pos_edges_all,2)
    pos_edges_shared = intersect(pos_edges_shared,pos_edges_all{k});
end

neg_edges_shared = intersect(neg_edges_all{1},neg_edges_all{2});
for k = 2:size(neg_edges_all,2)
    neg_edges_shared = intersect(neg_edges_shared,neg_edges_all{k});
end

pos_mask_shared = zeros(no_node1,no_node2);
neg_mask_shared = zeros(no_node1,no_node2);

pos_mask_shared(pos_edges_shared) = 1;
neg_mask_shared(neg_edges_shared) = 1;

% extract beta value using shared masks
    sumpos_shared = zeros(no_sub,1);
    sumneg_shared = zeros(no_sub,1);

    for aa = 1:no_sub
        sumpos_shared(aa) = sum(sum(all_mats(:,:,aa).*pos_mask_shared, 'omitnan'));
        sumneg_shared(aa) = sum(sum(all_mats(:,:,aa).*neg_mask_shared, 'omitnan'));
    end
    sumboth_shared = sumpos_shared - sumneg_shared;
    
    %all_behav = zscore(all_behav);
    %all_behav_z = zscore(all_behav);
    
    if regress_method == 1
        fit_pos_shared = polyfit(sumpos_shared, all_behav,1);
        fit_neg_shared = polyfit(sumneg_shared, all_behav,1);
        fit_both_shared = polyfit(sumboth_shared, all_behav,1);

    elseif regress_method == 2
        fit_pos_shared = regress(all_behav, [ones(no_sub,1), sumpos_shared]);
        fit_neg_shared = regress(all_behav, [ones(no_sub,1), sumneg_shared]);
        fit_both_shared = regress(all_behav, [ones(no_sub,1), sumboth_shared]);
        
    elseif regress_method == 3
        fit_pos_shared = robustfit(sumpos_shared, all_behav);
        fit_neg_shared = robustfit(sumneg_shared, all_behav);
        fit_both_shared = robustfit(sumboth_shared, all_behav);         
    else
        warning('Error building linear model, please select a regression method: 1-polyfit, 2-regress, 3-robustfit');
    end
    
ModelShared.pos_mask_shared = pos_mask_shared;
ModelShared.neg_mask_shared = neg_mask_shared;
ModelShared.fit_pos_shared = fit_pos_shared;
ModelShared.fit_neg_shared = fit_neg_shared;
ModelShared.fit_both_shared = fit_both_shared;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------- extract edges that appeared in full sample
    all_vcts = reshape(all_mats,[],size(all_mats,3));
    
    ind = isempty(all_cov);
    if ind == 0
        %all_cov = zscore(all_cov);
        %all_cov_z = zscore(all_cov);
        [r_mat_all, p_mat_all] = partialcorr(all_vcts', all_behav, all_cov, 'type', corr_type);
    elseif ind ~= 0
        [r_mat_all, p_mat_all] = corr(all_vcts', all_behav, 'type', corr_type);
    else
        warning('Error correlating the brain and the behavior, please check your data and correlation_method');
    end
    
    r_mat_all = reshape(r_mat_all, no_node1, no_node2);
    p_mat_all = reshape(p_mat_all, no_node1, no_node2);
    
    %-------(Step 4) Edge selection
    % set threshold and define masks
    
    pos_mask_all = zeros(no_node1,no_node2);
    neg_mask_all = zeros(no_node1,no_node2);
    
    pos_mask_all(r_mat_all > 0 & p_mat_all < thresh_pos) = 1;
    neg_mask_all(r_mat_all < 0 & p_mat_all < thresh_neg) = 1;
    
    % extract beta value using full sample's masks
    sumpos_all = zeros(no_sub,1);
    sumneg_all = zeros(no_sub,1);
    
    for bb = 1:no_sub
        sumpos_all(bb) = sum(sum(all_mats(:,:,bb).*pos_mask_all, 'omitnan'));
        sumneg_all(bb) = sum(sum(all_mats(:,:,bb).*neg_mask_all, 'omitnan'));
    end
    sumboth_all = sumpos_all - sumneg_all;
    
    if regress_method == 1
        fit_pos_all = polyfit(sumpos_all, all_behav,1);
        fit_neg_all = polyfit(sumneg_all, all_behav,1);
        fit_both_all = polyfit(sumboth_all, all_behav,1);

    elseif regress_method == 2
        fit_pos_all = regress(all_behav, [ones(no_sub,1), sumpos_all]);
        fit_neg_all = regress(all_behav, [ones(no_sub,1), sumneg_all]);
        fit_both_all = regress(all_behav, [ones(no_sub,1), sumboth_all]);
        
    elseif regress_method == 3
        fit_pos_all = robustfit(sumpos_all, all_behav);
        fit_neg_all = robustfit(sumneg_all, all_behav);
        fit_both_all = robustfit(sumboth_all, all_behav); 
    else
        warning('Error building linear model, please select a regression method: 1-polyfit, 2-regress, 3-robustfit');
    end
    
ModelAll.pos_mask_all = pos_mask_all;
ModelAll.neg_mask_all = neg_mask_all;
ModelAll.fit_pos_all = fit_pos_all;
ModelAll.fit_neg_all = fit_neg_all;
ModelAll.fit_both_all = fit_both_all;

end