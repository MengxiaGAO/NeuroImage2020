function [R_pos_repeat, R_neg_repeat, R_both_repeat, Rsquared_pos_repeat, Rsquared_neg_repeat, Rsquared_both_repeat] = CPM_n_folds (nfolds, no_repeat, thresh_pos, thresh_neg, all_mats, all_behav, all_cov, corr_type, regress_method)
%% This function is to use n-folds to calculate the R value of CPM analysis (behavioral scores normalized within each loop)
% Input:
%       nfolds:         n-folds (N, a number)
%       no_repeat:      the number of repeating (N, a number)
%       all_mats:       an X*Y*Z 3D matrix
%       all_behav:      a Z*1 vector
%       all_cov:        a Z*M vector contains different covariables (can be empty)
%       corr_type:      correlation type: 'Pearson', 'Kendall', 'Spearman'
%       regress_method: 1-polyfit, 2-regress

% Output:
%       R_pos_repeat:   r values of n-fold in N repeating calculation (positive)
%       R_neg_repeat:   r values of n-fold in N repeating calculation (negative)
%       R_both_repeat:  r values of n-fold in N repeating calculation (combined)

%%%%% The script is adopted from Xinlin Shen (2017, Nature Protocols). 
%%%%% Written by Mengxia Gao, PhD candidate of Dept. of Psychology, the University of Hong Kong.
%%%%% mengxia.gao@gmail.com, 20200509

no_node1 = size(all_mats,1);
no_node2 = size(all_mats,2);  
nsubs=size(all_mats,3);
ksample=round(nsubs/nfolds);

R_pos_repeat = zeros(nfolds,no_repeat);
R_neg_repeat = zeros(nfolds,no_repeat);
R_both_repeat = zeros(nfolds,no_repeat);

Rsquared_pos_repeat = zeros(nfolds,no_repeat);
Rsquared_neg_repeat = zeros(nfolds,no_repeat);
Rsquared_both_repeat = zeros(nfolds,no_repeat);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for repeat = 1:no_repeat
    fprintf('\n Performing repeat %d out of %d', repeat, no_repeat);
    
% get the testing and training set
randinds=randperm(nsubs);

list = cell(1,nfolds);
    list{1} = randinds(1:ksample);
    for num = 2:(nfolds-1)
        list{num} = randinds((num-1)*ksample+1:ksample*num);
    end
    list{nfolds} = randinds((nfolds-1)*ksample+1:nsubs);

R_pos_list = zeros(nfolds,1);
R_neg_list = zeros(nfolds,1);
R_both_list = zeros(nfolds,1);

Rsquared_pos_list = zeros(nfolds,1);
Rsquared_neg_list = zeros(nfolds,1);
Rsquared_both_list = zeros(nfolds,1);

for n_list = 1:size(list,2)
        testinds=list{n_list};
        traininds=setdiff(randinds,testinds);
        
    % Assign x and y data to train and test set   
    train_mats = all_mats(:,:,traininds);
    train_vcts = reshape(train_mats,[],size(train_mats,3));
    train_behav = all_behav(traininds);
    train_behav = zscore(train_behav);
    
    test_mats = all_mats(:,:,testinds);
%     test_vcts = reshape(test_mats,[],size(test_mats,3));
    test_behav = all_behav(testinds);
    test_behav = zscore(test_behav);
%     test_cov = all_cov(testinds,:);
   
    ind = isempty(all_cov);
    if ind == 0
        train_cov = all_cov(traininds,:);
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
    train_sumpos = zeros(size(traininds,2),1);
    train_sumneg = zeros(size(traininds,2),1);
    
    for ss = 1:size(train_sumpos)
        pos_select = train_mats(:,:,ss).*pos_mask;
        pos_select(isnan(pos_select)) = 0;    % turn the NaN value into zero
        train_sumpos(ss) = sum(sum(pos_select))/2;
        
        neg_select = train_mats(:,:,ss).*neg_mask;
        neg_select(isnan(neg_select)) = 0;    % turn the NaN value into zero
        train_sumneg(ss) = sum(sum(neg_select))/2;
    end
    
    train_sum_all = train_sumpos - train_sumneg;
    %-------(Step 6) Model fitting
    % build model on TRAIN subs
    if regress_method == 1
            fit_pos = polyfit(train_sumpos, train_behav,1);
            fit_neg = polyfit(train_sumneg, train_behav,1);
            fit_both = polyfit(train_sum_all, train_behav,1);
        
    elseif regress_method == 2
            fit_pos = regress(train_behav, [train_sumpos, ones(size(traininds,2),1)]);
            fit_neg = regress(train_behav, [train_sumneg, ones(size(traininds,2),1)]);
            fit_both = regress(train_behav, [train_sum_all, ones(size(traininds,2),1)]);
    else
        warning('Error building linear model, please select a regression method: 1-polyfit, 2-regress');
    end

    % run model on TEST sub 
    behav_pred_pos = zeros(size(testinds,2),1);
    behav_pred_neg = zeros(size(testinds,2),1);
    behav_pred_both = zeros(size(testinds,2),1);
    
    for kk = 1: size(testinds,2)
    test_pos_select = test_mats(:,:,kk).*pos_mask;
    test_pos_select(isnan(test_pos_select)) = 0;    % turn the NaN value into zero
    test_sumpos = sum(sum(test_pos_select))/2;
    
    test_neg_select = test_mats(:,:,kk).*neg_mask;
    test_neg_select(isnan(test_neg_select)) = 0;    % turn the NaN value into zero
    test_sumneg = sum(sum(test_neg_select))/2;
    
    test_sum_all = test_sumpos - test_sumneg;
    
    behav_pred_both(kk) = fit_both(1)*test_sum_all(1) + fit_both(2);
    behav_pred_pos(kk) = fit_pos(1)*test_sumpos + fit_pos(2);
    behav_pred_neg(kk) = fit_neg(1)*test_sumneg + fit_neg(2);
    end
    
    
    [R_pos, P_pos] = corr(behav_pred_pos,test_behav,'type', corr_type);
    [R_neg, P_neg] = corr(behav_pred_neg,test_behav,'type', corr_type);
    [R_both, P_both] = corr(behav_pred_both,test_behav,'type', corr_type);
    
    R_pos_list(n_list) = R_pos;
    R_neg_list(n_list) = R_neg;
    R_both_list(n_list) = R_both;

    model_pos = fitlm(behav_pred_pos,test_behav);
    Rsquared_pos = model_pos.Rsquared.Ordinary;

    model_neg = fitlm(behav_pred_neg,test_behav);
    Rsquared_neg = model_neg.Rsquared.Ordinary;

    model_both = fitlm(behav_pred_both,test_behav);
    Rsquared_both = model_both.Rsquared.Ordinary;
    
    Rsquared_pos_list(n_list) = Rsquared_pos;
    Rsquared_neg_list(n_list) = Rsquared_neg;
    Rsquared_both_list(n_list) = Rsquared_both;
    
    clear train_sumpos train_sumneg train_sum_all test_sumpos test_sumneg test_sum_all behav_pred_pos behav_pred_neg behav_pred_both
end

    R_pos_repeat(:,repeat) = R_pos_list;
    R_neg_repeat(:,repeat) = R_neg_list;
    R_both_repeat(:,repeat) = R_both_list;
    
    Rsquared_pos_repeat(:,repeat) = Rsquared_pos_list;
    Rsquared_neg_repeat(:,repeat) = Rsquared_neg_list;
    Rsquared_both_repeat(:,repeat) = Rsquared_both_list;
    
end
    
    
end