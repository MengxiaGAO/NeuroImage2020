function [results, predict] = CPM_single_modality_validation (all_mats_valid, all_behav_valid, all_cov_valid, pos_mask, neg_mask, fit_pos_valid, fit_neg_valid, fit_both_valid, corr_type)
%% This function is to use the beta value and mask in the internal validation dataset to validate in an external validation dataset
% Input:
%       all_mats_valid:  an X*Y*Z 3D matrix
%       all_behav_valid: a Z*1 vector     
%       all_cov_valid:   a Z*M matrix (can be empty)
%       pos_mask:        an X*Y matrix with 0 and 1 value
%       neg_mask:        an X*Y matrix with 0 and 1 value
%       fit_pos_valid:   beta value of positive network model (1*2)
%       fit_neg_valid:   beta value of negative network model (1*2)
%       fit_both_valid:  beta value of combining both networks model (1*3)
%       corr_type        correlation type: 'Pearson', 'Kendall', 'Spearman'

% Output:
%       results:     predicted R and P value in validation dataset (positive, negative, combined)
%       predict:     predicted values in validation dataset (positive, negative, combined)

%%%%% The script is adopted from Xinlin Shen (2017, Nature Protocols). 
%%%%% Written by Mengxia Gao, PhD candidate of Dept. of Psychology, the University of Hong Kong.
%%%%% mengxia.gao@gmail.com, 20200310
%%%%% updated on 20200430

no_sub_valid = size(all_mats_valid,3);
predict.behav_pred_pos_valid = zeros(no_sub_valid,1);
predict.behav_pred_neg_valid = zeros(no_sub_valid,1);
predict.behav_pred_both_valid = zeros(no_sub_valid,1);

    % run model on TEST sub
    for tt = 1:no_sub_valid
    test_mat_valid = all_mats_valid(:,:,tt);
    test_pos_select_valid = test_mat_valid.*pos_mask;
    test_pos_select_valid(isnan(test_pos_select_valid)) = 0;    % turn the NaN value into zero
    test_sumpos_valid= sum(sum(test_pos_select_valid))/2;
    
    test_neg_select_valid = test_mat_valid.*neg_mask;
    test_neg_select_valid(isnan(test_neg_select_valid)) = 0;    % turn the NaN value into zero
    test_sumneg_valid = sum(sum(test_neg_select_valid))/2;
    
    test_sumboth_valid = test_sumpos_valid - test_sumneg_valid;
    
    predict.behav_pred_pos_valid(tt) = fit_pos_valid(1)*test_sumpos_valid + fit_pos_valid(2);
    predict.behav_pred_neg_valid(tt) = fit_neg_valid(1)*test_sumneg_valid + fit_neg_valid(2);
    predict.behav_pred_both_valid(tt) = fit_both_valid(1)*test_sumboth_valid + fit_both_valid(2);
    end
    
% compare the predicted and original behavioral values
ind = isempty(all_cov_valid);
if ind == 0
    [results.R_pos_valid, results.P_pos_valid] = partialcorr(predict.behav_pred_pos_valid, all_behav_valid, all_cov_valid, 'type', corr_type);
    [results.R_neg_valid, results.P_neg_valid] = partialcorr(predict.behav_pred_neg_valid, all_behav_valid, all_cov_valid, 'type', corr_type);
    [results.R_both_valid, results.P_both_valid] = partialcorr(predict.behav_pred_both_valid, all_behav_valid, all_cov_valid, 'type', corr_type);

elseif ind == 1
    [results.R_pos_valid, results.P_pos_valid] = corr(predict.behav_pred_pos_valid, all_behav_valid,'type', corr_type);
    [results.R_neg_valid, results.P_neg_valid] = corr(predict.behav_pred_neg_valid, all_behav_valid,'type', corr_type);
    [results.R_both_valid, results.P_both_valid] = corr(predict.behav_pred_both_valid, all_behav_valid,'type', corr_type);
else
    warning('Error correlating the brain and the behavior, please check your data and correlation_method');
end

cov = isempty(all_cov_valid);
if cov == 0
    [~,~,results.Rsquared_pos_valid] = Rsquared_regress_out_variable (predict.behav_pred_pos_valid, all_behav_valid, all_cov_valid);
    [~,~,results.Rsquared_neg_valid] = Rsquared_regress_out_variable (predict.behav_pred_neg_valid, all_behav_valid, all_cov_valid);
    [~,~,results.Rsquared_both_valid] = Rsquared_regress_out_variable (predict.behav_pred_both_valid, all_behav_valid, all_cov_valid);
    
elseif cov == 1
    model_pos = fitlm(predict.behav_pred_pos_valid, all_behav_valid);
    results.Rsquared_pos_valid = model_pos.Rsquared.Ordinary;

    model_neg = fitlm(predict.behav_pred_neg_valid, all_behav_valid);
    results.Rsquared_neg_valid = model_neg.Rsquared.Ordinary;

    model_both = fitlm(predict.behav_pred_both_valid, all_behav_valid);
    results.Rsquared_both_valid = model_both.Rsquared.Ordinary;
else
    warning('Error calculate the R squared of the data');
end

end