function [x_behav_reg,y_behav_reg,Rsquared] = Rsquared_regress_out_variable (x_behav, y_behav, covariable)

model1 = fitlm(covariable,x_behav);
x_behav_reg = model1.Residuals.('Standardized');

model2 = fitlm(covariable,y_behav);
y_behav_reg = model2.Residuals.('Standardized');

model3 = fitlm(x_behav_reg,y_behav_reg);
Rsquared = model3.Rsquared.Ordinary;

end

