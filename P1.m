clc;
clear;
close all;

%% 1. Load Data
% Load the dataset generated from Python
filename = 'nonlinear_mlp_data_seed_17772130.csv';
data = readtable(filename);

disp('--- First 5 rows of data ---');
head(data, 5)

% Extract inputs and targets
% MATLAB NN Toolbox expects [Features x Samples]
inputs = [data.x, data.y, data.z]'; 
targets = data.u';

%% Check for Duplicate Rows
num_total_rows = height(data);
unique_data = unique(data, 'rows');
num_unique_rows = height(unique_data);
num_duplicates = num_total_rows - num_unique_rows;

fprintf('Number of duplicate rows: %d\n', num_duplicates);
if num_duplicates > 0
    disp('Duplicate rows exist in the dataset.');
else
    disp('No duplicate rows found.');
end

%% Check for Anomalous/Outlier Values (Z-score method)
% مقادیری که Z-score آن‌ها بزرگتر از 3 (یا کوچکتر از -3) باشد، معمولاً غیرعادی در نظر گرفته می‌شوند.
numeric_data = data{:, :}; % تبدیل جدول به ماتریس عددی
z_scores = zscore(numeric_data);

% یافتن سطرها و ستون‌هایی که داده پرت دارند
[outlier_rows, outlier_cols] = find(abs(z_scores) > 3);

fprintf('\nNumber of anomalous values (|Z-score| > 3): %d\n', length(outlier_rows));
if ~isempty(outlier_rows)
    disp('Anomalous values detected at (row, col):');
    disp([outlier_rows, outlier_cols]);
else
    disp('No extreme anomalous values found based on Z-score method.');
end

%% Basic Statistical Summary
% گرفتن خلاصه آماری برای بررسی چشمی مقادیر min و max
disp('--- Data Summary ---');
summary(data);

%% 2. Data Exploration (Scatter Plots)
figure('Name', 'Data Exploration');
subplot(1,3,1); 
plot(inputs(1,:), targets, 'b.'); 
xlabel('X'); ylabel('U'); title('U vs X');

subplot(1,3,2); 
plot(inputs(2,:), targets, 'g.'); 
xlabel('Y'); ylabel('U'); title('U vs Y');

subplot(1,3,3); 
plot(inputs(3,:), targets, 'r.'); 
xlabel('Z'); ylabel('U'); title('U vs Z');

%% 3 & 4. Setup 5 Configurations (Hyperparameter Tuning)
% We define 5 different network architectures and training parameters
% Note: 'poslin' is ReLU in MATLAB. 'tansig' is Tanh.
% 'trainlm' is Levenberg-Marquardt (standard/fast)
% 'traingdx' is Gradient Descent with Variable Learning Rate

configs = {
    struct('name', 'M1', 'hidden', [16 8], 'tf', {{'poslin', 'poslin', 'purelin'}}, 'trainFcn', 'trainlm', 'lr', 0.01);
    struct('name', 'M2', 'hidden', [32 16], 'tf', {{'tansig', 'tansig', 'purelin'}}, 'trainFcn', 'trainlm', 'lr', 0.01);
    struct('name', 'M3', 'hidden', [32 16 8], 'tf', {{'poslin', 'poslin', 'poslin', 'purelin'}}, 'trainFcn', 'traingdx', 'lr', 0.01);
    struct('name', 'M4', 'hidden', [16 8], 'tf', {{'poslin', 'poslin', 'purelin'}}, 'trainFcn', 'traingdx', 'lr', 0.001); 
    struct('name', 'M5', 'hidden', 10, 'tf', {{'tansig', 'purelin'}}, 'trainFcn', 'trainlm', 'lr', 0.01) % Standard baseline
};

best_val_perf = inf;
best_net = [];
best_tr = [];
best_name = '';

disp('--- Starting Training for 5 Configurations ---');

for i = 1:length(configs)
    conf = configs{i};
    fprintf('Training %s...\n', conf.name);
    
    %% Create a Fitting Network using user template
    net = newff(inputs, targets, conf.hidden, conf.tf);
    
    %% Choose Input and Output Pre/Post-Processing Functions
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};
    
    %% Setup Division of Data for Training, Validation, Testing
    % Set seed for reproducibility
    rng(17772130); 
    net.divideFcn = 'dividerand'; 
    net.divideMode = 'sample'; 
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    
    %% Choose a Performance Function
    net.performFcn = 'mse'; % Mean Squared Error
    net.trainFcn = conf.trainFcn;
    
    %% Setup training parameters
    net.trainParam.showCommandLine = false;
    net.trainParam.showWindow = false; % Hidden to avoid popups during loop
    net.trainParam.epochs = 100;
    net.trainParam.goal = 1e-8;
    net.trainParam.max_fail = 15; % Early Stopping
    
    % Set learning rate if applicable for the optimizer
    if isfield(net.trainParam, 'lr')
        net.trainParam.lr = conf.lr;
    end
    
    %% Train the Network
    [net, tr] = train(net, inputs, targets);
    
    %% Recalculate Validation Performance
    valInd = tr.valInd;
    valInputs = inputs(:, valInd);
    valTargets = targets(:, valInd);
    valOutputs = net(valInputs);
    
    valPerformance = perform(net, valTargets, valOutputs);
    fprintf('  -> %s Validation MSE: %f\n', conf.name, valPerformance);
    
    % Keep track of the best model based on Validation Error
    if valPerformance < best_val_perf
        best_val_perf = valPerformance;
        best_net = net;
        best_tr = tr;
        best_name = conf.name;
    end
end

fprintf('\n*** Best Model Selected: %s with Val MSE: %f ***\n\n', best_name, best_val_perf);

%% 6. Test the Network (Best Model)
% Evaluate ONLY on Test Set
testInd = best_tr.testInd;
testInputs = inputs(:, testInd);
testTargets = targets(:, testInd);
testOutputs = best_net(testInputs);
testErrors = gsubtract(testTargets, testOutputs);

testPerformance = perform(best_net, testTargets, testOutputs); % MSE
testMAE = mean(abs(testErrors));
testRMSE = sqrt(mean(testErrors.^2));

fprintf('--- Final Evaluation on Test Set (%s) ---\n', best_name);
fprintf('Test MSE: %f\n', testPerformance);
fprintf('Test MAE: %f\n', testMAE);
fprintf('Test RMSE: %f\n', testRMSE);

% Plot Regression for Actual vs Predicted
figure('Name', 'Actual vs Predicted (Test)');
plot(testTargets, testOutputs, 'mo', 'MarkerFaceColor', 'm'); hold on;
p_min = min([testTargets, testOutputs]);
p_max = max([testTargets, testOutputs]);
plot([p_min p_max], [p_min p_max], 'k--', 'LineWidth', 2);
xlabel('Actual U'); ylabel('Predicted U');
title('Test Data: Actual vs Predicted');
grid on;

%% 7. Bonus: Controlled Sweep
% Fix two variables to their mean, sweep the third
mean_x = mean(inputs(1,:));
mean_y = mean(inputs(2,:));
mean_z = mean(inputs(3,:));

% Sweep Z (keep X and Y at mean)
z_sweep = linspace(-2, 6, 100);
sweep_inputs_z = [repmat(mean_x, 1, 100); repmat(mean_y, 1, 100); z_sweep];
sweep_outputs_z = best_net(sweep_inputs_z);

% Sweep X (keep Y and Z at mean)
x_sweep = linspace(0.5, 3.5, 100);
sweep_inputs_x = [x_sweep; repmat(mean_y, 1, 100); repmat(mean_z, 1, 100)];
sweep_outputs_x = best_net(sweep_inputs_x);

figure('Name', 'Bonus: Controlled Sweep');
subplot(1,2,1);
plot(z_sweep, sweep_outputs_z, 'r-', 'LineWidth', 2);
xlabel('Z (Sweep)'); ylabel('Predicted U');
title(sprintf('Sweep Z (X=%.2f, Y=%.2f)', mean_x, mean_y));
grid on;

subplot(1,2,2);
plot(x_sweep, sweep_outputs_x, 'b-', 'LineWidth', 2);
xlabel('X (Sweep)'); ylabel('Predicted U');
title(sprintf('Sweep X (Y=%.2f, Z=%.2f)', mean_y, mean_z));
grid on;

disp('Done! All plots and models generated.');
