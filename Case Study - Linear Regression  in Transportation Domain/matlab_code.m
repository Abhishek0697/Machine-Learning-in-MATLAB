clc;
clear;


% Read Data and select the required entries in double format

Monthly_Transportation_Statistics = readtable('Monthly_Transportation_Statistics.csv');

Months = (table2array(Monthly_Transportation_Statistics(697:876, 2)));
Spending_Airways_USA = cell2mat(table2cell(Monthly_Transportation_Statistics(697:876, 46)));
Transportation_Employment_Air_USA = cell2mat(table2cell(Monthly_Transportation_Statistics(697:876, 91)));


% Linear Regression Model
model = fitlm(Spending_Airways_USA, Transportation_Employment_Air_USA);

% Scatterplot
figure('Name', 'Question 3 - Linear Regression')
plot(model, 'MarkerEdgeColor',[0.8500 0.3250 0.0980], 'Marker','o', 'LineWidth',1.5)
title('Scatter plot of Linear Regression', 'FontSize',14)
xlabel('State and Local Government Construction Spending – Airways, from 1st Jan 2005 to 31st Dec 2019 (USD)', 'FontSize',14)
ylabel('Transportation Employment - Air Transportation, from 1st Jan 2005 to 31st Dec 2019 (units)', 'FontSize',14)

% Correlation Coefficient
c = corrcoef(Spending_Airways_USA, Transportation_Employment_Air_USA,'Rows','complete');

Spending = fitlm(datenum(Months), Spending_Airways_USA);
% Find Prediction of Spending_Airways_USA at January 1st 2021
Spending_2021 = predict(Spending, datenum('01/01/2021 12:00:00 AM'));


%  Find Prediction of Transportation_Employment_Air_USA due to Government Construction Spending – Airways at Jan 1st  2021
Transportation_Employment_Air_USA_2021 = predict(model, Spending_2021);
