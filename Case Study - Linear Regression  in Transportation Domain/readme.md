### Trend:
The U.S. Department of Transportation releases Monthly Transportation Statistics which has the latest data from across the Federal government and transportation industry. I am interested in studying the trend between the money Government spends on Airways v/s the Employment in the Air Transport Industry.

### Dataset Source:<br>
U.S Department of Transportation, Monthly Transportation Statistics:
https://data.bts.gov/Research-and-Statistics/Monthly-Transportation-Statistics/crem-w557

### Assumptions and Methodology:
1. My Hypothesis is that there exists a significant relation (positive correlation) between these two variables. i.e. More Money the Government spends on Air Transportation Industry, the Employment rate should increase. 
2. Independent Variable – State and Local Government Construction Spending – Air
3. Dependent Variable - Transportation Employment - Air Transportation
4. I have used fitlm function in MATLAB to fit a Linear Regression Model and further calculated the correlation coefficient using the MATLAB function corrcoef.

### Result:

![](https://github.com/Abhishek0697/Machine-Learning-in-MATLAB/blob/main/Case%20Study%20-%20Linear%20Regression%20%20in%20Transportation%20Domain/data/Scatterplot%20of%20Linear%20Regression.png)


1. The slope of the predicted Linear Regression Line is positive which is in line with our hypothesis that increase in one variable will lead to increase in other variable.
2. The Correlation Coefficient ρ comes out to be 0.3815.
3. Interpretation: The Correlation Coefficient suggests a positive correlation between the two variables.
4. Prediction of Employment in the Air Transport Industry on Jan 1, 2021 = 482273.8 units.
