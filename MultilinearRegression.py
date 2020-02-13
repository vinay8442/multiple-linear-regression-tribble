# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
cars = pd.read_csv("C:\\Users\\vinay goud\\Documents\\DATA SCIENCE\\Codes With Examples - Python-20191016\\Multiple Linear Regression\\cars.csv")
cars
# to get top 6 rows
cars.head() # to get top n rows use cars.head(10)

# Correlation matrix 
cars.corr()

# we see there exists High collinearity between input variables especially between
# [Hp & SP] , [VOL,WT] so there exists collinearity problem
 
# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars)

# columns names
cars.columns

# pd.tools.plotting.scatter_matrix(cars); -> also used for plotting all in one graph
                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
# Preparing model                  
ml1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit() # regression model
    
# Getting coefficients of variables               
ml1.params

# Summary
ml1.summary()
# p-values for WT,VOL are more than 0.05 and also we know that [WT,VOL] has high correlation value 

# preparing model based only on Volume
ml_v=smf.ols('MPG~VOL',data = cars).fit()  
ml_v.summary()
# p-value <0.05 .. It is significant 

# Preparing model based only on WT
ml_w=smf.ols('MPG~WT',data = cars).fit()  
ml_w.summary()

# Preparing model based only on WT & VOL
ml_wv=smf.ols('MPG~WT+VOL',data = cars).fit()  
ml_wv.summary()
# Both coefficients p-value became insignificant... 
# So there may be a chance of considering only one among VOL & WT

# Checking whether data has any influential values 
# influence index plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)
# index 76 is showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals

cars_new=cars.drop(cars.index[[76]])
cars_new
# Preparing model                  
ml_new = smf.ols('MPG~WT+VOL+HP+SP',data = cars_new).fit()    

# Getting coefficients of variables        
ml_new.params

# Summary
ml_new.summary()

# Confidence values 99%
print(ml_new.conf_int(0.01)) # 99% confidence level


# Predicted values of MPG 
mpg_pred = ml_new.predict(cars_new[['WT','VOL','HP','SP']])
mpg_pred

cars_new.head()
# calculating VIF's values of independent variables
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars_new).fit().rsquared  
    rsq_hp

vif_hp = 1/(1-rsq_hp) # HP  VIF value
vif_hp

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars_new).fit().rsquared  
vif_wt = 1/(1-rsq_wt) # WT  VIF value
vif_wt

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars_new).fit().rsquared  
vif_vol = 1/(1-rsq_vol) # VOL VIF value
vif_vol
rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars_new).fit().rsquared  
vif_sp = 1/(1-rsq_sp) # SP  VIF value
vif_sp
           # Storing vif values in a data frame
d1 = {'Variables':pd.Series(['Hp','WT','VOL','SP']),'VIF':pd.Series([vif_hp,vif_wt,vif_vol,vif_sp])}
Vif_frame = pd.DataFrame(d1,columns=['Variables','VIF'])  
Vif_frame
# As weight is having higher VIF value, we are not going to include this prediction model
# Added varible plot 
#sm.graphics.plot_partregress_grid(ml_new)
# added varible plot for weight is not showing any significance 

# final model
final_ml= smf.ols('MPG~VOL+SP+HP',data = cars_new).fit()

# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)
final_ml.params
final_ml.summary()
# As we can see that r-squared value has increased from 0.810 to 0.812.
