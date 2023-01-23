#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import graphviz
import arviz as az
import pymc as pm
import xarray as xr
import pytensor
import pytensor.tensor as pt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import QuantileRegressor
#from xgboostlss.model import *
#from xgboostlss.distributions.Gaussian_AutoGrad import Gaussian
#from xgboostlss.datasets.data_loader import load_simulated_data



# %%
def generate_data(seed: int, n: int=100, scale=1000, follower_dist="NB"):

    # Set seed to make the experiment repeatable
    np.random.seed(336)

    # Draw the follower numbers from the negative binomial distribution. Approximates reality
    if follower_dist == 'NB':
        followers = np.round(np.random.negative_binomial(1, 0.2, size=n))
        followers = followers*scale
    elif follower_dist == 'Uniform':
        followers = np.round(np.random.uniform(0, 25000, n))
    else:
        raise("distribution not found")

    # Add an outlier to skew the results


    time_of_week = np.random.uniform(0, 161, size=n)

    # Create a weekly seasonality and daily seasonality
    seasonality = np.sin(2*np.pi*time_of_week/161) + np.sin(2*np.pi*time_of_week/23)

    # Create synthetic "engagements" data, our target variable
    engagements = followers*0.05 + 5 # 3*seasonality + 5

    simulated_outlier = np.where(followers > 10000)[-1][0]
    #engagements[simulated_outlier] = engagements[simulated_outlier] - 400

    # Add heteroskedasctic noise term, related to the numer of followers  in total
    # Use an intercept == 1, so that the noise term is strictly positive
    noise = np.random.normal(0, followers*0.03)

    engagements = engagements + noise
    engagements[engagements < 0] = 0
    engagements[20] = 1000
    engagements[13] = 1700

    return followers, time_of_week, seasonality, engagements, np.log(followers + 1)



#%%
# Generate follower list
followers, time_of_week, seasonality, engagements, log_followers = generate_data(seed=1, n=50, follower_dist="NB")
df = pd.concat([pd.Series(followers), pd.Series(time_of_week), pd.Series(engagements), pd.Series(log_followers)], axis=1)
df = df.set_axis(["followers", "time_of_week","engagements","log_followers"], axis=1, copy=False)
print(df.head())

#%%
# Plot the variables of interest
plt.scatter(time_of_week, seasonality)
plt.show()

# Plot the distribution of the followers
plt.hist(followers)
plt.show()

# Plot the target variable distribution
plt.hist(engagements)
plt.show()

# Plot the relationship between the target variables and the number of followers
plt.scatter(followers, engagements)
plt.xlabel("# Number of Followers")
plt.ylabel("# Number of Engagements")
plt.title("Number of Post Engagements by Number of Followers")
plt.show()

plt.scatter(seasonality, engagements)
plt.show()


plt.hist(log_followers)
plt.show()

plt.scatter(log_followers, engagements)
plt.show()

plt.scatter(followers, engagements*100/followers)
plt.show()

#%%
plt.scatter(np.log(followers), np.log(engagements))
plt.xlabel("Log of the Followers")
plt.ylabel("Log of the Engagements")
plt.title("Number of Post Engagements by Number of Followers")
plt.show()




#%% 
# Start with a standard linear regression for comparison

df = sm.add_constant(df)
lm = sm.OLS(df['engagements'], df.loc[:,['const','followers']])
res = lm.fit()
print(res.summary())
y_pred = res.params[1]*df['followers'] + res.params[0]

residuals = df['engagements'] - y_pred
std = np.std(residuals)



#%% 
# Solution 2: Quantile regression
from sklearn.linear_model import QuantileRegressor

X = np.array(df.loc[:,['followers']]).reshape(-1,1)
y = np.array(df['engagements'])

quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
predictions = {}
for quantile in quantiles:
    qr = QuantileRegressor(quantile=quantile, alpha=0, solver="highs")
    y_pred = qr.fit(X, y).predict(X)
    predictions[quantile] = y_pred





#%%
# Solution 2: Model the Variance
# Define the basic linear model
with pm.Model() as model:
    X = pm.Data("x", followers, dims="obs_id", mutable=False)

    # define priors
    intercept = pm.Uniform("intercept", lower=np.min(engagements), upper=np.max(engagements))
    slope = pm.Uniform("slope", lower = -1, upper = 1)
    #sigma = pm.HalfCauchy("sigma", beta=10)

    sig_m = pm.Uniform('sigma_m', lower=0, upper=1)
    sigma = pm.Deterministic('sig', np.log(np.exp(sig_m*df['followers'] + 1)))

    # Define the mean of th enormal distribution
    mu = pm.Deterministic("mu", slope*df['followers'] + intercept, dims="obs_id")

    # Define the likelihood function
    likelihood = pm.Normal("y", mu=mu, sigma=sigma, observed=df['engagements'], dims="obs_id")

    # inference
    trace = pm.sample(1000, tune=1000, cores=1)


# %%
# Chart the model structure
pm.model_to_graphviz(model)


#%%
sigma = np.mean(np.concatenate(trace.posterior['sigma_m']))


post = az.extract(trace, num_samples=10)
x_plot = xr.DataArray(np.linspace(df['followers'].min(), df['followers'].max(), 100), dims="plot_id")
lines = post["intercept"] + post["slope"] * x_plot


lines_p95 = post["intercept"] + post["slope"] * x_plot + 2*sigma*x_plot
lines_n95 = post["intercept"] + post["slope"] * x_plot - 2*sigma*x_plot
y_OLS = res.params[0] + res.params[1]*x_plot
y_OLSn95 = res.params[0] + res.params[1]*x_plot + 2*std
y_OLSn5 = res.params[0] + res.params[1]*x_plot - 2*std

plt.scatter(df['followers'], df['engagements'], label="data")
plt.plot(x_plot, y_OLS, alpha=0.4, color="red")
plt.plot(x_plot, y_OLSn95, alpha=0.4, color="blue")
plt.plot(x_plot, y_OLSn5, alpha=0.4, color="blue")
plt.xlabel("Number of Followers")
plt.ylabel("Number of Engagements")
plt.legend(loc=0)
plt.title("Linear Model Fit - Engagements vs Followers");


#%%
# Calculate MAPE, add 1 so that we never have zero in the divisor
PE = (df['engagements'] + 1 - y_pred) / (df['engagements'] + 1)
plt.scatter(df['followers'], PE)
plt.show()

plt.hist(PE)
plt.show()

#%%
# Plot the Quantile regression results
plt.scatter(df['followers'], df['engagements'], label="data")
plt.plot(df['followers'], predictions[0.5], alpha=0.4, color="red")
plt.plot(df['followers'], predictions[0.05], alpha=0.4, color="blue")
plt.plot(df['followers'], predictions[0.95], alpha=0.4, color="blue")
plt.plot(df['followers'], predictions[0.25], alpha=0.4, color="purple")
plt.plot(df['followers'], predictions[0.75], alpha=0.4, color="purple")
plt.xlabel("Number of Followers")
plt.ylabel("Number of Engagements")
plt.legend(loc=0)
plt.title("Quantile Regression");


#%%
sigma = np.mean(np.concatenate(trace.posterior['sigma_m']))


post = az.extract(trace, num_samples=10)
x_plot = xr.DataArray(np.linspace(df['followers'].min(), df['followers'].max(), 100), dims="plot_id")
lines = np.mean(post["intercept"]) + np.mean(post["slope"]) * x_plot


lines_p95 = np.mean(post["intercept"]) + np.mean(post["slope")) * x_plot + 2*sigma*x_plot
lines_n95 = np.mean(post["intercept"]) + np.mean(post["slope"]) * x_plot - 2*sigma*x_plot
y_OLS = res.params[0] + res.params[1]*x_plot

plt.scatter(df['followers'], df['engagements'], label="data")
plt.plot(x_plot, lines.transpose(), alpha=0.4, color="red")
plt.plot(x_plot, lines_p95.transpose(), alpha=0.4, color="blue")
plt.plot(x_plot, lines_n95.transpose(), alpha=0.4, color="blue")
#plt.plot(x_plot, y_OLS, alpha=0.4, color="red")
plt.legend(loc=0)
plt.title("Bayesian Posterior Predictions");




# %%
# Plot the posterior distribution of the beta
plt.hist(np.concatenate(trace.posterior['slope']))
plt.show()


# Plot the posterior distribution of the sigma beta
plt.hist(np.concatenate(trace.posterior['sigma_m']))
plt.show()

# %%
# Solution 3, model the variance with a neural network

