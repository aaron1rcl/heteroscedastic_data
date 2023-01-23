import pandas as pd
import numpy as np

def generate_data(seed: int, n: int=100, scale=1000, follower_dist="NB", seasonal=False):
    ''' Support function to generate example data'''

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

    time_of_week = np.random.uniform(0, 161, size=n)

    # Create a weekly seasonality and daily seasonality
    seasonality = np.sin(2*np.pi*time_of_week/161) + np.sin(2*np.pi*time_of_week/23)

    if seasonal is False:
        seasonality = np.repeat(0, len(followers))
        # Create synthetic "engagements" data, our target variable
        engagements = followers*0.05 + 5 + 3*seasonality

    simulated_outlier = np.where(followers > 10000)[-1][0]
    #engagements[simulated_outlier] = engagements[simulated_outlier] - 400

    # Add heteroskedasctic noise term, related to the numer of followers  in total
    # Use an intercept == 1, so that the noise term is strictly positive
    noise = np.random.normal(0, followers*0.03)

    engagements = engagements + noise
    engagements[engagements < 0] = 0
    # Add a couple of larger datapoints to better highlight the fit differences.
    engagements[20] = 1000
    engagements[13] = 1700

    return followers, time_of_week, seasonality, engagements, np.log(followers + 1)