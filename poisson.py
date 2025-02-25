'''
## try neg binomial - not working atm
X_train = sm.add_constant(X_train)
nb_model = NegativeBinomial(y_train,X_train)
nb_model.fit()
y_pred = nb_model.predict(X_test)
print(y_pred)
#mse2 = mean_squared_error(y_test,y_pred)
'''

#PLOT SHIT
'''
lambda_est = np.mean(y_labels) + 1
# Create a histogram of the count data
bins = np.arange(y_labels.min(), y_labels.max() + 2) - 0.5  # bins for integer counts
plt.hist(y_labels, bins=bins, density=True, alpha=0.6, color='gray', edgecolor='black', label='Data Histogram')

# Generate x-values for the Poisson PMF
x = np.arange(y_labels.min(), y_labels.max() + 1)

# Calculate the Poisson PMF using the estimated lambda
poisson_pmf = poisson.pmf(x, lambda_est)

# Plot the Poisson PMF
plt.plot(x, poisson_pmf, 'bo', ms=8, label='Poisson PMF')
plt.vlines(x, 0, poisson_pmf, colors='b', lw=2, alpha=0.7)

plt.xlabel("Count")
plt.ylabel("Probability")
plt.title("Count Data Histogram and Poisson PMF Overlay")
plt.legend()
plt.show()
'''