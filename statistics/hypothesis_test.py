import scipy.stats as stats

# Hypothesis Test
"""

    - Hypothesis testing is a statistical method used to make decisions about a population based on sample data
    
    1. State the null hypothesis
       - This is a statement about the assumed current distribution of the data and acts as a default to be disproved
       
    2. State the alternative hypothesis
       - This is a statement that there is an effect or a difference, it's what the test aims to support

    3. Choose a significance level, alpha
       - alpha is the probability of rejecting the null hypothesis when it is actually true. 
       - Commonly used values are 0.05 or 0.01

    4. Collect data and calculate the test statistic
       - Data is collected from a sample, and a test statistic is calculated
       - The test statistic summarises the data in a way that allows for the testing of hypotheses

    4. Determine the P-value
       - The P-value is the probability of obtaining a test statistic at least as extreme as the one observed, assuming the null 
         hypothesis is true.
       - Alternatively, a critical value can be determined, and the decision can be made based on whether the test statistic falls in the 
         critical region.

    5. Reject or fail to reject the alternative hypothesis
       - If the P-value is less than the chosen significance level alpha, reject the null hypothesis in favor of the alternative hypothesis
       - Otherwise, fail to reject the null hypothesis.

    - Hypothesis testing helps determine whether observed data deviates significantly from what is expected under the null hypothesis
    - In the event we reject the null hypothesis, hypothesis testing does not tell us what the new value of the parameter being tested 
      should be, merely that is varies significantly from the null hypothesis
    - Smaller p-values do not imply a greater difference in the parameter values between the null and alternative hypotheses

"""

# Suppose we wanted to run a hypothesis test to see if a coin flip was fair (i.e. even chance or heads and tails)

# Define the null hypothesis proportion (fair coin)
p_null = 0.5

# Define the observed data
number_of_flips = 100
number_of_heads = 80  # Try different values

# Perform the binomial test by computing the p value which represents the chance, assuming the null hypothesis is true, that we get data
# at least as extreme as the observed data (i.e. more than number_of_heads heads)

# The alternative parameter specifies the alternative hypothesis:
# 'two-sided' means the alternative hypothesis is that the coin is not fair (p != 0.5)
binom_test = stats.binomtest(number_of_heads, number_of_flips, p_null, alternative='two-sided')

# Output the result
print(f"Number of flips: {number_of_flips}")
print(f"Number of heads: {number_of_heads}")
print(f"p-value: {binom_test.pvalue}")

# Determine the conclusion
alpha = 0.05  # significance level
if binom_test.pvalue < alpha:
    print("Reject the null hypothesis: The coin is not fair.")
else:
    print("Fail to reject the null hypothesis: The coin is fair.")



