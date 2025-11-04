'''
network that predicts a volatility score for the 7 days following time t, the ground truth is calculated as follows:

 μ = 0.5 * Σᵢ₌₁⁷ [ (log(S_i / S_{i-1})^2 ] + 0.5 * Σᵢ₌₁⁷ [(log(S_i/S_0))^2]
 
 the first term is daily log return
 second term is cumulative log displacement from the day we are predicting
 
 that mu along with a standard variation of that prediction is going to be what the model predicts so that we can later
 use it in a buy/sell/hold signal calculation for straddle options
 
 the model will take as input (all of which will be time series over past 30 days, i.e. there will be 30 t's from which we will
 calculate these values):
 
 log return: log(C_t/C_{t-1}) shape: (B, C, 30, 1)
 
 squared return: (log(C_t/C_{t-1}))^2 shape: (B, C, 30, 1)
 
 hl_range_pct: (Hight_t - Low_t)/close_{t-1} shape: (B, C, 30, 1)
 
 close open return: ln(C_t/O_t) shape: (B, C, 30, 1)
 
 parkinson var: 1/(4 ln(2)) (ln(High_t/Low_t))^2 shape: (B, C, 30, 1)
 
 turnover ***: ln(Volume_t * close_t): shape: (B, C, 30, 1)
 
 gap return: ln(O_t/C_{t-1}) shape: (B, C, 30, 1)
 
 garman klass: 1/2 (ln(High_t/Low_t))^2 - (2 ln(2))(ln(C_t/O_t))^2 shape: (B, C, 30, 1)
 
 past 5 day sum of squared return: Σ [(log(S_{t-i}/S_{t-i-1}))^2] shape: (B, C, 30, 1)
 
 past 5 day sum of squared displacement: Σ [(log(S_{t-i}/S_{t-6}))^2] shape: (B, C, 30, 1)
 
 past 10 day sum of squared return, shape: (B, C, 30, 1)
 
 past 10 day sum of squared displacement, shape: (B, C, 30, 1)
 
 past 20 day sum of squared return, shape: (B, C, 30, 1)
 
 past 20 day sum of squared displacement, shape: (B, C, 30, 1)
'''